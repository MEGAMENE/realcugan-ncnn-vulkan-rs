use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, Rgb32FImage, RgbImage,
    Rgba, Rgba32FImage, RgbaImage,
};
use realcugan_ncnn_vulkan_rs::{
    RealCugan, RealCuganConfig, RealCuganError, RealCuganGpuType, RealCuganModelType,
    RealCuganNoise, RealCuganScale, RealCuganSyncGap, RealCuganTtaLevel,
};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("realcugan-ncnn-vulkan")
        .join("models")
}

fn cpu_config(
    model: RealCuganModelType,
    scale: RealCuganScale,
    noise: RealCuganNoise,
) -> RealCuganConfig {
    let mut cfg = RealCuganConfig::new(models_dir());
    cfg.gpuid = -1;
    cfg.model = model;
    cfg.scale = scale;
    cfg.noise = noise;
    cfg
}

fn gpu_config(
    gpuid: i32,
    model: RealCuganModelType,
    scale: RealCuganScale,
    noise: RealCuganNoise,
) -> RealCuganConfig {
    let mut cfg = RealCuganConfig::new(models_dir());
    cfg.gpuid = gpuid;
    cfg.model = model;
    cfg.scale = scale;
    cfg.noise = noise;
    cfg
}

fn make_patterned_rgb8(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbImage::new(w, h);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let r = ((x.wrapping_mul(17).wrapping_add(y.wrapping_mul(5))) % 256) as u8;
        let g = ((y.wrapping_mul(13).wrapping_add(40)) % 256) as u8;
        let b = (((x ^ y).wrapping_mul(9).wrapping_add(90)) % 256) as u8;
        *px = Rgb([r, g, b]);
    }
    DynamicImage::ImageRgb8(img)
}

fn make_patterned_rgba8(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let r = ((x.wrapping_mul(19).wrapping_add(20)) % 256) as u8;
        let g = ((y.wrapping_mul(11).wrapping_add(60)) % 256) as u8;
        let b = ((x.wrapping_add(y).wrapping_mul(7)) % 256) as u8;
        let band = (x * 4) / w.max(1);
        let a = match band {
            0 => 0,
            1 => 64,
            2 => 128,
            _ => 255,
        };
        *px = Rgba([r, g, b, a]);
    }
    DynamicImage::ImageRgba8(img)
}

/// High-contrast `[0, 255]` step / lineart pattern: white `[255, 255, 255]` background with a
/// centered solid black `[0, 0, 0]` square `[8..24, 8..24]`. Catches `is_pro_model = false`
/// instability where `models-pro` 2x/3x weights blow up and invert black interiors to white.
fn make_pro_high_contrast_pattern() -> DynamicImage {
    let mut img = RgbImage::from_pixel(32, 32, Rgb([255, 255, 255]));
    for y in 8..24 {
        for x in 8..24 {
            img.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn assert_alpha_ramp_preserved(output: &DynamicImage, label: &str) {
    assert!(
        output.color().has_alpha(),
        "{label}: expected alpha channel in output, got {:?}",
        output.color()
    );
    let rgba = output.to_rgba8();
    let w = rgba.width();
    let mid_y = rgba.height() / 2;
    let sample_a = [
        rgba.get_pixel(w / 8, mid_y)[3],
        rgba.get_pixel(w * 3 / 8, mid_y)[3],
        rgba.get_pixel(w * 5 / 8, mid_y)[3],
        rgba.get_pixel(w * 7 / 8, mid_y)[3],
    ];
    let expected = [0i16, 64, 128, 255];
    for (i, (&actual, &exp)) in sample_a.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual as i16 - exp).abs() <= 16,
            "{label}: alpha band {i} expected ~{exp}, got {actual} (samples={sample_a:?})"
        );
    }
    assert!(
        sample_a[0] < sample_a[1] && sample_a[1] < sample_a[2] && sample_a[2] < sample_a[3],
        "{label}: alpha ramp not strictly increasing: {sample_a:?}"
    );
}

fn assert_pro_fidelity(engine: &RealCugan, scale: RealCuganScale, label: &str) {
    let s = scale.as_u32();

    // 1. High-contrast [0, 255] step/lineart test
    let pattern = make_pro_high_contrast_pattern();
    let out_pattern = engine
        .proc_image(&pattern)
        .unwrap_or_else(|e| panic!("{label}: high-contrast pattern failed: {e}"))
        .to_rgb8();
    assert_eq!(out_pattern.width(), 32 * s);
    assert_eq!(out_pattern.height(), 32 * s);

    let black_px = out_pattern.get_pixel(16 * s, 16 * s);
    assert!(
        black_px[0] < 20 && black_px[1] < 20 && black_px[2] < 20,
        "{label}: black interior distorted (pro scaling broken?): {black_px:?}"
    );

    let white_px = out_pattern.get_pixel(2 * s, 2 * s);
    assert!(
        white_px[0] > 235 && white_px[1] > 235 && white_px[2] > 235,
        "{label}: white exterior distorted: {white_px:?}"
    );

    // 2. Dark, bright, saturated, and midtone flat patches
    let patches: [([u8; 3], &str); 5] = [
        ([5, 5, 5], "dark"),
        ([250, 250, 250], "bright"),
        ([240, 30, 30], "red"),
        ([30, 30, 240], "blue"),
        ([230, 190, 160], "skin-midtone"),
    ];
    for (rgb, patch_name) in patches {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb(rgb)));
        let out = engine
            .proc_image(&img)
            .unwrap_or_else(|e| panic!("{label} ({patch_name}): inference failed: {e}"))
            .to_rgb8();
        let px = out.get_pixel(8 * s, 8 * s);
        for c in 0..3 {
            let diff = (px[c] as i16 - rgb[c] as i16).abs();
            assert!(
                diff <= 4,
                "{label} ({patch_name}): channel {c} expected {}, got {} (pixel={px:?})",
                rgb[c],
                px[c]
            );
        }
    }
}

fn mean_abs_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum();
    sum as f64 / a.len() as f64
}

#[test]
fn test_invalid_configurations_and_validation() {
    // 1. All invalid (model, scale, noise) matrix entries
    let invalid_combos = [
        (RealCuganModelType::Nose, RealCuganScale::X2, RealCuganNoise::Conservative),
        (RealCuganModelType::Nose, RealCuganScale::X2, RealCuganNoise::Denoise1x),
        (RealCuganModelType::Nose, RealCuganScale::X2, RealCuganNoise::Denoise2x),
        (RealCuganModelType::Nose, RealCuganScale::X2, RealCuganNoise::Denoise3x),
        (RealCuganModelType::Nose, RealCuganScale::X3, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Nose, RealCuganScale::X4, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Pro, RealCuganScale::X4, RealCuganNoise::Conservative),
        (RealCuganModelType::Pro, RealCuganScale::X4, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Pro, RealCuganScale::X4, RealCuganNoise::Denoise3x),
        (RealCuganModelType::Pro, RealCuganScale::X2, RealCuganNoise::Denoise1x),
        (RealCuganModelType::Pro, RealCuganScale::X2, RealCuganNoise::Denoise2x),
        (RealCuganModelType::Pro, RealCuganScale::X3, RealCuganNoise::Denoise1x),
        (RealCuganModelType::Pro, RealCuganScale::X3, RealCuganNoise::Denoise2x),
        (RealCuganModelType::Se, RealCuganScale::X3, RealCuganNoise::Denoise1x),
        (RealCuganModelType::Se, RealCuganScale::X3, RealCuganNoise::Denoise2x),
        (RealCuganModelType::Se, RealCuganScale::X4, RealCuganNoise::Denoise1x),
        (RealCuganModelType::Se, RealCuganScale::X4, RealCuganNoise::Denoise2x),
    ];

    for (model, scale, noise) in invalid_combos {
        let cfg = cpu_config(model, scale, noise);
        let res = RealCugan::new(cfg);
        assert!(
            matches!(
                res,
                Err(RealCuganError::UnsupportedModelConfig {
                    model: m,
                    scale: s,
                    noise: n
                }) if m == model && s == scale && n == noise
            ),
            "expected UnsupportedModelConfig for ({model:?}, {scale:?}, {noise:?}), got {res:?}"
        );
    }

    // 2. Invalid tile sizes (1..=31 and > i32::MAX)
    for bad_tile in [1u32, 16, 31, (i32::MAX as u32) + 1, u32::MAX] {
        let mut cfg = cpu_config(
            RealCuganModelType::Se,
            RealCuganScale::X2,
            RealCuganNoise::Conservative,
        );
        cfg.tile_size = bad_tile;
        assert!(
            matches!(RealCugan::new(cfg), Err(RealCuganError::InvalidTileSize(v)) if v == bad_tile),
            "expected InvalidTileSize({bad_tile})"
        );
    }

    // 3. Invalid GPU device IDs
    for bad_gpu in [-2, -100, 9999] {
        let mut cfg = cpu_config(
            RealCuganModelType::Se,
            RealCuganScale::X2,
            RealCuganNoise::Conservative,
        );
        cfg.gpuid = bad_gpu;
        assert!(
            matches!(RealCugan::new(cfg), Err(RealCuganError::InvalidGpuDevice(id)) if id == bad_gpu),
            "expected InvalidGpuDevice({bad_gpu})"
        );
    }

    // 4. Nonexistent models directory
    let mut nonexistent_cfg = RealCuganConfig::new("path_that_does_not_exist_realcugan");
    nonexistent_cfg.gpuid = -1;
    assert!(matches!(
        RealCugan::new(nonexistent_cfg),
        Err(RealCuganError::ModelNotFound(_))
    ));

    // 5. Interior NUL in path
    let mut nul_cfg = RealCuganConfig::new("bad\0models_dir");
    nul_cfg.gpuid = -1;
    assert!(matches!(
        RealCugan::new(nul_cfg),
        Err(RealCuganError::InvalidPath(_))
    ));

    // 6. Directory-as-model-file rejected by `.is_file()`
    let temp_root = std::env::temp_dir().join(format!(
        "realcugan_dir_as_file_test_{}",
        std::process::id()
    ));
    let fake_param_dir = temp_root.join("models-se").join("up2x-conservative.param");
    std::fs::create_dir_all(&fake_param_dir).expect("create temp test directory");
    let mut dir_cfg = RealCuganConfig::new(&temp_root);
    dir_cfg.gpuid = -1;
    let dir_res = RealCugan::new(dir_cfg);
    let _ = std::fs::remove_dir_all(&temp_root);
    assert!(
        matches!(dir_res, Err(RealCuganError::ModelNotFound(_))),
        "directory matching .param path must be rejected as ModelNotFound, got {dir_res:?}"
    );

    // 7. Zero dimensions rejected in proc_image
    let engine = RealCugan::new(cpu_config(
        RealCuganModelType::Nose,
        RealCuganScale::X2,
        RealCuganNoise::NoDenoise,
    ))
    .expect("init CPU nose engine");

    for (w, h) in [(0, 0), (0, 16), (16, 0)] {
        let img = DynamicImage::ImageRgb8(RgbImage::new(w, h));
        assert_eq!(
            engine.proc_image(&img),
            Err(RealCuganError::InvalidDimensions(w, h))
        );
    }
}

#[test]
fn test_static_queries() {
    assert_eq!(RealCugan::default_prepadding(RealCuganScale::X2), 18);
    assert_eq!(RealCugan::default_prepadding(RealCuganScale::X3), 14);
    assert_eq!(RealCugan::default_prepadding(RealCuganScale::X4), 19);

    assert!(RealCugan::should_use_cpu(-1));
    assert_eq!(RealCugan::get_heap_budget(-1), 0);
    assert_eq!(RealCugan::get_gpu_device_type(-1), None);
    assert_eq!(RealCugan::get_gpu_device_type(9999), None);
    assert_eq!(
        RealCugan::compute_auto_tilesize(
            -1,
            RealCuganScale::X2,
            RealCuganTtaLevel::X1,
            NonZeroU32::MIN,
        ),
        Some(400)
    );

    let gpu_count = RealCugan::get_gpu_count();
    let default_gpu = RealCugan::get_default_gpu_index();
    if gpu_count > 0 {
        assert!((0..gpu_count).contains(&default_gpu));
        let dtype = RealCugan::get_gpu_device_type(0);
        assert!(dtype.is_some());
        if !RealCugan::should_use_cpu(0) {
            assert!(RealCugan::get_heap_budget(0) > 0);
            let ts = RealCugan::compute_auto_tilesize(
                0,
                RealCuganScale::X2,
                RealCuganTtaLevel::X1,
                NonZeroU32::MIN,
            );
            assert!(matches!(ts, Some(v) if (32..=512).contains(&v)));
            let ts_tta = RealCugan::compute_auto_tilesize(
                0,
                RealCuganScale::X2,
                RealCuganTtaLevel::X8,
                NonZeroU32::MIN,
            );
            assert!(matches!(ts_tta, Some(v) if (32..=400).contains(&v)));
            assert!(matches!(
                dtype,
                Some(
                    RealCuganGpuType::Discrete
                        | RealCuganGpuType::Integrated
                        | RealCuganGpuType::Virtual
                        | RealCuganGpuType::Unknown(_)
                )
            ));
        }
    } else {
        assert_eq!(default_gpu, -1);
    }
}

#[test]
fn test_cpu_rgb_bgr_channel_order() {
    let engine = RealCugan::new(cpu_config(
        RealCuganModelType::Se,
        RealCuganScale::X2,
        RealCuganNoise::Conservative,
    ))
    .expect("init CPU SE 2x");

    // Asymmetric red: R=240, G=20, B=10 -> must NOT swap R and B
    let red_in = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([240, 20, 10])));
    let red_out = engine.proc_image(&red_in).expect("CPU red").to_rgb8();
    let rpx = red_out.get_pixel(16, 16);
    assert!(
        rpx[0] > 200 && rpx[2] < 50,
        "CPU RGB channel order swapped on red: {rpx:?}"
    );

    // Asymmetric blue: R=10, G=20, B=240
    let blue_in = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([10, 20, 240])));
    let blue_out = engine.proc_image(&blue_in).expect("CPU blue").to_rgb8();
    let bpx = blue_out.get_pixel(16, 16);
    assert!(
        bpx[2] > 200 && bpx[0] < 50,
        "CPU RGB channel order swapped on blue: {bpx:?}"
    );
}

#[test]
fn test_cpu_formats_and_alpha_ramp_preservation() {
    let engine = RealCugan::new(cpu_config(
        RealCuganModelType::Nose,
        RealCuganScale::X2,
        RealCuganNoise::NoDenoise,
    ))
    .expect("init CPU Nose 2x");

    let (w, h) = (32u32, 16u32);

    // 1. Rgba8 varying alpha ramp
    let rgba8_out = engine
        .proc_image(&make_patterned_rgba8(w, h))
        .expect("CPU Rgba8");
    assert_eq!((rgba8_out.width(), rgba8_out.height()), (w * 2, h * 2));
    assert_alpha_ramp_preserved(&rgba8_out, "CPU Rgba8");

    // 2. La8 varying alpha ramp
    let mut la8 = GrayAlphaImage::new(w, h);
    for (x, _y, px) in la8.enumerate_pixels_mut() {
        let a = match (x * 4) / w {
            0 => 0,
            1 => 64,
            2 => 128,
            _ => 255,
        };
        *px = LumaA([160, a]);
    }
    let la8_out = engine
        .proc_image(&DynamicImage::ImageLumaA8(la8))
        .expect("CPU La8");
    assert_alpha_ramp_preserved(&la8_out, "CPU La8");

    // 3. Rgba16 varying alpha ramp
    let mut rgba16: ImageBuffer<Rgba<u16>, Vec<u16>> = ImageBuffer::new(w, h);
    for (x, _y, px) in rgba16.enumerate_pixels_mut() {
        let a = match (x * 4) / w {
            0 => 0u16,
            1 => 16448,
            2 => 32896,
            _ => 65535,
        };
        *px = Rgba([50000, 20000, 10000, a]);
    }
    let rgba16_out = engine
        .proc_image(&DynamicImage::ImageRgba16(rgba16))
        .expect("CPU Rgba16");
    assert_alpha_ramp_preserved(&rgba16_out, "CPU Rgba16");

    // 4. La16 varying alpha ramp
    let mut la16: ImageBuffer<LumaA<u16>, Vec<u16>> = ImageBuffer::new(w, h);
    for (x, _y, px) in la16.enumerate_pixels_mut() {
        let a = match (x * 4) / w {
            0 => 0u16,
            1 => 16448,
            2 => 32896,
            _ => 65535,
        };
        *px = LumaA([40000, a]);
    }
    let la16_out = engine
        .proc_image(&DynamicImage::ImageLumaA16(la16))
        .expect("CPU La16");
    assert_alpha_ramp_preserved(&la16_out, "CPU La16");

    // 5. Rgba32F varying alpha ramp (specifically tests that Rgba32F preserves alpha!)
    let mut rgba32f = Rgba32FImage::new(w, h);
    for (x, _y, px) in rgba32f.enumerate_pixels_mut() {
        let a = match (x * 4) / w {
            0 => 0.0f32,
            1 => 64.0 / 255.0,
            2 => 128.0 / 255.0,
            _ => 1.0,
        };
        *px = Rgba([0.8, 0.2, 0.1, a]);
    }
    let rgba32f_out = engine
        .proc_image(&DynamicImage::ImageRgba32F(rgba32f))
        .expect("CPU Rgba32F");
    assert_alpha_ramp_preserved(&rgba32f_out, "CPU Rgba32F");

    // 6. Opaque formats: L8, L16, Rgb16, Rgb32F -> Rgb8 output without alpha
    let l8 = DynamicImage::ImageLuma8(GrayImage::from_pixel(16, 16, Luma([140])));
    let l8_out = engine.proc_image(&l8).expect("CPU L8");
    assert!(!l8_out.color().has_alpha());
    let l8_px = l8_out.to_rgb8().get_pixel(16, 16).0;
    assert!((l8_px[0] as i16 - 140).abs() <= 10);

    let l16: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_pixel(16, 16, Luma([35000]));
    let l16_out = engine
        .proc_image(&DynamicImage::ImageLuma16(l16))
        .expect("CPU L16");
    assert!(!l16_out.color().has_alpha());
    assert_eq!((l16_out.width(), l16_out.height()), (32, 32));

    let rgb16: ImageBuffer<Rgb<u16>, Vec<u16>> =
        ImageBuffer::from_pixel(16, 16, Rgb([60000, 10000, 5000]));
    let rgb16_out = engine
        .proc_image(&DynamicImage::ImageRgb16(rgb16))
        .expect("CPU Rgb16");
    assert!(!rgb16_out.color().has_alpha());
    let rgb16_px = rgb16_out.to_rgb8().get_pixel(16, 16).0;
    assert!(rgb16_px[0] > 200 && rgb16_px[2] < 50);

    let rgb32f = Rgb32FImage::from_pixel(16, 16, Rgb([0.9, 0.1, 0.05]));
    let rgb32f_out = engine
        .proc_image(&DynamicImage::ImageRgb32F(rgb32f))
        .expect("CPU Rgb32F");
    assert!(!rgb32f_out.color().has_alpha());
    let rgb32f_px = rgb32f_out.to_rgb8().get_pixel(16, 16).0;
    assert!(rgb32f_px[0] > 200 && rgb32f_px[2] < 50);
}

#[test]
fn test_cpu_pro_fidelity_2x_and_3x() {
    for scale in [RealCuganScale::X2, RealCuganScale::X3] {
        let engine = RealCugan::new(cpu_config(
            RealCuganModelType::Pro,
            scale,
            RealCuganNoise::Conservative,
        ))
        .unwrap_or_else(|e| panic!("init CPU Pro {scale:?} failed: {e}"));
        assert_eq!(engine.scale(), scale);
        assert_pro_fidelity(&engine, scale, &format!("CPU Pro {scale:?} FP32 TTA1"));
    }
}

#[test]
fn test_cpu_model_matrix_and_edge_dimensions() {
    let small = make_patterned_rgb8(12, 12);

    // Every supported (model, scale, noise) combination on CPU
    let matrix = [
        (RealCuganModelType::Nose, RealCuganScale::X2, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Se, RealCuganScale::X2, RealCuganNoise::Conservative),
        (RealCuganModelType::Se, RealCuganScale::X2, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Se, RealCuganScale::X2, RealCuganNoise::Denoise1x),
        (RealCuganModelType::Se, RealCuganScale::X2, RealCuganNoise::Denoise2x),
        (RealCuganModelType::Se, RealCuganScale::X2, RealCuganNoise::Denoise3x),
        (RealCuganModelType::Se, RealCuganScale::X3, RealCuganNoise::Conservative),
        (RealCuganModelType::Se, RealCuganScale::X3, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Se, RealCuganScale::X3, RealCuganNoise::Denoise3x),
        (RealCuganModelType::Se, RealCuganScale::X4, RealCuganNoise::Conservative),
        (RealCuganModelType::Se, RealCuganScale::X4, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Se, RealCuganScale::X4, RealCuganNoise::Denoise3x),
        (RealCuganModelType::Pro, RealCuganScale::X2, RealCuganNoise::Conservative),
        (RealCuganModelType::Pro, RealCuganScale::X2, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Pro, RealCuganScale::X2, RealCuganNoise::Denoise3x),
        (RealCuganModelType::Pro, RealCuganScale::X3, RealCuganNoise::Conservative),
        (RealCuganModelType::Pro, RealCuganScale::X3, RealCuganNoise::NoDenoise),
        (RealCuganModelType::Pro, RealCuganScale::X3, RealCuganNoise::Denoise3x),
    ];

    for (model, scale, noise) in matrix {
        // Use VeryRough even for Nose to verify Layer 1 gap0 auto-fallback to syncgap=0
        let mut cfg = cpu_config(model, scale, noise);
        cfg.sync_gap = RealCuganSyncGap::VeryRough;
        let engine = RealCugan::new(cfg)
            .unwrap_or_else(|e| panic!("CPU init ({model:?}, {scale:?}, {noise:?}) failed: {e}"));
        let out = engine
            .proc_image(&small)
            .unwrap_or_else(|e| panic!("CPU proc ({model:?}, {scale:?}, {noise:?}) failed: {e}"));
        let s = scale.as_u32();
        assert_eq!((out.width(), out.height()), (12 * s, 12 * s));
    }

    // Edge dimensions including 1x1, thin strips, odd dims, and non-square TTA X8
    let mut tta8_cfg = cpu_config(
        RealCuganModelType::Nose,
        RealCuganScale::X2,
        RealCuganNoise::NoDenoise,
    );
    tta8_cfg.tta_level = RealCuganTtaLevel::X8;
    let tta8_engine = RealCugan::new(tta8_cfg).expect("init CPU Nose TTA8");

    for (w, h) in [(1, 1), (1, 17), (17, 1), (15, 23), (13, 29)] {
        let img = make_patterned_rgb8(w, h);
        let out = tta8_engine
            .proc_image(&img)
            .unwrap_or_else(|e| panic!("CPU edge dim {w}x{h} TTA8 failed: {e}"));
        assert_eq!((out.width(), out.height()), (w * 2, h * 2));
    }
}

#[test]
fn test_cpu_multitile_syncgap_and_concurrent_send_sync() {
    // 64x64 at tile_size=32 -> 2x2 grid (tests VeryRough -> Rough downgrade)
    // 97x97 at tile_size=32 -> 4x4 grid (tests full VeryRough stride-2 sync)
    let img_64 = make_patterned_rgb8(64, 64);
    let img_97 = make_patterned_rgb8(97, 97);

    for sync_gap in [
        RealCuganSyncGap::None,
        RealCuganSyncGap::Accurate,
        RealCuganSyncGap::Rough,
        RealCuganSyncGap::VeryRough,
    ] {
        let mut cfg = cpu_config(
            RealCuganModelType::Se,
            RealCuganScale::X2,
            RealCuganNoise::Conservative,
        );
        cfg.tile_size = 32;
        cfg.sync_gap = sync_gap;
        let engine = RealCugan::new(cfg).expect("init CPU multitile");

        let out_64 = engine.proc_image(&img_64).expect("CPU 64x64 multitile");
        assert_eq!((out_64.width(), out_64.height()), (128, 128));

        if sync_gap == RealCuganSyncGap::VeryRough || sync_gap == RealCuganSyncGap::Rough {
            let out_97 = engine.proc_image(&img_97).expect("CPU 97x97 multitile");
            assert_eq!((out_97.width(), out_97.height()), (194, 194));
        }
    }

    // Concurrent multi-thread Send + Sync stress test on CPU via Arc<RealCugan>
    let mut cfg = cpu_config(
        RealCuganModelType::Se,
        RealCuganScale::X2,
        RealCuganNoise::Conservative,
    );
    cfg.num_threads = NonZeroU32::new(1);
    let engine = Arc::new(RealCugan::new(cfg).expect("init shared CPU engine"));

    let mut handles = Vec::new();
    for tid in 0..4u8 {
        let shared = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            let color = Rgb([
                200u8.saturating_sub(tid * 40),
                30 + tid * 35,
                20 + tid * 50,
            ]);
            let input = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, color));
            let output = shared.proc_image(&input).expect("concurrent CPU proc").to_rgb8();
            assert_eq!((output.width(), output.height()), (32, 32));
            let px = output.get_pixel(16, 16);
            for c in 0..3 {
                assert!(
                    (px[c] as i16 - color[c] as i16).abs() <= 10,
                    "thread {tid} CPU channel {c} mismatch: expected {}, got {}",
                    color[c],
                    px[c]
                );
            }
        }));
    }
    for h in handles {
        h.join().expect("CPU worker panicked");
    }
}

#[test]
fn test_gpu_comprehensive_if_available() {
    let gpu_count = RealCugan::get_gpu_count();
    println!("Available Vulkan GPU count: {gpu_count}");
    if gpu_count <= 0 {
        println!("No Vulkan GPU available; skipping GPU integration suite");
        return;
    }

    let gpuid = RealCugan::get_default_gpu_index().max(0);
    println!(
        "Using GPU {gpuid} (type={:?}, heap_budget={} MiB, should_use_cpu={})",
        RealCugan::get_gpu_device_type(gpuid),
        RealCugan::get_heap_budget(gpuid),
        RealCugan::should_use_cpu(gpuid)
    );

    // 1. Asymmetric RGB/BGR order on GPU
    let se_2x = RealCugan::new(gpu_config(
        gpuid,
        RealCuganModelType::Se,
        RealCuganScale::X2,
        RealCuganNoise::Conservative,
    ))
    .expect("init GPU SE 2x");
    assert_eq!(se_2x.scale(), RealCuganScale::X2);

    let red_in = DynamicImage::ImageRgb8(RgbImage::from_pixel(16, 16, Rgb([240, 20, 10])));
    let red_out = se_2x.proc_image(&red_in).expect("GPU red").to_rgb8();
    let rpx = red_out.get_pixel(16, 16);
    assert!(
        rpx[0] > 200 && rpx[2] < 50,
        "GPU RGB channel order swapped on red: {rpx:?}"
    );

    // 2. Alpha ramp preservation on GPU (Rgba8 and Rgba32F)
    let rgba8_out = se_2x
        .proc_image(&make_patterned_rgba8(32, 16))
        .expect("GPU Rgba8");
    assert_alpha_ramp_preserved(&rgba8_out, "GPU Rgba8");

    let mut rgba32f = Rgba32FImage::new(32, 16);
    for (x, _y, px) in rgba32f.enumerate_pixels_mut() {
        let a = match (x * 4) / 32 {
            0 => 0.0f32,
            1 => 64.0 / 255.0,
            2 => 128.0 / 255.0,
            _ => 1.0,
        };
        *px = Rgba([0.85, 0.15, 0.1, a]);
    }
    let rgba32f_out = se_2x
        .proc_image(&DynamicImage::ImageRgba32F(rgba32f))
        .expect("GPU Rgba32F");
    assert_alpha_ramp_preserved(&rgba32f_out, "GPU Rgba32F");

    // 3. 2x & 3x Pro high-contrast + color fidelity across FP32/FP16 & TTA 1/2/4/8
    for scale in [RealCuganScale::X2, RealCuganScale::X3] {
        for &(tta, fp16) in &[
            (RealCuganTtaLevel::X1, false),
            (RealCuganTtaLevel::X2, false),
            (RealCuganTtaLevel::X4, false),
            (RealCuganTtaLevel::X8, false),
            (RealCuganTtaLevel::X8, true),
        ] {
            let mut cfg = gpu_config(gpuid, RealCuganModelType::Pro, scale, RealCuganNoise::Conservative);
            cfg.tta_level = tta;
            cfg.sync_gap = RealCuganSyncGap::Accurate;
            cfg.use_fp16_arithmetic = fp16;
            let engine = RealCugan::new(cfg)
                .unwrap_or_else(|e| panic!("init GPU Pro {scale:?} {tta:?} fp16={fp16}: {e}"));
            assert_pro_fidelity(
                &engine,
                scale,
                &format!("GPU Pro {scale:?} {tta:?} fp16={fp16}"),
            );
        }
    }

    // 4. Multi-tile SyncGap modes (97x97 -> 4x4 grid, 64x64 -> 2x2 downgrade) & auto tile_size=0 with concurrent_jobs=1 and 4
    let img_64 = make_patterned_rgb8(64, 64);
    let img_97 = make_patterned_rgb8(97, 97);
    for sync_gap in [
        RealCuganSyncGap::None,
        RealCuganSyncGap::Accurate,
        RealCuganSyncGap::Rough,
        RealCuganSyncGap::VeryRough,
    ] {
        let mut cfg = gpu_config(
            gpuid,
            RealCuganModelType::Se,
            RealCuganScale::X2,
            RealCuganNoise::Conservative,
        );
        cfg.tile_size = 32;
        cfg.sync_gap = sync_gap;
        let engine = RealCugan::new(cfg).expect("init GPU multitile");
        let out_64 = engine.proc_image(&img_64).expect("GPU 64x64 multitile");
        assert_eq!((out_64.width(), out_64.height()), (128, 128));
        let out_97 = engine.proc_image(&img_97).expect("GPU 97x97 multitile");
        assert_eq!((out_97.width(), out_97.height()), (194, 194));
    }

    for jobs in [1u32, 4] {
        let mut cfg = gpu_config(
            gpuid,
            RealCuganModelType::Se,
            RealCuganScale::X2,
            RealCuganNoise::Conservative,
        );
        cfg.tile_size = 0;
        cfg.concurrent_jobs = NonZeroU32::new(jobs).unwrap();
        let engine = RealCugan::new(cfg).expect("init GPU auto tile_size=0");
        let out = engine.proc_image(&img_97).expect("GPU auto tile_size proc");
        assert_eq!((out.width(), out.height()), (194, 194));
    }

    // 5. Edge dimensions on GPU (including non-square TTA X8) & Nose gap0 fallback
    let mut nose_cfg = gpu_config(
        gpuid,
        RealCuganModelType::Nose,
        RealCuganScale::X2,
        RealCuganNoise::NoDenoise,
    );
    nose_cfg.sync_gap = RealCuganSyncGap::VeryRough;
    nose_cfg.tta_level = RealCuganTtaLevel::X8;
    let nose_gpu = RealCugan::new(nose_cfg).expect("init GPU Nose TTA8");
    for (w, h) in [(1, 1), (1, 17), (17, 1), (15, 23), (13, 29)] {
        let out = nose_gpu
            .proc_image(&make_patterned_rgb8(w, h))
            .unwrap_or_else(|e| panic!("GPU edge dim {w}x{h} failed: {e}"));
        assert_eq!((out.width(), out.height()), (w * 2, h * 2));
    }

    // 6. Concurrent Arc<RealCugan> Send + Sync stress test on GPU (4 threads, concurrent_jobs = 4)
    let mut shared_cfg = gpu_config(
        gpuid,
        RealCuganModelType::Se,
        RealCuganScale::X2,
        RealCuganNoise::Conservative,
    );
    shared_cfg.concurrent_jobs = NonZeroU32::new(4).unwrap();
    let shared_gpu = Arc::new(RealCugan::new(shared_cfg).expect("init shared GPU engine"));

    let mut handles = Vec::new();
    for tid in 0..4u8 {
        let engine = Arc::clone(&shared_gpu);
        handles.push(thread::spawn(move || {
            let color = Rgb([
                220u8.saturating_sub(tid * 45),
                25 + tid * 40,
                30 + tid * 50,
            ]);
            let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(24, 24, color));
            for _ in 0..2 {
                let out = engine.proc_image(&img).expect("concurrent GPU proc").to_rgb8();
                assert_eq!((out.width(), out.height()), (48, 48));
                let px = out.get_pixel(24, 24);
                for c in 0..3 {
                    assert!(
                        (px[c] as i16 - color[c] as i16).abs() <= 10,
                        "GPU thread {tid} channel {c} mismatch: expected {}, got {}",
                        color[c],
                        px[c]
                    );
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("GPU worker panicked");
    }

    // 7. CPU vs GPU parity across 2x/3x/4x and Se/Pro (RGB and RGBA, MAE <= 2.0/channel)
    let parity_rgb = make_patterned_rgb8(24, 24);
    let parity_rgba = make_patterned_rgba8(24, 24);
    let parity_cases = [
        (RealCuganModelType::Se, RealCuganScale::X2, RealCuganNoise::Conservative),
        (RealCuganModelType::Se, RealCuganScale::X3, RealCuganNoise::Conservative),
        (RealCuganModelType::Se, RealCuganScale::X4, RealCuganNoise::Conservative),
        (RealCuganModelType::Pro, RealCuganScale::X2, RealCuganNoise::Conservative),
        (RealCuganModelType::Pro, RealCuganScale::X3, RealCuganNoise::Conservative),
    ];

    for (model, scale, noise) in parity_cases {
        let cpu_eng = RealCugan::new(cpu_config(model, scale, noise)).expect("parity CPU init");
        let gpu_eng = RealCugan::new(gpu_config(gpuid, model, scale, noise)).expect("parity GPU init");

        let cpu_rgb = cpu_eng.proc_image(&parity_rgb).expect("parity CPU RGB").to_rgb8();
        let gpu_rgb = gpu_eng.proc_image(&parity_rgb).expect("parity GPU RGB").to_rgb8();
        let mae_rgb = mean_abs_error(cpu_rgb.as_raw(), gpu_rgb.as_raw());
        assert!(
            mae_rgb <= 2.0,
            "CPU vs GPU RGB parity MAE too high for ({model:?}, {scale:?}): {mae_rgb:.4}"
        );

        let cpu_rgba = cpu_eng.proc_image(&parity_rgba).expect("parity CPU RGBA").to_rgba8();
        let gpu_rgba = gpu_eng.proc_image(&parity_rgba).expect("parity GPU RGBA").to_rgba8();
        let mae_rgba = mean_abs_error(cpu_rgba.as_raw(), gpu_rgba.as_raw());
        assert!(
            mae_rgba <= 2.0,
            "CPU vs GPU RGBA parity MAE too high for ({model:?}, {scale:?}): {mae_rgba:.4}"
        );
    }
}
