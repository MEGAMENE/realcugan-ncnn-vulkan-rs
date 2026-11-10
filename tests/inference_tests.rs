use std::path::PathBuf;
use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use realcugan_ncnn_vulkan_rs::{RealCugan, RealCuganError, RealCuganModelType};

fn get_models_dir() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("src")
        .join("realcugan-ncnn-vulkan")
        .join("models")
        .to_str()
        .unwrap()
        .to_string()
}

#[test]
fn test_invalid_parameters() {
    let models_dir = get_models_dir();

    // Invalid TTA level
    let res = RealCugan::new_with_options(
        -1,
        -1,
        2,
        RealCuganModelType::Se,
        400,
        3,
        3, // invalid: must be 1, 2, 4, 8
        false,
        1,
        models_dir.clone(),
    );
    assert!(matches!(res, Err(RealCuganError::InvalidTtaLevel(3))));

    // Invalid scale
    let res = RealCugan::new_with_options(
        -1,
        -1,
        5, // invalid scale
        RealCuganModelType::Se,
        400,
        3,
        1,
        false,
        1,
        models_dir.clone(),
    );
    assert!(matches!(res, Err(RealCuganError::UnsupportedScale(5))));

    // Nonexistent model path
    let res = RealCugan::new_with_options(
        -1,
        -1,
        2,
        RealCuganModelType::Se,
        400,
        3,
        1,
        false,
        1,
        "invalid_path_that_does_not_exist".to_string(),
    );
    assert!(matches!(res, Err(RealCuganError::ModelNotFound(_))));
}

#[test]
fn test_cpu_inference_rgb() {
    let models_dir = get_models_dir();
    let cugan = RealCugan::new_with_options(
        -1, // CPU
        -1, // conservative
        2,  // 2x
        RealCuganModelType::Se,
        400,
        3,
        1, // TTA 1x
        false,
        2,
        models_dir,
    ).expect("Failed to initialize RealCugan on CPU");

    assert_eq!(cugan.scale(), 2);

    // Create a 16x16 test RGB image
    let mut img = RgbImage::new(16, 16);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([(x * 15) as u8, (y * 15) as u8, 128]);
    }
    let dynamic_img = DynamicImage::ImageRgb8(img);

    let output = cugan.proc_image(&dynamic_img).expect("CPU inference failed");
    assert_eq!(output.width(), 32);
    assert_eq!(output.height(), 32);
}

#[test]
fn test_cpu_inference_rgba() {
    let models_dir = get_models_dir();
    let cugan = RealCugan::new_with_options(
        -1, // CPU
        -1,
        2,
        RealCuganModelType::Se,
        400,
        3,
        1,
        false,
        2,
        models_dir,
    ).expect("Failed to initialize RealCugan on CPU");

    // Create a 16x16 test RGBA image with varying alpha
    let mut img = RgbaImage::new(16, 16);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgba([(x * 15) as u8, (y * 15) as u8, 128, 200]);
    }
    let dynamic_img = DynamicImage::ImageRgba8(img);

    let output = cugan.proc_image_cpu(&dynamic_img).expect("CPU RGBA inference failed");
    assert_eq!(output.width(), 32);
    assert_eq!(output.height(), 32);
}

#[test]
fn test_backwards_compatible_new() {
    let models_dir = get_models_dir();
    let cugan = RealCugan::new(
        -1, // CPU
        -1,
        2,
        RealCuganModelType::Se,
        400,
        3,
        false, // tta_mode = false -> tta_level = 1
        1,
        models_dir,
    ).expect("Failed to initialize RealCugan via backwards-compatible new()");

    let img = DynamicImage::ImageRgb8(RgbImage::new(8, 8));
    let output = cugan.proc_image(&img).expect("Inference failed");
    assert_eq!(output.width(), 16);
    assert_eq!(output.height(), 16);
}

#[test]
fn test_gpu_inference_if_available() {
    let gpu_count = RealCugan::get_gpu_count();
    println!("Available GPU count: {}", gpu_count);
    if gpu_count <= 0 {
        println!("No Vulkan GPU device available, skipping GPU test");
        return;
    }

    let models_dir = get_models_dir();

    // Test GPU inference with default options (TTA 1x)
    let cugan = RealCugan::new_with_options(
        0,  // GPU 0
        -1, // conservative
        2,  // 2x
        RealCuganModelType::Se,
        0,  // auto tile size
        3,
        1,  // TTA 1x
        false,
        1,
        models_dir.clone(),
    ).expect("Failed to initialize RealCugan on GPU");

    assert_eq!(cugan.scale(), 2);

    let mut img = RgbImage::new(24, 24);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([(x * 10) as u8, (y * 10) as u8, ((x + y) * 5) as u8]);
    }
    let dynamic_img = DynamicImage::ImageRgb8(img);

    let output = cugan.proc_image(&dynamic_img).expect("GPU inference failed");
    assert_eq!(output.width(), 48);
    assert_eq!(output.height(), 48);

    // Test TTA 2x, 4x, 8x on GPU
    for tta_level in [2, 4, 8] {
        let cugan_tta = RealCugan::new_with_options(
            0,
            -1,
            2,
            RealCuganModelType::Se,
            0,
            3,
            tta_level,
            false,
            1,
            models_dir.clone(),
        ).expect(&format!("Failed to initialize RealCugan on GPU with TTA {}", tta_level));

        let output_tta = cugan_tta.proc_image(&dynamic_img).expect("TTA GPU inference failed");
        assert_eq!(output_tta.width(), 48);
        assert_eq!(output_tta.height(), 48);
    }

    // Test Pro model (verifying pro postproc shader and pro math normalization)
    let cugan_pro = RealCugan::new_with_options(
        0,
        -1,
        2,
        RealCuganModelType::Pro,
        0,
        3,
        1,
        false,
        1,
        models_dir.clone(),
    ).expect("Failed to initialize RealCugan Pro model on GPU");
    let output_pro = cugan_pro.proc_image(&dynamic_img).expect("Pro model GPU inference failed");
    assert_eq!(output_pro.width(), 48);
    assert_eq!(output_pro.height(), 48);

    // Test FP16 arithmetic option
    let cugan_fp16 = RealCugan::new_with_options(
        0,
        -1,
        2,
        RealCuganModelType::Se,
        0,
        3,
        1,
        true, // use_fp16_arithmetic
        1,
        models_dir.clone(),
    ).expect("Failed to initialize RealCugan with FP16 arithmetic on GPU");
    let output_fp16 = cugan_fp16.proc_image(&dynamic_img).expect("FP16 arithmetic GPU inference failed");
    assert_eq!(output_fp16.width(), 48);
    assert_eq!(output_fp16.height(), 48);

    // Test 3x and 4x scale models
    for scale in [3, 4] {
        let cugan_scale = RealCugan::new_with_options(
            0,
            -1,
            scale,
            RealCuganModelType::Se,
            0,
            3,
            1,
            false,
            1,
            models_dir.clone(),
        ).expect(&format!("Failed to initialize RealCugan {}x scale on GPU", scale));
        let out = cugan_scale.proc_image(&dynamic_img).expect("Scale inference failed");
        assert_eq!(out.width(), 24 * scale);
        assert_eq!(out.height(), 24 * scale);
    }
}
