use std::convert::TryFrom;
use std::ffi::CString;
use std::fmt;

use image::{DynamicImage, RgbImage, RgbaImage};
use libc::{c_char, c_int, c_uchar, c_uint, c_void};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealCuganError {
    ModelNotFound(String),
    InvalidGpuDevice(i32),
    LoadFailed(String, String),
    InvalidDimensions(u32, u32),
    ProcessFailed(i32),
    UnsupportedScale(u32),
    InvalidTtaLevel(u32),
}

impl fmt::Display for RealCuganError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(p) => write!(f, "Real-CUGAN model file not found: {}", p),
            Self::InvalidGpuDevice(id) => write!(f, "Invalid GPU device id: {}", id),
            Self::LoadFailed(p, m) => write!(f, "Failed to load Real-CUGAN model from {} and {}", p, m),
            Self::InvalidDimensions(w, h) => write!(f, "Invalid image dimensions: {}x{}", w, h),
            Self::ProcessFailed(code) => write!(f, "Real-CUGAN processing failed with code: {}", code),
            Self::UnsupportedScale(s) => write!(f, "Unsupported Real-CUGAN scale: {}", s),
            Self::InvalidTtaLevel(lvl) => write!(f, "Invalid TTA level (must be 1, 2, 4, or 8): {}", lvl),
        }
    }
}

impl std::error::Error for RealCuganError {}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RealCuganModelType {
    Nose,
    Pro,
    Se,
}

#[repr(C)]
#[derive(Debug)]
pub struct Image {
    pub data: *const c_uchar,
    pub w: c_int,
    pub h: c_int,
    pub c: c_int,
}

#[cfg(not(headless_stub))]
extern "C" {
    fn realcugan_init(
        gpuid: c_int,
        tta_level: c_int,
        num_threads: c_int,
        noise: c_int,
        scale: c_int,
        tilesize: c_int,
        prepadding: c_int,
        sync_gap: c_int,
        use_fp16_arithmetic: bool,
    ) -> *mut c_void;

    fn realcugan_init_gpu_instance();

    fn realcugan_get_gpu_count() -> c_int;

    fn realcugan_destroy_gpu_instance();

    fn realcugan_load(realcugan: *mut c_void, param_path: *const c_char, model_path: *const c_char) -> c_int;

    fn realcugan_process(
        realcugan: *mut c_void,
        in_image: *const Image,
        out_image: *const Image,
        mat_ptr: *mut *mut c_void,
    ) -> c_int;

    fn realcugan_process_cpu(
        realcugan: *mut c_void,
        in_image: *const Image,
        out_image: *const Image,
        mat_ptr: *mut *mut c_void,
    ) -> c_int;

    fn realcugan_get_heap_budget(gpuid: c_int) -> c_uint;

    fn realcugan_get_gpu_device_type(gpuid: c_int) -> c_int;

    fn realcugan_calculate_tile_size(gpuid: c_int, scale: c_int, prepadding: c_int) -> c_int;

    #[allow(dead_code)]
    fn realcugan_free_image(mat_ptr: *mut c_void);

    fn realcugan_free(realcugan: *mut c_void);
}

#[cfg(headless_stub)]
unsafe fn realcugan_init(
    _gpuid: c_int,
    _tta_level: c_int,
    _num_threads: c_int,
    _noise: c_int,
    _scale: c_int,
    _tilesize: c_int,
    _prepadding: c_int,
    _sync_gap: c_int,
    _use_fp16_arithmetic: bool,
) -> *mut c_void {
    std::ptr::NonNull::<u8>::dangling().as_ptr() as *mut c_void
}
#[cfg(headless_stub)]
unsafe fn realcugan_init_gpu_instance() {}
#[cfg(headless_stub)]
unsafe fn realcugan_get_gpu_count() -> c_int { 1 }
#[cfg(headless_stub)]
unsafe fn realcugan_destroy_gpu_instance() {}
#[cfg(headless_stub)]
unsafe fn realcugan_load(_realcugan: *mut c_void, _param_path: *const c_char, _model_path: *const c_char) -> c_int { 0 }
#[cfg(headless_stub)]
unsafe fn realcugan_process(
    _realcugan: *mut c_void,
    _in_image: *const Image,
    _out_image: *const Image,
    _mat_ptr: *mut *mut c_void,
) -> c_int { 0 }
#[cfg(headless_stub)]
unsafe fn realcugan_process_cpu(
    _realcugan: *mut c_void,
    _in_image: *const Image,
    _out_image: *const Image,
    _mat_ptr: *mut *mut c_void,
) -> c_int { 0 }
#[cfg(headless_stub)]
unsafe fn realcugan_get_heap_budget(_gpuid: c_int) -> c_uint { 16384 }
#[cfg(headless_stub)]
unsafe fn realcugan_get_gpu_device_type(_gpuid: c_int) -> c_int { 2 }
#[cfg(headless_stub)]
unsafe fn realcugan_calculate_tile_size(_gpuid: c_int, _scale: c_int, _prepadding: c_int) -> c_int { 400 }
#[cfg(headless_stub)]
unsafe fn realcugan_free_image(_mat_ptr: *mut c_void) {}
#[cfg(headless_stub)]
unsafe fn realcugan_free(_realcugan: *mut c_void) {}

pub struct RealCugan {
    realcugan: *mut c_void,
    scale: u32,
}

unsafe impl Send for RealCugan {}

impl RealCugan {
    pub fn new(
        gpuid: i32,
        noise: i32,
        scale: u32,
        model: RealCuganModelType,
        tile_size: u32,
        sync_gap: u32,
        tta_mode: bool,
        num_threads: i32,
        models_path: String,
    ) -> Result<Self, RealCuganError> {
        let tta_level = if tta_mode { 8 } else { 1 };
        Self::new_with_options(
            gpuid,
            noise,
            scale,
            model,
            tile_size,
            sync_gap,
            tta_level,
            false,
            num_threads,
            models_path,
        )
    }

    pub fn new_with_options(
        gpuid: i32,
        noise: i32,
        scale: u32,
        model: RealCuganModelType,
        tile_size: u32,
        sync_gap: u32,
        tta_level: u32,
        use_fp16_arithmetic: bool,
        num_threads: i32,
        models_path: String,
    ) -> Result<Self, RealCuganError> {
        if tta_level != 1 && tta_level != 2 && tta_level != 4 && tta_level != 8 {
            return Err(RealCuganError::InvalidTtaLevel(tta_level));
        }

        let prepadding = match scale {
            2 => 18,
            3 => 14,
            4 => 19,
            _ => return Err(RealCuganError::UnsupportedScale(scale)),
        };

        let sync_gap = if model == RealCuganModelType::Nose { 0 } else { sync_gap };
        let (model, noise) = match (model, scale, noise) {
            // 4x Pro does not exist in Real-CUGAN, fallback to SE
            (RealCuganModelType::Pro, 4, n) => {
                log::warn!("4x scale is not supported for pro model, falling back to models-se");
                (RealCuganModelType::Se, n)
            }
            // 3x and 4x only have conservative (-1), no-denoise (0), and denoise3x (3)
            (m, s, 1 | 2) if s >= 3 => {
                log::warn!("denoise level {} is not available for scale {}, falling back to denoise 3x", noise, s);
                (m, 3)
            }
            (m, _, n) => (m, n),
        };

        let model_dir = match model {
            RealCuganModelType::Nose => "models-nose",
            RealCuganModelType::Pro => "models-pro",
            RealCuganModelType::Se => "models-se",
        };

        let (model_path, param_path) = if noise == -1 {
            (
                format!("{}/{}/up{}x-conservative.bin", models_path, model_dir, scale),
                format!("{}/{}/up{}x-conservative.param", models_path, model_dir, scale),
            )
        } else if noise == 0 {
            (
                format!("{}/{}/up{}x-no-denoise.bin", models_path, model_dir, scale),
                format!("{}/{}/up{}x-no-denoise.param", models_path, model_dir, scale),
            )
        } else {
            (
                format!("{}/{}/up{}x-denoise{}x.bin", models_path, model_dir, scale, noise),
                format!("{}/{}/up{}x-denoise{}x.param", models_path, model_dir, scale, noise),
            )
        };

        if !std::path::Path::new(&param_path).exists() {
            return Err(RealCuganError::ModelNotFound(param_path));
        }
        if !std::path::Path::new(&model_path).exists() {
            return Err(RealCuganError::ModelNotFound(model_path));
        }

        let param_path_cstr = CString::new(param_path.clone())
            .map_err(|_| RealCuganError::ModelNotFound(param_path.clone()))?;
        let model_path_cstr = CString::new(model_path.clone())
            .map_err(|_| RealCuganError::ModelNotFound(model_path.clone()))?;

        unsafe {
            realcugan_init_gpu_instance();
            let gpu_count = realcugan_get_gpu_count() as i32;
            if gpuid < -1 || gpuid >= gpu_count {
                realcugan_destroy_gpu_instance();
                return Err(RealCuganError::InvalidGpuDevice(gpuid));
            }

            let tile_size = if tile_size == 0 {
                if gpuid == -1 {
                    400
                } else {
                    realcugan_calculate_tile_size(gpuid, scale as c_int, prepadding) as u32
                }
            } else {
                tile_size
            };

            let realcugan = realcugan_init(
                gpuid,
                tta_level as c_int,
                num_threads,
                noise,
                scale as i32,
                tile_size as i32,
                prepadding,
                sync_gap as i32,
                use_fp16_arithmetic,
            );
            if realcugan.is_null() {
                realcugan_destroy_gpu_instance();
                return Err(RealCuganError::LoadFailed(param_path, model_path));
            }

            let ret = realcugan_load(realcugan, param_path_cstr.as_ptr(), model_path_cstr.as_ptr());
            if ret != 0 {
                realcugan_free(realcugan);
                return Err(RealCuganError::LoadFailed(param_path, model_path));
            }

            Ok(Self {
                realcugan,
                scale,
            })
        }
    }

    pub fn scale(&self) -> u32 {
        self.scale
    }

    pub fn get_gpu_count() -> i32 {
        unsafe {
            realcugan_init_gpu_instance();
            let count = realcugan_get_gpu_count();
            realcugan_destroy_gpu_instance();
            count
        }
    }

    pub fn get_heap_budget(gpuid: i32) -> u32 {
        unsafe {
            realcugan_init_gpu_instance();
            let budget = realcugan_get_heap_budget(gpuid);
            realcugan_destroy_gpu_instance();
            budget
        }
    }

    pub fn get_gpu_device_type(gpuid: i32) -> i32 {
        unsafe {
            realcugan_init_gpu_instance();
            let dtype = realcugan_get_gpu_device_type(gpuid);
            realcugan_destroy_gpu_instance();
            dtype
        }
    }

    pub fn calculate_tile_size(gpuid: i32, scale: u32, prepadding: i32) -> u32 {
        unsafe {
            realcugan_init_gpu_instance();
            let size = realcugan_calculate_tile_size(gpuid, scale as c_int, prepadding);
            realcugan_destroy_gpu_instance();
            size as u32
        }
    }

    pub fn proc_image(&self, image: &DynamicImage) -> Result<DynamicImage, RealCuganError> {
        self.process_internal(image, false)
    }

    pub fn proc_image_cpu(&self, image: &DynamicImage) -> Result<DynamicImage, RealCuganError> {
        self.process_internal(image, true)
    }

    fn process_internal(&self, image: &DynamicImage, force_cpu: bool) -> Result<DynamicImage, RealCuganError> {
        // Zero-copy borrow for standard 8-bit RGB/RGBA images
        enum PixelData<'a> {
            Borrowed(&'a [u8]),
            Owned(Vec<u8>),
        }

        let (pixel_data, channels) = match image.color() {
            image::ColorType::Rgb8 => (PixelData::Borrowed(image.as_bytes()), 3),
            image::ColorType::Rgba8 => (PixelData::Borrowed(image.as_bytes()), 4),
            image::ColorType::L8 => (PixelData::Owned(image.to_rgb8().into_raw()), 3),
            image::ColorType::La8 => (PixelData::Owned(image.to_rgba8().into_raw()), 4),
            // 16-bit depth normalization: map [0, 65535] -> [0, 255]
            image::ColorType::Rgb16 | image::ColorType::L16 => (PixelData::Owned(image.to_rgb8().into_raw()), 3),
            image::ColorType::Rgba16 | image::ColorType::La16 => (PixelData::Owned(image.to_rgba8().into_raw()), 4),
            _ => (PixelData::Owned(image.to_rgb8().into_raw()), 3),
        };

        let raw_slice = match &pixel_data {
            PixelData::Borrowed(slice) => *slice,
            PixelData::Owned(vec) => vec.as_slice(),
        };

        let in_w = i32::try_from(image.width())
            .map_err(|_| RealCuganError::InvalidDimensions(image.width(), image.height()))?;
        let in_h = i32::try_from(image.height())
            .map_err(|_| RealCuganError::InvalidDimensions(image.width(), image.height()))?;

        if in_w <= 0 || in_h <= 0 {
            return Err(RealCuganError::InvalidDimensions(image.width(), image.height()));
        }

        let expected_in_len = (in_w as usize)
            .checked_mul(in_h as usize)
            .and_then(|wh| wh.checked_mul(channels as usize))
            .ok_or_else(|| RealCuganError::InvalidDimensions(image.width(), image.height()))?;

        if raw_slice.len() != expected_in_len {
            return Err(RealCuganError::InvalidDimensions(image.width(), image.height()));
        }

        let in_buffer = Image {
            data: raw_slice.as_ptr() as *const c_uchar,
            w: in_w,
            h: in_h,
            c: i32::from(channels),
        };

        let out_w = in_w
            .checked_mul(self.scale as i32)
            .ok_or_else(|| RealCuganError::InvalidDimensions(image.width(), image.height()))?;
        let out_h = in_h
            .checked_mul(self.scale as i32)
            .ok_or_else(|| RealCuganError::InvalidDimensions(image.width(), image.height()))?;
        let length = (out_w as usize)
            .checked_mul(out_h as usize)
            .and_then(|wh| wh.checked_mul(channels as usize))
            .ok_or_else(|| RealCuganError::InvalidDimensions(out_w as u32, out_h as u32))?;

        let mut out_bytes: Vec<u8> = vec![0u8; length];

        let out_buffer = Image {
            data: out_bytes.as_mut_ptr() as *mut c_uchar,
            w: out_w,
            h: out_h,
            c: i32::from(channels),
        };

        let mut mat_ptr = std::ptr::null_mut();
        let ret = unsafe {
            if force_cpu {
                realcugan_process_cpu(
                    self.realcugan,
                    &in_buffer as *const Image,
                    &out_buffer as *const Image,
                    &mut mat_ptr,
                )
            } else {
                realcugan_process(
                    self.realcugan,
                    &in_buffer as *const Image,
                    &out_buffer as *const Image,
                    &mut mat_ptr,
                )
            }
        };

        if ret != 0 {
            return Err(RealCuganError::ProcessFailed(ret));
        }

        Self::convert_image(out_w as u32, out_h as u32, channels, out_bytes)
    }

    fn convert_image(width: u32, height: u32, channels: u8, bytes: Vec<u8>) -> Result<DynamicImage, RealCuganError> {
        match channels {
            4 => RgbaImage::from_raw(width, height, bytes)
                .map(DynamicImage::from)
                .ok_or(RealCuganError::InvalidDimensions(width, height)),
            3 => RgbImage::from_raw(width, height, bytes)
                .map(DynamicImage::from)
                .ok_or(RealCuganError::InvalidDimensions(width, height)),
            _ => Err(RealCuganError::InvalidDimensions(width, height)),
        }
    }
}

impl Drop for RealCugan {
    fn drop(&mut self) {
        unsafe {
            realcugan_free(self.realcugan);
        }
    }
}
