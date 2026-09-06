use std::convert::TryFrom;
use std::ffi::CString;
use std::fmt;

use image::{DynamicImage, RgbaImage, RgbImage};
use libc::{c_char, c_int, c_uchar, c_uint, c_void};

#[derive(Debug)]
pub enum RealCuganError {
    ModelNotFound(String),
    InvalidGpuDevice(i32),
    LoadFailed(String, String),
    InvalidDimensions(u32, u32),
    ProcessFailed(i32),
}

impl fmt::Display for RealCuganError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(p) => write!(f, "Real-CUGAN model file not found: {}", p),
            Self::InvalidGpuDevice(id) => write!(f, "Invalid GPU device id: {}", id),
            Self::LoadFailed(p, m) => write!(f, "Failed to load Real-CUGAN model from {} and {}", p, m),
            Self::InvalidDimensions(w, h) => write!(f, "Invalid image dimensions: {}x{}", w, h),
            Self::ProcessFailed(code) => write!(f, "Real-CUGAN GPU processing failed with code: {}", code),
        }
    }
}

impl std::error::Error for RealCuganError {}

#[derive(Debug)]
#[derive(Copy, Clone, PartialEq)]
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

extern "C" {
    fn realcugan_init(
        gpuid: c_int,
        tta_mode: bool,
        num_threads: c_int,
        noise: c_int,
        scale: c_int,
        tilesize: c_int,
        prepadding: c_int,
        sync_gap: c_int,
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
        in_image: &Image,
        out_image: &Image,
        mat_ptr: *mut *mut c_void,
    ) -> c_int;

    fn realcugan_get_heap_budget(gpuid: c_int) -> c_uint;

    fn realcugan_free_image(mat_ptr: *mut c_void);

    fn realcugan_free(realcugan: *mut c_void);
}

pub struct RealCugan {
    realcugan: *mut c_void,
    scale: u32,
}

unsafe impl Send for RealCugan {}

impl RealCugan {
    pub fn new(gpuid: i32,
               noise: i32,
               scale: u32,
               model: RealCuganModelType,
               tile_size: u32,
               sync_gap: u32,
               tta_mode: bool,
               num_threads: i32,
               models_path: String,
    ) -> Self {
        unsafe {
            let prepadding = match scale {
                2 => 18,
                3 => 14,
                4 => 19,
                _ => panic!("unsupported scale: {}", scale)
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
                RealCuganModelType::Se => "models-se"
            };

            let (model_path, param_path) = if noise == -1 {
                (format!("{}/{}/up{}x-conservative.bin", models_path, model_dir, scale),
                 format!("{}/{}/up{}x-conservative.param", models_path, model_dir, scale))
            } else if noise == 0 {
                (format!("{}/{}/up{}x-no-denoise.bin", models_path, model_dir, scale),
                 format!("{}/{}/up{}x-no-denoise.param", models_path, model_dir, scale))
            } else {
                (format!("{}/{}/up{}x-denoise{}x.bin", models_path, model_dir, scale, noise),
                 format!("{}/{}/up{}x-denoise{}x.param", models_path, model_dir, scale, noise))
            };

            if !std::path::Path::new(&param_path).exists() {
                realcugan_destroy_gpu_instance();
                panic!("model parameter file not found: {}", param_path);
            }
            if !std::path::Path::new(&model_path).exists() {
                realcugan_destroy_gpu_instance();
                panic!("model weights file not found: {}", model_path);
            }

            realcugan_init_gpu_instance();
            let gpu_count = realcugan_get_gpu_count() as i32;
            if gpuid < -1 || gpuid >= gpu_count {
                realcugan_destroy_gpu_instance();
                panic!("invalid gpu device")
            }
            let tile_size = if tile_size == 0 {
                if gpuid == -1 { 400 } else {
                    let calculated_tile_size;
                    let heap_budget = realcugan_get_heap_budget(gpuid);

                    if scale == 2 {
                        if heap_budget > 9000 {
                            calculated_tile_size = 800
                        } else if heap_budget > 6000 {
                            calculated_tile_size = 600
                        } else if heap_budget > 1300 {
                            calculated_tile_size = 400
                        } else if heap_budget > 800 {
                            calculated_tile_size = 300
                        } else if heap_budget > 200 {
                            calculated_tile_size = 100
                        } else {
                            calculated_tile_size = 32
                        }
                    } else if scale == 3 {
                        if heap_budget > 9000 {
                            calculated_tile_size = 800
                        } else if heap_budget > 6000 {
                            calculated_tile_size = 600
                        } else if heap_budget > 3300 {
                            calculated_tile_size = 400
                        } else if heap_budget > 1900 {
                            calculated_tile_size = 300
                        } else if heap_budget > 950 {
                            calculated_tile_size = 200
                        } else if heap_budget > 320 {
                            calculated_tile_size = 100
                        } else {
                            calculated_tile_size = 32
                        }
                    } else if scale == 4 {
                        if heap_budget > 9000 {
                            calculated_tile_size = 800
                        } else if heap_budget > 6000 {
                            calculated_tile_size = 600
                        } else if heap_budget > 1690 {
                            calculated_tile_size = 400
                        } else if heap_budget > 980 {
                            calculated_tile_size = 300
                        } else if heap_budget > 530 {
                            calculated_tile_size = 200
                        } else if heap_budget > 240 {
                            calculated_tile_size = 100
                        } else {
                            calculated_tile_size = 32
                        }
                    } else {
                        calculated_tile_size = 32
                    }

                    if tta_mode {
                        std::cmp::max(32, calculated_tile_size / 2)
                    } else {
                        calculated_tile_size
                    }
                }
            } else { tile_size };
            let realcugan = realcugan_init(
                gpuid,
                tta_mode,
                num_threads,
                noise,
                scale as i32,
                tile_size as i32,
                prepadding,
                sync_gap as i32,
            );

            let param_path_cstr = CString::new(param_path.clone()).unwrap();
            let model_path_cstr = CString::new(model_path.clone()).unwrap();
            let ret = realcugan_load(realcugan, param_path_cstr.as_ptr(), model_path_cstr.as_ptr());
            if ret != 0 {
                realcugan_free(realcugan);
                panic!("failed to load realcugan model from {} and {}", param_path, model_path);
            }

            Self {
                realcugan,
                scale,
            }
        }
    }

    pub fn proc_image(&self, image: DynamicImage) -> Result<DynamicImage, RealCuganError> {
        let (input_image, channels) = match image.color() {
            image::ColorType::Rgb8 => (image, 3),
            image::ColorType::Rgba8 => (image, 4),
            image::ColorType::L8 => (DynamicImage::from(image.to_rgb8()), 3),
            image::ColorType::La8 => (DynamicImage::from(image.to_rgba8()), 4),
            // 16-bit depth normalization: map [0, 65535] -> [0, 255]
            image::ColorType::Rgb16 | image::ColorType::L16 => (DynamicImage::from(image.to_rgb8()), 3),
            image::ColorType::Rgba16 | image::ColorType::La16 => (DynamicImage::from(image.to_rgba8()), 4),
            _ => (DynamicImage::from(image.to_rgb8()), 3),
        };

        let in_w = i32::try_from(input_image.width())
            .map_err(|_| RealCuganError::InvalidDimensions(input_image.width(), input_image.height()))?;
        let in_h = i32::try_from(input_image.height())
            .map_err(|_| RealCuganError::InvalidDimensions(input_image.width(), input_image.height()))?;

        let in_buffer = Image {
            data: input_image.as_bytes().as_ptr() as *const c_uchar,
            w: in_w,
            h: in_h,
            c: i32::from(channels),
        };

        let out_w = in_w * (self.scale as i32);
        let out_h = in_h * (self.scale as i32);
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
            realcugan_process(
                self.realcugan,
                &in_buffer as *const Image,
                &out_buffer as *const Image,
                &mut mat_ptr,
            )
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
