use std::convert::TryFrom;
use std::fmt;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use image::{ColorType, DynamicImage, RgbImage, RgbaImage};

const RC_OK: i32 = 0;
const RC_ERR_GPU_INIT_FAILED: i32 = -2;
const RC_ERR_INVALID_GPU: i32 = -3;

#[repr(C)]
struct RealCuganHandle {
    _data: [u8; 0],
    _marker: std::marker::PhantomData<(*mut u8, std::marker::PhantomPinned)>,
}

#[repr(C)]
struct RealCuganFfiConfig {
    gpuid: i32,
    scale: i32,
    tilesize: i32,
    syncgap: i32,
    tta_level: i32,
    num_threads: i32,
    concurrent_jobs: i32,
    is_pro_model: u8,
    use_fp16_arithmetic: u8,
}

#[cfg(windows)]
type NativePathChar = u16;
#[cfg(not(windows))]
type NativePathChar = std::ffi::c_char;

extern "C" {
    fn realcugan_create(
        cfg: *const RealCuganFfiConfig,
        param_path: *const NativePathChar,
        model_path: *const NativePathChar,
        out_handle: *mut *mut RealCuganHandle,
    ) -> i32;

    fn realcugan_destroy(handle: *mut RealCuganHandle);

    fn realcugan_process(
        handle: *const RealCuganHandle,
        in_data: *const u8,
        in_w: i32,
        in_h: i32,
        out_data: *mut u8,
        out_w: i32,
        out_h: i32,
        channels: i32,
    ) -> i32;

    fn realcugan_get_gpu_count() -> i32;

    fn realcugan_get_default_gpu_index() -> i32;

    fn realcugan_get_heap_budget(gpuid: i32) -> u32;

    fn realcugan_get_gpu_device_type(gpuid: i32, out_type: *mut i32) -> i32;

    fn realcugan_should_use_cpu(gpuid: i32) -> i32;

    fn realcugan_compute_auto_tilesize(
        gpuid: i32,
        scale: i32,
        tta_level: i32,
        concurrent_jobs: i32,
    ) -> i32;

    fn realcugan_default_prepadding(scale: i32) -> i32;
}

/// Real-CUGAN model family.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RealCuganModelType {
    /// Lightweight `models-nose` (2x no-denoise only; no sync-gap blobs).
    Nose,
    /// `models-pro` (2x and 3x, conservative / no-denoise / denoise3x; uses 0.7/0.15 affine scaling).
    Pro,
    /// Standard `models-se` (2x/3x/4x).
    Se,
}

/// Supported Real-CUGAN upscaling ratios.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RealCuganScale {
    X2 = 2,
    X3 = 3,
    X4 = 4,
}

impl RealCuganScale {
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as u32 as i32
    }
}

impl TryFrom<u32> for RealCuganScale {
    type Error = RealCuganError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::X2),
            3 => Ok(Self::X3),
            4 => Ok(Self::X4),
            other => Err(RealCuganError::UnsupportedScale(other)),
        }
    }
}

/// Real-CUGAN denoising strength level.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum RealCuganNoise {
    Conservative = -1,
    NoDenoise = 0,
    Denoise1x = 1,
    Denoise2x = 2,
    Denoise3x = 3,
}

impl RealCuganNoise {
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as i32
    }
}

impl TryFrom<i32> for RealCuganNoise {
    type Error = RealCuganError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Conservative),
            0 => Ok(Self::NoDenoise),
            1 => Ok(Self::Denoise1x),
            2 => Ok(Self::Denoise2x),
            3 => Ok(Self::Denoise3x),
            other => Err(RealCuganError::InvalidNoise(other)),
        }
    }
}

/// Tile feature synchronization mode (`syncgap`).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RealCuganSyncGap {
    /// Disable global feature sync across tiles.
    None = 0,
    /// Full 4-stage per-tile synchronization.
    Accurate = 1,
    /// Single-pass all-gap synchronization over all tiles.
    Rough = 2,
    /// Subsampled stride-2 synchronization (auto-downgrades to `Rough` if `< 3` tiles on either axis).
    VeryRough = 3,
}

impl RealCuganSyncGap {
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as u32 as i32
    }
}

impl TryFrom<u32> for RealCuganSyncGap {
    type Error = RealCuganError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Accurate),
            2 => Ok(Self::Rough),
            3 => Ok(Self::VeryRough),
            other => Err(RealCuganError::InvalidSyncGap(other)),
        }
    }
}

/// Test-Time Augmentation (TTA) pass count.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RealCuganTtaLevel {
    X1 = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
}

impl RealCuganTtaLevel {
    #[inline]
    pub const fn as_i32(self) -> i32 {
        self as u32 as i32
    }
}

impl TryFrom<u32> for RealCuganTtaLevel {
    type Error = RealCuganError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::X1),
            2 => Ok(Self::X2),
            4 => Ok(Self::X4),
            8 => Ok(Self::X8),
            other => Err(RealCuganError::InvalidTtaLevel(other)),
        }
    }
}

/// Vulkan physical device classification reported by NCNN.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RealCuganGpuType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Unknown(i32),
}

/// Errors returned by Real-CUGAN configuration, model loading, or inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealCuganError {
    UnsupportedScale(u32),
    InvalidNoise(i32),
    InvalidSyncGap(u32),
    InvalidTtaLevel(u32),
    UnsupportedModelConfig {
        model: RealCuganModelType,
        scale: RealCuganScale,
        noise: RealCuganNoise,
    },
    InvalidTileSize(u32),
    InvalidGpuDevice(i32),
    GpuInitFailed,
    InvalidPath(PathBuf),
    ModelNotFound(PathBuf),
    LoadFailed {
        param_path: PathBuf,
        model_path: PathBuf,
    },
    InvalidDimensions(u32, u32),
    AllocationFailed(usize),
    ProcessFailed(i32),
}

impl fmt::Display for RealCuganError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedScale(s) => write!(f, "Unsupported Real-CUGAN scale (must be 2, 3, or 4): {s}"),
            Self::InvalidNoise(n) => write!(f, "Invalid Real-CUGAN noise level (must be -1..=3): {n}"),
            Self::InvalidSyncGap(g) => write!(f, "Invalid Real-CUGAN sync gap (must be 0..=3): {g}"),
            Self::InvalidTtaLevel(lvl) => {
                write!(f, "Invalid TTA level (must be 1, 2, 4, or 8): {lvl}")
            }
            Self::UnsupportedModelConfig { model, scale, noise } => write!(
                f,
                "Unsupported Real-CUGAN model combination: model={model:?}, scale={scale:?}, noise={noise:?}"
            ),
            Self::InvalidTileSize(ts) => {
                write!(f, "Invalid tile size (must be 0 for auto, or 32..=i32::MAX): {ts}")
            }
            Self::InvalidGpuDevice(id) => write!(f, "Invalid GPU device id: {id}"),
            Self::GpuInitFailed => write!(f, "Failed to initialize Vulkan GPU instance"),
            Self::InvalidPath(p) => write!(f, "Invalid model path (contains interior NUL or empty): {}", p.display()),
            Self::ModelNotFound(p) => write!(f, "Real-CUGAN model file not found: {}", p.display()),
            Self::LoadFailed {
                param_path,
                model_path,
            } => write!(
                f,
                "Failed to load Real-CUGAN model from {} and {}",
                param_path.display(),
                model_path.display()
            ),
            Self::InvalidDimensions(w, h) => write!(f, "Invalid or overflowing image dimensions: {w}x{h}"),
            Self::AllocationFailed(bytes) => {
                write!(f, "Failed to allocate {bytes} bytes for Real-CUGAN image buffer")
            }
            Self::ProcessFailed(code) => write!(f, "Real-CUGAN processing failed with code: {code}"),
        }
    }
}

impl std::error::Error for RealCuganError {}

/// Configuration for creating a [`RealCugan`] upscaler instance, aligned 1:1 with Layer 1's `RealCUGANConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealCuganConfig {
    /// Device index (`-1` for CPU, `>= 0` for Vulkan GPU; software Vulkan devices automatically fall back to CPU).
    pub gpuid: i32,
    /// Model architecture family (`Nose`, `Pro`, or `Se`).
    pub model: RealCuganModelType,
    /// Upscaling factor (`X2`, `X3`, or `X4`).
    pub scale: RealCuganScale,
    /// Denoise strength (`Conservative`, `NoDenoise`, `Denoise1x`, `Denoise2x`, `Denoise3x`).
    pub noise: RealCuganNoise,
    /// Root directory containing `models-se`, `models-pro`, and/or `models-nose`.
    pub models_dir: PathBuf,
    /// Tile size in pixels: `0` delegates dynamic VRAM auto-sizing to Layer 1, or specify `>= 32`.
    pub tile_size: u32,
    /// Cross-tile feature synchronization mode.
    pub sync_gap: RealCuganSyncGap,
    /// Test-Time Augmentation pass count.
    pub tta_level: RealCuganTtaLevel,
    /// CPU thread count. `None` chooses `available_parallelism()` when executing on CPU (`gpuid == -1` or software GPU), else `1`.
    pub num_threads: Option<NonZeroU32>,
    /// Number of concurrent worker jobs sharing the same `gpuid` (scales Layer 1's VRAM tile and row-cache budgets).
    pub concurrent_jobs: NonZeroU32,
    /// Enable FP16 shader arithmetic when supported by the Vulkan device.
    pub use_fp16_arithmetic: bool,
}

impl RealCuganConfig {
    /// Creates a configuration with sensible defaults for the given `models_dir`:
    /// GPU `0`, `Se` model, `2x` scale, `Conservative` denoise, auto `tile_size` (`0`),
    /// `VeryRough` sync gap, `X1` TTA, automatic `num_threads`, `1` concurrent job, FP16 arithmetic disabled.
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            gpuid: 0,
            model: RealCuganModelType::Se,
            scale: RealCuganScale::X2,
            noise: RealCuganNoise::Conservative,
            models_dir: models_dir.into(),
            tile_size: 0,
            sync_gap: RealCuganSyncGap::VeryRough,
            tta_level: RealCuganTtaLevel::X1,
            num_threads: None,
            concurrent_jobs: NonZeroU32::MIN,
            use_fp16_arithmetic: false,
        }
    }
}

/// Thread-safe handle to a loaded Real-CUGAN NCNN Vulkan/CPU upscaler.
///
/// # Concurrency & Process-Exit Lifetime Safety
/// `RealCugan` implements both `Send` and `Sync` and can be shared across worker threads via
/// `Arc<RealCugan>` for concurrent [`RealCugan::proc_image`] calls (configure
/// [`RealCuganConfig::concurrent_jobs`] to match your concurrency level for optimal VRAM sizing).
///
/// **Lifetime caveat**: NCNN registers an `atexit(ncnn::destroy_gpu_instance)` handler when a
/// Vulkan GPU instance is initialized. Do **not** store GPU-backed `RealCugan` instances in
/// `thread_local!` destructors or custom exit handlers that may run *after* C `atexit` teardown,
/// as `~RealCUGAN()` accesses the underlying `ncnn::VulkanDevice`. Drop all `RealCugan` instances
/// (and join worker threads) before `main` returns.
pub struct RealCugan {
    handle: NonNull<RealCuganHandle>,
    scale: RealCuganScale,
}

// SAFETY: `RealCUGAN` holds no thread-affine state; per-call Vulkan command pools/buffers and
// allocators are acquired/released per `process()` call, and destructor pipeline-cache persistence
// is mutex-protected. GPU instance teardown is never invoked from `Drop`.
unsafe impl Send for RealCugan {}

// SAFETY: `RealCUGAN::load()` runs exclusively during `RealCugan::new()` before `&RealCugan`
// exists, and `RealCUGAN::process()` is `const` with all shared NCNN device resources
// (allocators, queues, utility operators, `PoolAllocator`, atomic `Mat`/`VkMat` refcounts)
// internally synchronized.
unsafe impl Sync for RealCugan {}

impl RealCugan {
    /// Creates and loads a new Real-CUGAN instance from a validated [`RealCuganConfig`].
    pub fn new(config: RealCuganConfig) -> Result<Self, RealCuganError> {
        if !is_valid_model_combo(config.model, config.scale, config.noise) {
            return Err(RealCuganError::UnsupportedModelConfig {
                model: config.model,
                scale: config.scale,
                noise: config.noise,
            });
        }

        if config.tile_size != 0 && !(32..=i32::MAX as u32).contains(&config.tile_size) {
            return Err(RealCuganError::InvalidTileSize(config.tile_size));
        }

        if config.gpuid < -1 {
            return Err(RealCuganError::InvalidGpuDevice(config.gpuid));
        }

        let (param_path, model_path) = resolve_model_paths(
            &config.models_dir,
            config.model,
            config.scale,
            config.noise,
        );

        let param_native = encode_native_path(&param_path)?;
        let model_native = encode_native_path(&model_path)?;

        if !param_path.is_file() {
            return Err(RealCuganError::ModelNotFound(param_path));
        }
        if !model_path.is_file() {
            return Err(RealCuganError::ModelNotFound(model_path));
        }

        let num_threads = match config.num_threads {
            Some(n) => i32::try_from(n.get()).unwrap_or(i32::MAX),
            None => {
                let cpu_exec = config.gpuid == -1 || Self::should_use_cpu(config.gpuid);
                if cpu_exec {
                    std::thread::available_parallelism()
                        .map(|n| i32::try_from(n.get()).unwrap_or(i32::MAX))
                        .unwrap_or(1)
                } else {
                    1
                }
            }
        };

        let concurrent_jobs = i32::try_from(config.concurrent_jobs.get()).unwrap_or(i32::MAX);
        let is_pro_model = config.model == RealCuganModelType::Pro
            && matches!(config.scale, RealCuganScale::X2 | RealCuganScale::X3);

        let ffi_cfg = RealCuganFfiConfig {
            gpuid: config.gpuid,
            scale: config.scale.as_i32(),
            tilesize: config.tile_size as i32,
            syncgap: config.sync_gap.as_i32(),
            tta_level: config.tta_level.as_i32(),
            num_threads,
            concurrent_jobs,
            is_pro_model: u8::from(is_pro_model),
            use_fp16_arithmetic: u8::from(config.use_fp16_arithmetic),
        };

        let mut raw_handle: *mut RealCuganHandle = std::ptr::null_mut();
        let ret = unsafe {
            realcugan_create(
                &ffi_cfg,
                param_native.as_ptr(),
                model_native.as_ptr(),
                &mut raw_handle,
            )
        };

        if ret != RC_OK {
            return Err(match ret {
                RC_ERR_GPU_INIT_FAILED => RealCuganError::GpuInitFailed,
                RC_ERR_INVALID_GPU => RealCuganError::InvalidGpuDevice(config.gpuid),
                _ => RealCuganError::LoadFailed {
                    param_path,
                    model_path,
                },
            });
        }

        let handle = NonNull::new(raw_handle).ok_or(RealCuganError::LoadFailed {
            param_path,
            model_path,
        })?;

        Ok(Self {
            handle,
            scale: config.scale,
        })
    }

    /// Returns the upscaling factor configured on this instance.
    #[inline]
    pub fn scale(&self) -> RealCuganScale {
        self.scale
    }

    /// Returns the number of Vulkan GPU devices detected by NCNN.
    pub fn get_gpu_count() -> i32 {
        unsafe { realcugan_get_gpu_count() }
    }

    /// Returns NCNN's preferred default Vulkan GPU index, or `-1` if no GPU is available.
    pub fn get_default_gpu_index() -> i32 {
        unsafe { realcugan_get_default_gpu_index() }
    }

    /// Returns the estimated device-local VRAM heap budget in MiB for `gpuid`, or `0` if invalid.
    pub fn get_heap_budget(gpuid: i32) -> u32 {
        unsafe { realcugan_get_heap_budget(gpuid) }
    }

    /// Returns the Vulkan physical device type for `gpuid`, or `None` if `gpuid` is invalid.
    pub fn get_gpu_device_type(gpuid: i32) -> Option<RealCuganGpuType> {
        let mut raw_type: i32 = 0;
        let ret = unsafe { realcugan_get_gpu_device_type(gpuid, &mut raw_type) };
        if ret != RC_OK {
            return None;
        }
        Some(match raw_type {
            0 => RealCuganGpuType::Discrete,
            1 => RealCuganGpuType::Integrated,
            2 => RealCuganGpuType::Virtual,
            3 => RealCuganGpuType::Cpu,
            other => RealCuganGpuType::Unknown(other),
        })
    }

    /// Returns `true` if `gpuid` will execute on CPU (`gpuid < 0`, invalid device, or software Vulkan adapter).
    pub fn should_use_cpu(gpuid: i32) -> bool {
        unsafe { realcugan_should_use_cpu(gpuid) != 0 }
    }

    /// Queries Layer 1's native VRAM-aware automatic tile size calculation for the given parameters.
    pub fn compute_auto_tilesize(
        gpuid: i32,
        scale: RealCuganScale,
        tta_level: RealCuganTtaLevel,
        concurrent_jobs: NonZeroU32,
    ) -> Option<u32> {
        let jobs = i32::try_from(concurrent_jobs.get()).unwrap_or(i32::MAX);
        let size = unsafe {
            realcugan_compute_auto_tilesize(gpuid, scale.as_i32(), tta_level.as_i32(), jobs)
        };
        u32::try_from(size).ok().filter(|&v| v > 0)
    }

    /// Returns Layer 1's default tile prepadding in pixels for `scale`.
    pub fn default_prepadding(scale: RealCuganScale) -> u32 {
        let pad = unsafe { realcugan_default_prepadding(scale.as_i32()) };
        u32::try_from(pad).unwrap_or(0)
    }

    /// Upscales `image` using the configured Real-CUGAN engine (Vulkan GPU or CPU).
    ///
    /// # Pixel Format Semantics
    /// - 8-bit `Rgb8` and `Rgba8` inputs are borrowed zero-copy across FFI.
    /// - `L8` and `La8` inputs are expanded fallibly to 3-channel RGB / 4-channel RGBA.
    /// - 16-bit (`L16`, `La16`, `Rgb16`, `Rgba16`) and 32-bit float (`Rgb32F`, `Rgba32F`) inputs
    ///   are converted to 8-bit `Rgb8` / `Rgba8` prior to inference, preserving alpha channels.
    /// - Output is always 8-bit `DynamicImage::ImageRgb8` (for opaque inputs) or
    ///   `DynamicImage::ImageRgba8` (for inputs with an alpha channel), matching Layer 1's
    ///   packed 8-bit `RealCUGAN::process()` contract.
    pub fn proc_image(&self, image: &DynamicImage) -> Result<DynamicImage, RealCuganError> {
        let color = image.color();
        let channels: u8 = match color {
            ColorType::Rgb8 | ColorType::L8 | ColorType::Rgb16 | ColorType::L16 | ColorType::Rgb32F => 3,
            ColorType::Rgba8
            | ColorType::La8
            | ColorType::Rgba16
            | ColorType::La16
            | ColorType::Rgba32F => 4,
            _ => {
                if color.has_alpha() {
                    4
                } else {
                    3
                }
            }
        };

        let dims = validate_process_dims(image.width(), image.height(), channels, self.scale)?;

        enum PixelData<'a> {
            Borrowed(&'a [u8]),
            Owned(Vec<u8>),
        }

        let pixel_data = match color {
            ColorType::Rgb8 | ColorType::Rgba8 => PixelData::Borrowed(image.as_bytes()),
            ColorType::L8 => {
                let src = image.as_bytes();
                let expected_src = (dims.in_w as usize) * (dims.in_h as usize);
                if src.len() != expected_src {
                    return Err(RealCuganError::InvalidDimensions(
                        image.width(),
                        image.height(),
                    ));
                }
                let mut rgb = Vec::new();
                rgb.try_reserve_exact(dims.expected_in_len)
                    .map_err(|_| RealCuganError::AllocationFailed(dims.expected_in_len))?;
                for &y in src {
                    rgb.extend_from_slice(&[y, y, y]);
                }
                PixelData::Owned(rgb)
            }
            ColorType::La8 => {
                let src = image.as_bytes();
                let expected_src = (dims.in_w as usize) * (dims.in_h as usize) * 2;
                if src.len() != expected_src {
                    return Err(RealCuganError::InvalidDimensions(
                        image.width(),
                        image.height(),
                    ));
                }
                let mut rgba = Vec::new();
                rgba.try_reserve_exact(dims.expected_in_len)
                    .map_err(|_| RealCuganError::AllocationFailed(dims.expected_in_len))?;
                for &[y, a] in src.as_chunks::<2>().0 {
                    rgba.extend_from_slice(&[y, y, y, a]);
                }
                PixelData::Owned(rgba)
            }
            ColorType::Rgb16 | ColorType::L16 | ColorType::Rgb32F => {
                PixelData::Owned(image.to_rgb8().into_raw())
            }
            ColorType::Rgba16 | ColorType::La16 | ColorType::Rgba32F => {
                PixelData::Owned(image.to_rgba8().into_raw())
            }
            _ => {
                if color.has_alpha() {
                    PixelData::Owned(image.to_rgba8().into_raw())
                } else {
                    PixelData::Owned(image.to_rgb8().into_raw())
                }
            }
        };

        let raw_slice = match &pixel_data {
            PixelData::Borrowed(slice) => *slice,
            PixelData::Owned(vec) => vec.as_slice(),
        };

        if raw_slice.len() != dims.expected_in_len {
            return Err(RealCuganError::InvalidDimensions(
                image.width(),
                image.height(),
            ));
        }

        let mut out_bytes: Vec<u8> = Vec::new();
        out_bytes
            .try_reserve_exact(dims.out_len)
            .map_err(|_| RealCuganError::AllocationFailed(dims.out_len))?;

        let out_ptr = out_bytes.spare_capacity_mut().as_mut_ptr().cast::<u8>();

        let ret = unsafe {
            realcugan_process(
                self.handle.as_ptr(),
                raw_slice.as_ptr(),
                dims.in_w,
                dims.in_h,
                out_ptr,
                dims.out_w,
                dims.out_h,
                i32::from(channels),
            )
        };

        if ret != RC_OK {
            return Err(RealCuganError::ProcessFailed(ret));
        }

        // SAFETY: `realcugan_process` returned `RC_OK` only after verifying exact output dimensions,
        // completing `RealCUGAN::process()` over all `out_w * out_h * channels` bytes, and
        // confirming `out_mat.data == out_ptr` (no internal reallocation).
        unsafe {
            out_bytes.set_len(dims.out_len);
        }

        convert_output_image(dims.out_w as u32, dims.out_h as u32, channels, out_bytes)
    }
}

impl fmt::Debug for RealCugan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealCugan")
            .field("scale", &self.scale)
            .finish_non_exhaustive()
    }
}

impl Drop for RealCugan {
    fn drop(&mut self) {
        unsafe {
            realcugan_destroy(self.handle.as_ptr());
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct ValidatedDims {
    in_w: i32,
    in_h: i32,
    out_w: i32,
    out_h: i32,
    expected_in_len: usize,
    out_len: usize,
}

fn validate_process_dims(
    width: u32,
    height: u32,
    channels: u8,
    scale: RealCuganScale,
) -> Result<ValidatedDims, RealCuganError> {
    if channels != 3 && channels != 4 {
        return Err(RealCuganError::InvalidDimensions(width, height));
    }

    let in_w = i32::try_from(width).map_err(|_| RealCuganError::InvalidDimensions(width, height))?;
    let in_h =
        i32::try_from(height).map_err(|_| RealCuganError::InvalidDimensions(width, height))?;

    if in_w <= 0 || in_h <= 0 {
        return Err(RealCuganError::InvalidDimensions(width, height));
    }

    let c_i32 = i32::from(channels);

    // Guard `in_w * in_h * c <= i32::MAX` (required by ncnn::Mat::from_pixels_roi int offset math)
    let in_bytes_i32 = in_w
        .checked_mul(c_i32)
        .and_then(|stride| stride.checked_mul(in_h))
        .ok_or(RealCuganError::InvalidDimensions(width, height))?;
    let expected_in_len = usize::try_from(in_bytes_i32)
        .map_err(|_| RealCuganError::InvalidDimensions(width, height))?;

    let out_w = in_w
        .checked_mul(scale.as_i32())
        .ok_or(RealCuganError::InvalidDimensions(width, height))?;
    let out_h = in_h
        .checked_mul(scale.as_i32())
        .ok_or(RealCuganError::InvalidDimensions(width, height))?;

    // Guard `out_w * c <= i32::MAX` (cpu_postproc_and_store_tile signed int stride)
    out_w
        .checked_mul(c_i32)
        .ok_or(RealCuganError::InvalidDimensions(width, height))?;

    // Guard `out_w * out_h <= i32::MAX` (download_row_output -> ncnn::Mat::to_pixels signed int w * h)
    out_w
        .checked_mul(out_h)
        .ok_or(RealCuganError::InvalidDimensions(width, height))?;

    let out_len = (out_w as usize)
        .checked_mul(out_h as usize)
        .and_then(|wh| wh.checked_mul(usize::from(channels)))
        .filter(|&len| len <= isize::MAX as usize)
        .ok_or(RealCuganError::InvalidDimensions(width, height))?;

    Ok(ValidatedDims {
        in_w,
        in_h,
        out_w,
        out_h,
        expected_in_len,
        out_len,
    })
}

fn is_valid_model_combo(
    model: RealCuganModelType,
    scale: RealCuganScale,
    noise: RealCuganNoise,
) -> bool {
    matches!(
        (model, scale, noise),
        (
            RealCuganModelType::Nose,
            RealCuganScale::X2,
            RealCuganNoise::NoDenoise
        ) | (
            RealCuganModelType::Pro,
            RealCuganScale::X2 | RealCuganScale::X3,
            RealCuganNoise::Conservative
                | RealCuganNoise::NoDenoise
                | RealCuganNoise::Denoise3x
        ) | (RealCuganModelType::Se, RealCuganScale::X2, _)
            | (
                RealCuganModelType::Se,
                RealCuganScale::X3 | RealCuganScale::X4,
                RealCuganNoise::Conservative
                    | RealCuganNoise::NoDenoise
                    | RealCuganNoise::Denoise3x
            )
    )
}

fn resolve_model_paths(
    models_dir: &Path,
    model: RealCuganModelType,
    scale: RealCuganScale,
    noise: RealCuganNoise,
) -> (PathBuf, PathBuf) {
    let model_subdir = match model {
        RealCuganModelType::Nose => "models-nose",
        RealCuganModelType::Pro => "models-pro",
        RealCuganModelType::Se => "models-se",
    };

    let stem = match noise {
        RealCuganNoise::Conservative => format!("up{}x-conservative", scale.as_u32()),
        RealCuganNoise::NoDenoise => format!("up{}x-no-denoise", scale.as_u32()),
        RealCuganNoise::Denoise1x => format!("up{}x-denoise1x", scale.as_u32()),
        RealCuganNoise::Denoise2x => format!("up{}x-denoise2x", scale.as_u32()),
        RealCuganNoise::Denoise3x => format!("up{}x-denoise3x", scale.as_u32()),
    };

    let dir = models_dir.join(model_subdir);
    (dir.join(format!("{stem}.param")), dir.join(format!("{stem}.bin")))
}

#[cfg(windows)]
fn encode_native_path(path: &Path) -> Result<Vec<u16>, RealCuganError> {
    use std::os::windows::ffi::OsStrExt;

    let os = path.as_os_str();
    if os.is_empty() {
        return Err(RealCuganError::InvalidPath(path.to_path_buf()));
    }
    let mut wide: Vec<u16> = Vec::new();
    for unit in os.encode_wide() {
        if unit == 0 {
            return Err(RealCuganError::InvalidPath(path.to_path_buf()));
        }
        wide.push(unit);
    }
    wide.push(0);
    Ok(wide)
}

#[cfg(unix)]
fn encode_native_path(path: &Path) -> Result<std::ffi::CString, RealCuganError> {
    use std::os::unix::ffi::OsStrExt;

    let bytes = path.as_os_str().as_bytes();
    if bytes.is_empty() {
        return Err(RealCuganError::InvalidPath(path.to_path_buf()));
    }
    std::ffi::CString::new(bytes).map_err(|_| RealCuganError::InvalidPath(path.to_path_buf()))
}

#[cfg(not(any(windows, unix)))]
fn encode_native_path(path: &Path) -> Result<std::ffi::CString, RealCuganError> {
    let s = path
        .to_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| RealCuganError::InvalidPath(path.to_path_buf()))?;
    std::ffi::CString::new(s).map_err(|_| RealCuganError::InvalidPath(path.to_path_buf()))
}

fn convert_output_image(
    width: u32,
    height: u32,
    channels: u8,
    bytes: Vec<u8>,
) -> Result<DynamicImage, RealCuganError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_process_dims_zero_and_overflow_guards() {
        // Zero dimensions
        assert_eq!(
            validate_process_dims(0, 0, 3, RealCuganScale::X2),
            Err(RealCuganError::InvalidDimensions(0, 0))
        );
        assert_eq!(
            validate_process_dims(0, 16, 3, RealCuganScale::X2),
            Err(RealCuganError::InvalidDimensions(0, 16))
        );
        assert_eq!(
            validate_process_dims(16, 0, 4, RealCuganScale::X2),
            Err(RealCuganError::InvalidDimensions(16, 0))
        );

        // Input signed-32-bit byte offset overflow: 23171 * 23171 * 4 = 2,147,580,964 > i32::MAX
        assert_eq!(
            validate_process_dims(23_171, 23_171, 4, RealCuganScale::X2),
            Err(RealCuganError::InvalidDimensions(23_171, 23_171))
        );

        // Output stride overflow: out_w = 200_000_000 * 3 = 600_000_000 <= i32::MAX,
        // but out_w * 4 = 2,400,000,000 > i32::MAX
        assert_eq!(
            validate_process_dims(200_000_000, 1, 4, RealCuganScale::X3),
            Err(RealCuganError::InvalidDimensions(200_000_000, 1))
        );

        // Output pixel count overflow (`out_w * out_h > i32::MAX` while `in_w * in_h * c <= i32::MAX`):
        // in = 1_000_000 x 500 x 3 = 1.5e9 <= i32::MAX, out_w * out_h = 4_000_000 * 2_000 = 8e9 > i32::MAX
        assert_eq!(
            validate_process_dims(1_000_000, 500, 3, RealCuganScale::X4),
            Err(RealCuganError::InvalidDimensions(1_000_000, 500))
        );

        // Width > i32::MAX
        assert_eq!(
            validate_process_dims(u32::MAX, 1, 3, RealCuganScale::X2),
            Err(RealCuganError::InvalidDimensions(u32::MAX, 1))
        );

        // Valid normal case
        let ok = validate_process_dims(16, 24, 4, RealCuganScale::X3).unwrap();
        assert_eq!(ok.in_w, 16);
        assert_eq!(ok.in_h, 24);
        assert_eq!(ok.out_w, 48);
        assert_eq!(ok.out_h, 72);
        assert_eq!(ok.expected_in_len, 16 * 24 * 4);
        assert_eq!(ok.out_len, 48 * 72 * 4);
    }

    #[test]
    fn test_encode_native_path_rejects_empty_and_interior_nul() {
        assert!(matches!(
            encode_native_path(Path::new("")),
            Err(RealCuganError::InvalidPath(_))
        ));
        assert!(matches!(
            encode_native_path(Path::new("models/up2x\0.param")),
            Err(RealCuganError::InvalidPath(_))
        ));
    }

    #[test]
    fn test_enum_try_from_validation() {
        assert_eq!(RealCuganScale::try_from(2), Ok(RealCuganScale::X2));
        assert_eq!(RealCuganScale::try_from(3), Ok(RealCuganScale::X3));
        assert_eq!(RealCuganScale::try_from(4), Ok(RealCuganScale::X4));
        assert_eq!(
            RealCuganScale::try_from(1),
            Err(RealCuganError::UnsupportedScale(1))
        );

        assert_eq!(RealCuganNoise::try_from(-1), Ok(RealCuganNoise::Conservative));
        assert_eq!(RealCuganNoise::try_from(3), Ok(RealCuganNoise::Denoise3x));
        assert_eq!(
            RealCuganNoise::try_from(4),
            Err(RealCuganError::InvalidNoise(4))
        );

        assert_eq!(RealCuganSyncGap::try_from(0), Ok(RealCuganSyncGap::None));
        assert_eq!(
            RealCuganSyncGap::try_from(4),
            Err(RealCuganError::InvalidSyncGap(4))
        );

        assert_eq!(RealCuganTtaLevel::try_from(8), Ok(RealCuganTtaLevel::X8));
        assert_eq!(
            RealCuganTtaLevel::try_from(3),
            Err(RealCuganError::InvalidTtaLevel(3))
        );
    }
}
