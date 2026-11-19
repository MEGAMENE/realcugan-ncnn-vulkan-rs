# realcugan-ncnn-vulkan-rs

Rust bindings for [realcugan-ncnn-vulkan](https://github.com/MEGAMENE/realcugan-ncnn-vulkan), an anime-style image super-resolution and denoising engine built on Real-CUGAN and Tencent's NCNN Vulkan framework.

## Features

- **1:1 Layer 1 `RealCUGANConfig` Alignment**: Strongly typed Rust configuration (`RealCuganConfig`, `RealCuganScale`, `RealCuganNoise`, `RealCuganSyncGap`, `RealCuganTtaLevel`, `RealCuganModelType`) with strict upfront `(model, scale, noise)` matrix validation.
- **Native Dynamic VRAM Auto-Tiling**: Passing `tile_size = 0` delegates continuous VRAM budget sizing, `concurrent_jobs` scaling, single-tile bypass, and OOM tile-halving recovery directly to Layer 1.
- **Accurate `models-pro` 2x/3x Affine Scaling**: Automatically forwards `is_pro_model` for 2x and 3x `models-pro` models (`0.7 / 0.15` normalization) and overrides Windows BGR defaults for Rust RGB/RGBA images.
- **Multi-Level TTA & FP16 Arithmetic**: Supports `X1`, `X2`, `X4`, and `X8` test-time augmentation levels and configurable FP16 shader arithmetic.
- **Persistent Per-GPU Pipeline Caching**: Reuses NCNN's per-device `VkPipelineCache` files (`pipeline_gpu<id>_<vendor>_<device>_v3.bin`) across runs.
- **Thread-Safe (`Send + Sync`)**: Share a single `Arc<RealCugan>` across worker threads for concurrent inference on GPU or CPU.
- **Zero-Copy 8-Bit Input & Alpha Preservation**: Borrows `Rgb8`/`Rgba8` inputs zero-copy, expands `L8`/`La8` fallibly, converts 16-bit and `Rgb32F`/`Rgba32F` inputs to 8-bit `Rgb8`/`Rgba8` while preserving alpha channels via bicubic upscaling.
- **Self-Contained In-Tree Static Build**: Builds the pinned NCNN (`20261020`) submodule statically with a minimal 20-layer Real-CUGAN allowlist and NCNN's dynamic `simplevk` Vulkan loader — no external NCNN SDK or Vulkan SDK installation required.

## Building

### Prerequisites

- Rust 1.88+ (Edition 2021)
- CMake 3.20+
- C++17 compatible compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- Vulkan runtime driver for GPU inference (loaded dynamically at runtime; CPU inference via `gpuid = -1` works without Vulkan)

### Build & Test

1. Initialize submodules recursively:
   ```bash
   git submodule update --init --recursive
   ```

2. Build and run tests:
   ```bash
   cargo test --release
   ```

*(Optional)* Set `REALCUGAN_NATIVE=1` to optimize C++ wrapper compilation for the host CPU instruction set (`/arch:AVX2` on MSVC x64, `-march=native` on GCC/Clang).

## Usage

### Basic GPU Inference

```rust
use std::num::NonZeroU32;
use image::open;
use realcugan_ncnn_vulkan_rs::{
    RealCugan, RealCuganConfig, RealCuganModelType, RealCuganNoise, RealCuganScale,
    RealCuganSyncGap, RealCuganTtaLevel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = open("input.png")?;

    let mut cfg = RealCuganConfig::new("path/to/models");
    cfg.gpuid = 0; // 0 = first Vulkan GPU, -1 = CPU
    cfg.model = RealCuganModelType::Pro;
    cfg.scale = RealCuganScale::X2;
    cfg.noise = RealCuganNoise::Conservative;
    cfg.tile_size = 0; // 0 = automatic Layer 1 VRAM sizing
    cfg.sync_gap = RealCuganSyncGap::VeryRough;
    cfg.tta_level = RealCuganTtaLevel::X4;
    cfg.concurrent_jobs = NonZeroU32::new(1).unwrap();
    cfg.use_fp16_arithmetic = false;

    let cugan = RealCugan::new(cfg)?;
    let output = cugan.proc_image(&img)?;
    output.save("output.png")?;
    Ok(())
}
```

### CPU Inference

Set `cfg.gpuid = -1` (and leave `cfg.num_threads = None` to automatically use all available CPU cores, or specify `Some(NonZeroU32::new(n).unwrap())`):

```rust
use realcugan_ncnn_vulkan_rs::{RealCugan, RealCuganConfig};

let mut cfg = RealCuganConfig::new("path/to/models");
cfg.gpuid = -1;
let cugan = RealCugan::new(cfg)?;
let output = cugan.proc_image(&img)?;
```

### Concurrent Multi-Threaded Inference (`Send + Sync`)

`RealCugan` is both `Send` and `Sync` and can be shared via `Arc<RealCugan>`:

```rust
use std::num::NonZeroU32;
use std::sync::Arc;
use realcugan_ncnn_vulkan_rs::{RealCugan, RealCuganConfig};

let mut cfg = RealCuganConfig::new("path/to/models");
cfg.concurrent_jobs = NonZeroU32::new(4).unwrap();
let cugan = Arc::new(RealCugan::new(cfg)?);
```

> **Lifetime Note**: NCNN registers `atexit(ncnn::destroy_gpu_instance)` when a Vulkan GPU instance is initialized. Avoid storing GPU-backed `RealCugan` instances inside `thread_local!` destructors or exit handlers that execute after `atexit`, and join worker threads before `main` exits.

## Supported Model Matrix

| Model (`RealCuganModelType`) | Scales (`RealCuganScale`) | Noise Levels (`RealCuganNoise`) |
|---|---|---|
| `Se` | `X2` | `Conservative`, `NoDenoise`, `Denoise1x`, `Denoise2x`, `Denoise3x` |
| `Se` | `X3`, `X4` | `Conservative`, `NoDenoise`, `Denoise3x` |
| `Pro` | `X2`, `X3` | `Conservative`, `NoDenoise`, `Denoise3x` |
| `Nose` | `X2` | `NoDenoise` |

## License

MIT / BSD-3-Clause
