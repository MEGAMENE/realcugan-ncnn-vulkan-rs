# realcugan-ncnn-vulkan-rs

Rust wrapper for [realcugan-ncnn-vulkan](https://github.com/MEGAMENE/realcugan-ncnn-vulkan), an anime-style image super-resolution and restoration engine based on Real-CUGAN and Tencent's NCNN Vulkan framework.

## Features

- **Continuous Safe VRAM Sizing**: Dynamically computes optimal tile dimensions based on device heap budget, scaling ratio, and GPU architecture (integrated vs. discrete).
- **Single-Tile Bypass**: Skips tiling overhead and sync-gap overhead entirely when images fit within safe VRAM limits.
- **Dynamic OOM Recovery**: Automatically catches GPU command buffer memory exhaustion and halves tile sizes on the fly down to 32.
- **Multi-Level TTA (Test-Time Augmentation)**: Supports 1x (off), 2x, 4x, and 8x augmentation levels for higher quality reconstruction.
- **FP16 Arithmetic Support**: Configurable FP16 execution flags for Intel and AMD GPUs.
- **Persistent Pipeline Caching**: Reusable per-GPU Vulkan pipeline caches saved to platform cache folders (`pipeline_gpu<gpuid>.bin`) to minimize startup latency.
- **RGB and RGBA Alpha Channel Preservation**: Full support for transparency and depth normalization.
- **CPU and GPU Execution**: Automatic CPU routing for `gpuid = -1` or explicit CPU inference via `proc_image_cpu`.

## Building

### Prerequisites

- CMake 3.20+
- C++17 compatible compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- Vulkan runtime driver (NCNN uses a dynamic in-house loader; no separate Vulkan SDK installation is required)

### Setup

1. Initialize submodules:
   ```bash
   git submodule update --init --recursive
   ```

2. Build the crate:
   ```bash
   cargo build --release
   ```

*(Optional)* To link against an existing system NCNN installation instead of building in-tree, point `NCNN_DIR` to the directory containing `ncnnConfig.cmake`:
```bash
NCNN_DIR=/path/to/ncnn/lib/cmake/ncnn cargo build --release
```

## Usage

### Basic Inference

```rust
use image::open;
use realcugan_ncnn_vulkan_rs::{RealCugan, RealCuganModelType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = open("input.png")?;

    // Initialize 2x upscaler with auto-calculated VRAM tile size and default SE model
    let cugan = RealCugan::new(
        0,                        // gpuid (0 for first GPU, -1 for CPU)
        -1,                       // noise level (-1: conservative, 0: no-denoise, 1-3: denoise level)
        2,                        // scale (2, 3, or 4)
        RealCuganModelType::Se,   // model type: Nose, Pro, or Se
        0,                        // tile_size (0 = auto continuous safe VRAM sizing)
        3,                        // sync_gap (0, 1, 2, or 3)
        false,                    // tta_mode (legacy bool: false = 1x, true = 8x)
        1,                        // num_threads (processing threads)
        "path/to/models".into(),  // path to models directory
    )?;

    let output = cugan.proc_image(&img)?;
    output.save("output.png")?;
    Ok(())
}
```

### Advanced Options: TTA Levels & FP16 Arithmetic

Use `new_with_options` to specify granular TTA levels (1, 2, 4, 8) and enable FP16 arithmetic:

```rust
use realcugan_ncnn_vulkan_rs::{RealCugan, RealCuganModelType};

let cugan = RealCugan::new_with_options(
    0,                        // gpuid (0 = GPU 0)
    -1,                       // noise level
    2,                        // scale (2x)
    RealCuganModelType::Pro,  // model architecture
    0,                        // auto tile size
    3,                        // sync gap mode
    4,                        // tta_level (1, 2, 4, or 8)
    true,                     // use_fp16_arithmetic
    1,                        // num_threads
    "models".into(),          // models directory
)?;

let result = cugan.proc_image(&image)?;
```

### Explicit CPU Processing

```rust
let cpu_cugan = RealCugan::new_with_options(
    -1, // gpuid = -1 triggers CPU engine
    -1,
    2,
    RealCuganModelType::Se,
    400,
    3,
    1,
    false,
    2,
    "models".into(),
)?;

let result = cpu_cugan.proc_image_cpu(&image)?;
```

## License

MIT / BSD-3-Clause
