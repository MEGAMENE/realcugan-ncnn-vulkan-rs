#include "realcugan.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <new>

namespace {

constexpr int32_t RC_OK = 0;
constexpr int32_t RC_ERR_INVALID_ARG = -1;
constexpr int32_t RC_ERR_GPU_INIT_FAILED = -2;
constexpr int32_t RC_ERR_INVALID_GPU = -3;
constexpr int32_t RC_ERR_LOAD_FAILED = -4;
constexpr int32_t RC_ERR_PROCESS_FAILED = -5;
constexpr int32_t RC_ERR_OUTPUT_MISMATCH = -6;
constexpr int32_t RC_ERR_EXCEPTION = -99;

bool ensure_gpu_instance() {
    // ncnn::create_gpu_instance() is internally mutex-guarded and idempotent.
    // We intentionally never call ncnn::destroy_gpu_instance(); ncnn registers
    // atexit(destroy_gpu_instance) and a static holder destructor for orderly
    // process-exit teardown without breaking concurrent or sibling ncnn users.
    const int ret = ncnn::create_gpu_instance();
    return ret == 0 && ncnn::get_gpu_instance() != nullptr;
}

ncnn::VulkanDevice *get_valid_gpu_device(int32_t gpuid) {
    if (gpuid < 0) {
        return nullptr;
    }
    if (!ensure_gpu_instance()) {
        return nullptr;
    }
    if (gpuid >= ncnn::get_gpu_count()) {
        return nullptr;
    }
    ncnn::VulkanDevice *dev = ncnn::get_gpu_device(gpuid);
    if (!dev || !dev->is_valid()) {
        return nullptr;
    }
    return dev;
}

bool is_valid_scale(int32_t scale) {
    return scale == 2 || scale == 3 || scale == 4;
}

bool is_valid_tta_level(int32_t tta_level) {
    return tta_level == 1 || tta_level == 2 || tta_level == 4 || tta_level == 8;
}

bool is_valid_syncgap(int32_t syncgap) {
    return syncgap >= 0 && syncgap <= 3;
}

} // namespace

#if defined(_WIN32)
using RealCuganPathChar = wchar_t;
#else
using RealCuganPathChar = char;
#endif

struct RealCuganHandle {
    RealCUGAN engine;

    explicit RealCuganHandle(const RealCUGANConfig &cfg) : engine(cfg) {}
};

struct RealCuganFfiConfig {
    int32_t gpuid;
    int32_t scale;
    int32_t tilesize;
    int32_t syncgap;
    int32_t tta_level;
    int32_t num_threads;
    int32_t concurrent_jobs;
    uint8_t is_pro_model;
    uint8_t use_fp16_arithmetic;
};

extern "C" int32_t realcugan_create(
        const RealCuganFfiConfig *cfg,
        const RealCuganPathChar *param_path,
        const RealCuganPathChar *model_path,
        RealCuganHandle **out_handle
) {
    if (out_handle) {
        *out_handle = nullptr;
    }
    if (!cfg || !param_path || !model_path || !out_handle) {
        return RC_ERR_INVALID_ARG;
    }
    if (param_path[0] == static_cast<RealCuganPathChar>(0) ||
        model_path[0] == static_cast<RealCuganPathChar>(0)) {
        return RC_ERR_INVALID_ARG;
    }
    if (cfg->gpuid < -1 ||
        !is_valid_scale(cfg->scale) ||
        (cfg->tilesize != 0 && cfg->tilesize < 32) ||
        !is_valid_syncgap(cfg->syncgap) ||
        !is_valid_tta_level(cfg->tta_level) ||
        cfg->num_threads < 1 ||
        cfg->concurrent_jobs < 1) {
        return RC_ERR_INVALID_ARG;
    }

    try {
        if (cfg->gpuid >= 0) {
            if (!ensure_gpu_instance()) {
                return RC_ERR_GPU_INIT_FAILED;
            }
            if (cfg->gpuid >= ncnn::get_gpu_count()) {
                return RC_ERR_INVALID_GPU;
            }
            const ncnn::VulkanDevice *dev = ncnn::get_gpu_device(cfg->gpuid);
            if (!dev || !dev->is_valid()) {
                return RC_ERR_INVALID_GPU;
            }
        }

        RealCUGANConfig engine_cfg;
        engine_cfg.gpuid = cfg->gpuid;
        engine_cfg.scale = cfg->scale;
        engine_cfg.tilesize = cfg->tilesize;
        engine_cfg.syncgap = cfg->syncgap;
        engine_cfg.tta_level = cfg->tta_level;
        engine_cfg.num_threads = cfg->num_threads;
        engine_cfg.concurrent_jobs = cfg->concurrent_jobs;
        engine_cfg.is_pro_model = (cfg->is_pro_model != 0);
        // Rust `image::DynamicImage` buffers are always RGB/RGBA; override Win32 BGR default.
        engine_cfg.bgr_mode = false;
        engine_cfg.use_fp16_arithmetic = (cfg->use_fp16_arithmetic != 0);

        auto *handle = new (std::nothrow) RealCuganHandle(engine_cfg);
        if (!handle) {
            return RC_ERR_LOAD_FAILED;
        }

        const std::filesystem::path param_fs(param_path);
        const std::filesystem::path model_fs(model_path);
        if (handle->engine.load(param_fs, model_fs) != 0) {
            delete handle;
            return RC_ERR_LOAD_FAILED;
        }

        *out_handle = handle;
        return RC_OK;
    } catch (...) {
        return RC_ERR_EXCEPTION;
    }
}

extern "C" void realcugan_destroy(RealCuganHandle *handle) {
    try {
        delete handle;
    } catch (...) {
    }
}

extern "C" int32_t realcugan_process(
        const RealCuganHandle *handle,
        const uint8_t *in_data,
        int32_t in_w,
        int32_t in_h,
        uint8_t *out_data,
        int32_t out_w,
        int32_t out_h,
        int32_t channels
) {
    if (!handle || !in_data || !out_data || in_data == out_data) {
        return RC_ERR_INVALID_ARG;
    }
    if (in_w <= 0 || in_h <= 0 || (channels != 3 && channels != 4)) {
        return RC_ERR_INVALID_ARG;
    }

    try {
        const int scale = handle->engine.config().scale;
        if (!is_valid_scale(scale)) {
            return RC_ERR_INVALID_ARG;
        }

        if (in_w > INT_MAX / scale || in_h > INT_MAX / scale) {
            return RC_ERR_INVALID_ARG;
        }
        if (out_w != in_w * scale || out_h != in_h * scale) {
            return RC_ERR_INVALID_ARG;
        }

        // Guard signed 32-bit strides/offsets inside Layer 1 and ncnn::Mat::from_pixels_roi / to_pixels
        if (in_w > INT_MAX / channels || in_h > INT_MAX / (in_w * channels)) {
            return RC_ERR_INVALID_ARG;
        }
        if (out_w > INT_MAX / channels || out_w > INT_MAX / out_h) {
            return RC_ERR_INVALID_ARG;
        }
        if (static_cast<size_t>(out_w) * static_cast<size_t>(channels) >
            SIZE_MAX / static_cast<size_t>(out_h)) {
            return RC_ERR_INVALID_ARG;
        }

        ncnn::Mat in_mat(
                in_w,
                in_h,
                const_cast<uint8_t *>(in_data),
                static_cast<size_t>(channels),
                channels
        );
        ncnn::Mat out_mat(
                out_w,
                out_h,
                static_cast<void *>(out_data),
                static_cast<size_t>(channels),
                channels
        );

        const int ret = handle->engine.process(in_mat, out_mat);
        if (ret != 0) {
            return ret < 0 ? static_cast<int32_t>(ret) : RC_ERR_PROCESS_FAILED;
        }
        if (out_mat.data != static_cast<void *>(out_data)) {
            return RC_ERR_OUTPUT_MISMATCH;
        }
        return RC_OK;
    } catch (...) {
        return RC_ERR_EXCEPTION;
    }
}

extern "C" int32_t realcugan_get_gpu_count(void) {
    try {
        if (!ensure_gpu_instance()) {
            return 0;
        }
        return ncnn::get_gpu_count();
    } catch (...) {
        return 0;
    }
}

extern "C" int32_t realcugan_get_default_gpu_index(void) {
    try {
        if (!ensure_gpu_instance()) {
            return -1;
        }
        return ncnn::get_default_gpu_index();
    } catch (...) {
        return -1;
    }
}

extern "C" uint32_t realcugan_get_heap_budget(int32_t gpuid) {
    try {
        const ncnn::VulkanDevice *dev = get_valid_gpu_device(gpuid);
        if (!dev) {
            return 0;
        }
        return dev->get_heap_budget();
    } catch (...) {
        return 0;
    }
}

extern "C" int32_t realcugan_get_gpu_device_type(int32_t gpuid, int32_t *out_type) {
    if (!out_type) {
        return RC_ERR_INVALID_ARG;
    }
    try {
        const ncnn::VulkanDevice *dev = get_valid_gpu_device(gpuid);
        if (!dev) {
            return RC_ERR_INVALID_GPU;
        }
        *out_type = dev->info.type();
        return RC_OK;
    } catch (...) {
        return RC_ERR_EXCEPTION;
    }
}

extern "C" int32_t realcugan_should_use_cpu(int32_t gpuid) {
    try {
        if (gpuid < 0) {
            return 1;
        }
        const ncnn::VulkanDevice *dev = get_valid_gpu_device(gpuid);
        return RealCUGAN::should_use_cpu(dev) ? 1 : 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int32_t realcugan_compute_auto_tilesize(
        int32_t gpuid,
        int32_t scale,
        int32_t tta_level,
        int32_t concurrent_jobs
) {
    if (!is_valid_scale(scale) || !is_valid_tta_level(tta_level) || concurrent_jobs < 1) {
        return RC_ERR_INVALID_ARG;
    }
    try {
        const ncnn::VulkanDevice *dev = (gpuid < 0) ? nullptr : get_valid_gpu_device(gpuid);
        return RealCUGAN::compute_auto_tilesize(dev, scale, tta_level, concurrent_jobs);
    } catch (...) {
        return RC_ERR_EXCEPTION;
    }
}

extern "C" int32_t realcugan_default_prepadding(int32_t scale) {
    try {
        return RealCUGAN::default_prepadding(scale);
    } catch (...) {
        return 0;
    }
}
