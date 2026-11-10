#if _WIN32
#include <locale>
#include <codecvt>
#include <string>
#endif

#include "realcugan.h"

#include <atomic>
#include <cmath>
#include <algorithm>
#include <mutex>

typedef struct Image {
    unsigned char *data;
    int w;
    int h;
    int c;
} Image;

extern "C" RealCUGAN *realcugan_init(
        int gpuid,
        int tta_level,
        int num_threads,
        int noise,
        int scale,
        int tilesize,
        int prepadding,
        int syncgap,
        bool use_fp16_arithmetic
) {
    try {
        if (tta_level != 1 && tta_level != 2 && tta_level != 4 && tta_level != 8) {
            tta_level = 1;
        }
        auto realcugan = new RealCUGAN(gpuid, tta_level, num_threads);
        realcugan->noise = noise;
        realcugan->scale = scale;
        realcugan->tilesize = tilesize;
        realcugan->prepadding = prepadding;
        realcugan->syncgap = syncgap;
        realcugan->bgr_mode = false;
        realcugan->use_fp16_arithmetic = use_fp16_arithmetic;
        return realcugan;
    } catch (...) {
        return nullptr;
    }
}

static std::atomic<int> s_gpu_instance_refcount{0};
static std::mutex s_gpu_instance_mutex;

extern "C" void realcugan_init_gpu_instance() {
    try {
        std::lock_guard<std::mutex> lock(s_gpu_instance_mutex);
        if (s_gpu_instance_refcount == 0) {
            ncnn::create_gpu_instance();
        }
        s_gpu_instance_refcount++;
    } catch (...) {
    }
}

extern "C" int realcugan_get_gpu_count() {
    try {
        return ncnn::get_gpu_count();
    } catch (...) {
        return 0;
    }
}

extern "C" void realcugan_destroy_gpu_instance() {
    try {
        std::lock_guard<std::mutex> lock(s_gpu_instance_mutex);
        if (s_gpu_instance_refcount > 0) {
            s_gpu_instance_refcount--;
            if (s_gpu_instance_refcount == 0) {
                ncnn::destroy_gpu_instance();
            }
        }
    } catch (...) {
    }
}

extern "C" int realcugan_load(RealCUGAN *realcugan, const char *param_path, const char *model_path) {
    if (!realcugan || !param_path || !model_path) {
        return -1;
    }
    try {
#if _WIN32
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        return realcugan->load(converter.from_bytes(param_path), converter.from_bytes(model_path));
#else
        return realcugan->load(param_path, model_path);
#endif
    } catch (...) {
        return -1;
    }
}

extern "C" int realcugan_process(RealCUGAN *realcugan, const Image *in_image, Image *out_image, void **mat_ptr) {
    if (!realcugan || !in_image || !out_image || !in_image->data || !out_image->data) {
        return -1;
    }
    try {
        int c = in_image->c;
        ncnn::Mat in_image_mat(in_image->w, in_image->h, (void *) in_image->data, (size_t) c, c);
        ncnn::Mat out_image_mat(out_image->w, out_image->h, (void *) out_image->data, (size_t) c, c);

        int result = realcugan->process(in_image_mat, out_image_mat);
        if (mat_ptr) {
            *mat_ptr = nullptr;
        }
        return result;
    } catch (...) {
        return -1;
    }
}

extern "C" int realcugan_process_cpu(RealCUGAN *realcugan, const Image *in_image, Image *out_image, void **mat_ptr) {
    if (!realcugan || !in_image || !out_image || !in_image->data || !out_image->data) {
        return -1;
    }
    try {
        int c = in_image->c;
        ncnn::Mat in_image_mat(in_image->w, in_image->h, (void *) in_image->data, (size_t) c, c);
        ncnn::Mat out_image_mat(out_image->w, out_image->h, (void *) out_image->data, (size_t) c, c);

        int result = realcugan->process_cpu(in_image_mat, out_image_mat);
        if (mat_ptr) {
            *mat_ptr = nullptr;
        }
        return result;
    } catch (...) {
        return -1;
    }
}

extern "C" uint32_t realcugan_get_heap_budget(int gpuid) {
    try {
        if (gpuid < 0) return 0;
        auto* dev = ncnn::get_gpu_device(gpuid);
        if (!dev) return 0;
        return dev->get_heap_budget();
    } catch (...) {
        return 0;
    }
}

extern "C" int realcugan_get_gpu_device_type(int gpuid) {
    try {
        if (gpuid < 0) return -1;
        auto* dev = ncnn::get_gpu_device(gpuid);
        if (!dev) return -1;
        return dev->info.type();
    } catch (...) {
        return -1;
    }
}

extern "C" int realcugan_calculate_tile_size(int gpuid, int scale, int prepadding) {
    try {
        if (gpuid < 0) return 400;
        auto* dev = ncnn::get_gpu_device(gpuid);
        if (!dev) return 400;
        uint32_t heap_budget_mb = dev->get_heap_budget();
        double safe_vram = (double)heap_budget_mb * 1024.0 * 1024.0 * 0.70;
        double bpp = 7004.0;
        if (scale == 2) bpp = 7004.0;
        else if (scale == 3) bpp = 18432.0;
        else if (scale == 4) bpp = 9011.0;

        int calculated_tile = (int)(std::sqrt(safe_vram / bpp) - 2.0 * prepadding);
        calculated_tile = (calculated_tile / 32) * 32;

        int max_tile = 1600;
        if (dev->info.type() == 1) { // integrated GPU
            max_tile = (scale == 3) ? 256 : 384;
        }
        return std::max(32, std::min(calculated_tile, max_tile));
    } catch (...) {
        return 400;
    }
}

extern "C" void realcugan_free_image(ncnn::Mat *mat_ptr) {
    try {
        if (mat_ptr) {
            delete mat_ptr;
        }
    } catch (...) {
    }
}

extern "C" void realcugan_free(RealCUGAN *realcugan) {
    try {
        if (realcugan) {
            delete realcugan;
        }
    } catch (...) {
    }
    realcugan_destroy_gpu_instance();
}
