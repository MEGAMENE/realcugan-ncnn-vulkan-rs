#if _WIN32
#include <locale>
#include <codecvt>
#include <string>
#endif

#include "realcugan.h"

#include <atomic>
#include <mutex>

typedef struct Image {
    unsigned char *data;
    int w;
    int h;
    int c;
} Image;

extern "C" RealCUGAN *realcugan_init(
        int gpuid,
        bool tta_mode,
        int num_threads,
        int noise,
        int scale,
        int tilesize,
        int prepadding,
        int syncgap
) {
    auto realcugan = new RealCUGAN(gpuid, tta_mode, num_threads);
    realcugan->noise = noise;
    realcugan->scale = scale;
    realcugan->tilesize = tilesize;
    realcugan->prepadding = prepadding;
    realcugan->syncgap = syncgap;
    realcugan->bgr_mode = false;
    return realcugan;
}

static std::atomic<int> s_gpu_instance_refcount{0};
static std::mutex s_gpu_instance_mutex;

extern "C" void realcugan_init_gpu_instance() {
    std::lock_guard<std::mutex> lock(s_gpu_instance_mutex);
    if (s_gpu_instance_refcount == 0) {
        ncnn::create_gpu_instance();
    }
    s_gpu_instance_refcount++;
}
extern "C" int realcugan_get_gpu_count() {
    return ncnn::get_gpu_count();
}

extern "C" void realcugan_destroy_gpu_instance() {
    std::lock_guard<std::mutex> lock(s_gpu_instance_mutex);
    if (s_gpu_instance_refcount > 0) {
        s_gpu_instance_refcount--;
        if (s_gpu_instance_refcount == 0) {
            ncnn::destroy_gpu_instance();
        }
    }
}

extern "C" int realcugan_load(RealCUGAN *realcugan, const char *param_path, const char *model_path) {
#if _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return realcugan->load(converter.from_bytes(param_path), converter.from_bytes(model_path));
#else
    return realcugan->load(param_path, model_path);
#endif
}

extern "C" int realcugan_process(RealCUGAN *realcugan, const Image *in_image, Image *out_image, void **mat_ptr) {
    if (!realcugan || !in_image || !out_image || !in_image->data || !out_image->data) {
        return -1;
    }
    int c = in_image->c;
    ncnn::Mat in_image_mat(in_image->w, in_image->h, (void *) in_image->data, (size_t) c, c);
    ncnn::Mat out_image_mat(out_image->w, out_image->h, (void *) out_image->data, (size_t) c, c);

    int result = realcugan->process(in_image_mat, out_image_mat);
    if (mat_ptr) {
        *mat_ptr = nullptr;
    }
    return result;
}

extern "C" int realcugan_process_cpu(RealCUGAN *realcugan, const Image *in_image, Image *out_image, void **mat_ptr) {
    if (!realcugan || !in_image || !out_image || !in_image->data || !out_image->data) {
        return -1;
    }
    int c = in_image->c;
    ncnn::Mat in_image_mat(in_image->w, in_image->h, (void *) in_image->data, (size_t) c, c);
    ncnn::Mat out_image_mat(out_image->w, out_image->h, (void *) out_image->data, (size_t) c, c);

    int result = realcugan->process_cpu(in_image_mat, out_image_mat);
    if (mat_ptr) {
        *mat_ptr = nullptr;
    }
    return result;
}

extern "C" uint32_t realcugan_get_heap_budget(int gpuid) {
    if (gpuid < 0) return 0;
    auto* dev = ncnn::get_gpu_device(gpuid);
    if (!dev) return 0;
    return dev->get_heap_budget();
}

extern "C" void realcugan_free_image(ncnn::Mat *mat_ptr) {
    if (mat_ptr) {
        delete mat_ptr;
    }
}

extern "C" void realcugan_free(RealCUGAN *realcugan) {
    if (realcugan) {
        delete realcugan;
    }
    realcugan_destroy_gpu_instance();
}


