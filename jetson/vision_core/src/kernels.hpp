// §5.5.2 / §5.5.4 CUDA kernels (kernels.cu). All sources are pitch-linear RGBA NvBuffers mapped once via EGL; the
// destination element type follows the engine binding (FP16 or FP32), so any trtexec I/O format works.
#pragma once
#include <cuda_runtime.h>

namespace vc {

struct Norm {  // out = (px - mean) * scale
  float mean, scale;
};
constexpr Norm kYolo{0.f, 1.f / 255.f}, kInsight{127.5f, 1.f / 128.f};

// RGBA (W x H) -> NCHW plane set at dst. `gray` (device int, 0/1) switches to luma x3 for IR-lit night frames without
// changing kernel arguments, so the captured CUDA graph stays valid (§5.5.1 night logic).
void preprocess(const void* rgba, int pitch_bytes, int W, int H, void* dst, bool half, Norm n, const int* gray,
                cudaStream_t s);

// Bilinear crop+resize of up to 8 source rectangles (x1 y1 x2 y2, source px) into `out`x`out` letterboxed NCHW
// slots (SCRFD head crops). Padding is the normalised zero.
struct Crop {
  float x1, y1, x2, y2;
};
void crop_resize(const void* rgba, int pitch_bytes, int SW, int SH, const Crop* crops, int n, int out, void* dst,
                 bool half, Norm nm, cudaStream_t s);

// Stretched crops -> OW x OH NCHW with ImageNet mean/std (OSNet ReID, §7.6).
void crop_stretch(const void* rgba, int pitch_bytes, int SW, int SH, const Crop* crops, int n, int OW, int OH,
                  void* dst, bool half, cudaStream_t s);

// 5-point aligned 112x112 faces (§5.5.4): M[f] maps template px -> source px (vc::similarity).
void warp_faces(const void* rgba, int pitch_bytes, int SW, int SH, const float (*M)[6], int n, void* dst, bool half,
                cudaStream_t s);

}  // namespace vc
