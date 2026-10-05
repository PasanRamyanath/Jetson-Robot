// §5.5.2 / §5.5.4 CUDA kernels for sm_53 (CUDA 10.2). Memory-bound: one thread per output pixel, coalesced writes.
#include <cuda_fp16.h>

#include "kernels.hpp"

namespace vc {
namespace {

__device__ __forceinline__ void put(float* d, int i, float v) { d[i] = v; }
__device__ __forceinline__ void put(__half* d, int i, float v) { d[i] = __float2half(v); }

__device__ __forceinline__ uchar4 px(const unsigned char* base, int pitch, int x, int y) {
  return reinterpret_cast<const uchar4*>(base + size_t(y) * pitch)[x];
}

template <typename T>
__global__ void k_preprocess(const unsigned char* __restrict__ src, int pitch, int W, int H, T* __restrict__ dst,
                             Norm n, const int* __restrict__ gray) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  uchar4 p = px(src, pitch, x, y);
  float r = p.x, g = p.y, b = p.z;
  if (gray && *gray) r = g = b = 0.299f * r + 0.587f * g + 0.114f * b;
  int plane = W * H, i = y * W + x;
  put(dst, i, (r - n.mean) * n.scale);
  put(dst, i + plane, (g - n.mean) * n.scale);
  put(dst, i + 2 * plane, (b - n.mean) * n.scale);
}

__device__ __forceinline__ float3 bilinear(const unsigned char* src, int pitch, int SW, int SH, float sx, float sy) {
  int x0 = floorf(sx), y0 = floorf(sy);
  float ax = sx - x0, ay = sy - y0;
  float3 acc = make_float3(0, 0, 0);
#pragma unroll
  for (int dy = 0; dy < 2; ++dy)
#pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
      int xx = min(max(x0 + dx, 0), SW - 1), yy = min(max(y0 + dy, 0), SH - 1);
      float w = (dx ? ax : 1 - ax) * (dy ? ay : 1 - ay);
      uchar4 p = px(src, pitch, xx, yy);
      acc.x += w * p.x, acc.y += w * p.y, acc.z += w * p.z;
    }
  return acc;
}

struct Crops {
  Crop c[8];
};

template <typename T>
__global__ void k_crop(const unsigned char* __restrict__ src, int pitch, int SW, int SH, Crops cs, int out,
                       T* __restrict__ dst, Norm n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, f = blockIdx.z;
  if (x >= out || y >= out) return;
  Crop c = cs.c[f];
  float cw = c.x2 - c.x1, ch = c.y2 - c.y1, s = fminf(out / cw, out / ch);
  float px0 = 0.5f * (out - cw * s), py0 = 0.5f * (out - ch * s);
  float sx = (x + 0.5f - px0) / s - 0.5f + c.x1, sy = (y + 0.5f - py0) / s - 0.5f + c.y1;
  bool inside = sx >= c.x1 - 0.5f && sx <= c.x2 - 0.5f && sy >= c.y1 - 0.5f && sy <= c.y2 - 0.5f;
  float3 v = inside ? bilinear(src, pitch, SW, SH, sx, sy) : make_float3(n.mean, n.mean, n.mean);
  int plane = out * out, i = f * 3 * plane + y * out + x;
  put(dst, i, (v.x - n.mean) * n.scale);
  put(dst, i + plane, (v.y - n.mean) * n.scale);
  put(dst, i + 2 * plane, (v.z - n.mean) * n.scale);
}

// ReID crops: stretched (not letterboxed) as OSNet was trained, ImageNet mean/std per channel.
template <typename T>
__global__ void k_stretch(const unsigned char* __restrict__ src, int pitch, int SW, int SH, Crops cs, int OW, int OH,
                          T* __restrict__ dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, f = blockIdx.z;
  if (x >= OW || y >= OH) return;
  Crop c = cs.c[f];
  float sx = c.x1 + (x + 0.5f) * (c.x2 - c.x1) / OW - 0.5f, sy = c.y1 + (y + 0.5f) * (c.y2 - c.y1) / OH - 0.5f;
  float3 v = bilinear(src, pitch, SW, SH, sx, sy);
  int plane = OW * OH, i = f * 3 * plane + y * OW + x;
  put(dst, i, (v.x - 123.675f) * (1.f / 58.395f));
  put(dst, i + plane, (v.y - 116.28f) * (1.f / 57.12f));
  put(dst, i + 2 * plane, (v.z - 103.53f) * (1.f / 57.375f));
}

struct Warps {  // by value, not __constant__: the live and replay face chains warp concurrently on two streams
  float m[8][6];
};

template <typename T>
__global__ void k_warp(const unsigned char* __restrict__ src, int pitch, int SW, int SH, T* __restrict__ dst,
                       Warps ws, int nFaces) {
  int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y, f = blockIdx.z;
  if (x >= 112 || y >= 112 || f >= nFaces) return;
  const float* m = ws.m[f];
  float sx = m[0] * x + m[1] * y + m[2], sy = m[3] * x + m[4] * y + m[5];
  float3 v = bilinear(src, pitch, SW, SH, sx, sy);
  int plane = 112 * 112, i = f * 3 * plane + y * 112 + x;
  put(dst, i, (v.x - 127.5f) / 128.f);
  put(dst, i + plane, (v.y - 127.5f) / 128.f);
  put(dst, i + 2 * plane, (v.z - 127.5f) / 128.f);
}

}  // namespace

void preprocess(const void* rgba, int pitch, int W, int H, void* dst, bool half, Norm n, const int* gray,
                cudaStream_t s) {
  dim3 b(32, 8), g((W + 31) / 32, (H + 7) / 8);
  auto src = static_cast<const unsigned char*>(rgba);
  if (half) k_preprocess<<<g, b, 0, s>>>(src, pitch, W, H, static_cast<__half*>(dst), n, gray);
  else k_preprocess<<<g, b, 0, s>>>(src, pitch, W, H, static_cast<float*>(dst), n, gray);
}

void crop_resize(const void* rgba, int pitch, int SW, int SH, const Crop* crops, int n, int out, void* dst, bool half,
                 Norm nm, cudaStream_t s) {
  if (n <= 0) return;
  Crops cs;
  for (int i = 0; i < n && i < 8; ++i) cs.c[i] = crops[i];
  dim3 b(32, 8), g((out + 31) / 32, (out + 7) / 8, n < 8 ? n : 8);
  auto src = static_cast<const unsigned char*>(rgba);
  if (half) k_crop<<<g, b, 0, s>>>(src, pitch, SW, SH, cs, out, static_cast<__half*>(dst), nm);
  else k_crop<<<g, b, 0, s>>>(src, pitch, SW, SH, cs, out, static_cast<float*>(dst), nm);
}

void crop_stretch(const void* rgba, int pitch, int SW, int SH, const Crop* crops, int n, int OW, int OH, void* dst,
                  bool half, cudaStream_t s) {
  if (n <= 0) return;
  Crops cs;
  for (int i = 0; i < n && i < 8; ++i) cs.c[i] = crops[i];
  dim3 b(32, 8), g((OW + 31) / 32, (OH + 7) / 8, n < 8 ? n : 8);
  auto src = static_cast<const unsigned char*>(rgba);
  if (half) k_stretch<<<g, b, 0, s>>>(src, pitch, SW, SH, cs, OW, OH, static_cast<__half*>(dst));
  else k_stretch<<<g, b, 0, s>>>(src, pitch, SW, SH, cs, OW, OH, static_cast<float*>(dst));
}

void warp_faces(const void* rgba, int pitch, int SW, int SH, const float (*M)[6], int n, void* dst, bool half,
                cudaStream_t s) {
  if (n <= 0) return;
  n = n < 8 ? n : 8;
  Warps ws;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 6; ++k) ws.m[i][k] = M[i][k];
  dim3 b(16, 16), g(7, 7, n);
  auto src = static_cast<const unsigned char*>(rgba);
  if (half) k_warp<<<g, b, 0, s>>>(src, pitch, SW, SH, static_cast<__half*>(dst), ws, n);
  else k_warp<<<g, b, 0, s>>>(src, pitch, SW, SH, static_cast<float*>(dst), ws, n);
}

}  // namespace vc
