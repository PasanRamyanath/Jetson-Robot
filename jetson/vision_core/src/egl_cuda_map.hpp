// §5.5.2 zero-copy NvBuffer fd -> CUDA pointer, cached per fd (buffer pools recycle a fixed set of fds).
#pragma once
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <nvbuf_utils.h>

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace vc {

struct Mapped {
  const void* ptr = nullptr;  // plane 0, pitch-linear
  int pitch = 0;              // bytes
  int w = 0, h = 0;
};

class EglCudaMapper {
 public:
  EglCudaMapper() {
    dpy_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy_ == EGL_NO_DISPLAY || !eglInitialize(dpy_, nullptr, nullptr)) std::fprintf(stderr, "[egl] no display\n");
    cudaFree(nullptr);  // make the runtime's primary context current for the driver-API calls below
  }
  ~EglCudaMapper() {
    clear();
    if (dpy_ != EGL_NO_DISPLAY) eglTerminate(dpy_);
  }
  EglCudaMapper(const EglCudaMapper&) = delete;
  EglCudaMapper& operator=(const EglCudaMapper&) = delete;

  EGLDisplay display() const { return dpy_; }

  // nullptr ptr on failure; thread-safe. Callers must only pass fds from long-lived pools (entries are never evicted).
  Mapped map(int fd) {
    std::lock_guard<std::mutex> g(mu_);  // shared by the infer and ReID threads
    auto it = cache_.find(fd);
    if (it != cache_.end()) return it->second.m;
    Entry e;
    e.img = NvEGLImageFromFd(dpy_, fd);
    if (!e.img) return Mapped{};
    CUeglFrame fr;
    if (cuGraphicsEGLRegisterImage(&e.res, e.img, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS ||
        cuGraphicsResourceGetMappedEglFrame(&fr, e.res, 0, 0) != CUDA_SUCCESS ||
        fr.frameType != CU_EGL_FRAME_TYPE_PITCH) {
      if (e.res) cuGraphicsUnregisterResource(e.res);
      NvDestroyEGLImage(dpy_, e.img);
      return Mapped{};
    }
    e.m.ptr = fr.frame.pPitch[0];
    e.m.pitch = int(fr.pitch);
    e.m.w = int(fr.width), e.m.h = int(fr.height);
    return cache_.emplace(fd, e).first->second.m;
  }

  void clear() {
    std::lock_guard<std::mutex> g(mu_);
    cudaDeviceSynchronize();
    for (auto& kv : cache_) {
      cuGraphicsUnregisterResource(kv.second.res);
      NvDestroyEGLImage(dpy_, kv.second.img);
    }
    cache_.clear();
  }

 private:
  struct Entry {
    EGLImageKHR img = nullptr;
    CUgraphicsResource res = nullptr;
    Mapped m;
  };
  EGLDisplay dpy_ = EGL_NO_DISPLAY;
  std::unordered_map<int, Entry> cache_;
  std::mutex mu_;
};

}  // namespace vc
