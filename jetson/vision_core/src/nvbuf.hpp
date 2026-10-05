// §5.5.6 NvBuffer (dmabuf) pools shared by the camera, VIC, NVENC, NVJPG and CUDA. Fds are allocated once at
// start-up, so the EGL->CUDA map cache stays valid and nothing allocates on the frame path.
#pragma once
#include <nvbuf_utils.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace vc {

inline int nvbuf_create(int w, int h, NvBufferColorFormat fmt, NvBufferLayout lay, NvBufferTag tag) {
  NvBufferCreateParams p = {};
  p.width = uint32_t(w), p.height = uint32_t(h);
  p.payloadType = NvBufferPayload_SurfArray;
  p.layout = lay, p.colorFormat = fmt, p.nvbuf_tag = tag;
  int fd = -1;
  return NvBufferCreateEx(&fd, &p) == 0 ? fd : -1;
}

// §4 item 13: one VIC session per thread, so the camera, infer, worker and control threads' transforms run in
// parallel instead of queueing on the default session. Created on first use, destroyed at thread exit.
inline NvBufferSession vic_session() {
  struct Holder {
    NvBufferSession s = NvBufferSessionCreate();
    ~Holder() {
      if (s) NvBufferSessionDestroy(s);
    }
  };
  static thread_local Holder h;
  return h.s;
}

// Full-frame VIC scale / colour conversion / layout change (~0.2-1 ms, no CPU or GPU time).
inline bool vic_blit(int src, int dst) {
  NvBufferTransformParams p = {};
  p.session = vic_session();
  p.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  p.transform_filter = NvBufferTransform_Filter_Smart;
  return NvBufferTransform(src, dst, &p) == 0;
}

// VIC crop (source pixels) + scale into dst: memory thumbnails.
inline bool vic_crop(int src, int dst, int x, int y, int w, int h) {
  NvBufferTransformParams p = {};
  p.session = vic_session();
  p.transform_flag = NVBUFFER_TRANSFORM_FILTER | NVBUFFER_TRANSFORM_CROP_SRC;
  p.transform_filter = NvBufferTransform_Filter_Smart;
  p.src_rect.top = uint32_t(y), p.src_rect.left = uint32_t(x), p.src_rect.width = uint32_t(w);
  p.src_rect.height = uint32_t(h);
  return NvBufferTransform(src, dst, &p) == 0;
}

// Reference-counted slots: a camera frame can be held by the recorder, the teleop compositor and a snapshot at once.
class FdPool {
 public:
  ~FdPool() {
    for (int fd : fds_) NvBufferDestroy(fd);
  }
  bool create(int n, int w, int h, NvBufferColorFormat fmt, NvBufferLayout lay, NvBufferTag tag) {
    refs_.reset(new std::atomic<int>[size_t(n)]);
    for (int i = 0; i < n; ++i) {
      refs_[size_t(i)] = 0;
      int fd = nvbuf_create(w, h, fmt, lay, tag);
      if (fd < 0) return false;
      fds_.push_back(fd);
    }
    w_ = w, h_ = h;
    return true;
  }
  int acquire() {  // slot with one reference, or -1 when every slot is busy (the caller drops the frame)
    for (size_t i = 0; i < fds_.size(); ++i) {
      int z = 0;
      if (refs_[i].compare_exchange_strong(z, 1)) return int(i);
    }
    return -1;
  }
  void ref(int s) { refs_[size_t(s)].fetch_add(1); }
  void unref(int s) { refs_[size_t(s)].fetch_sub(1); }
  int fd(int s) const { return fds_[size_t(s)]; }
  int slot_of(int fd) const {
    for (size_t i = 0; i < fds_.size(); ++i)
      if (fds_[i] == fd) return int(i);
    return -1;
  }
  int w() const { return w_; }
  int h() const { return h_; }

 private:
  std::vector<int> fds_;
  std::unique_ptr<std::atomic<int>[]> refs_;
  int w_ = 0, h_ = 0;
};

// "Newest frame" holder over a pool: set() takes over one reference, get() hands the caller a new one (or -1).
class Latest {
 public:
  explicit Latest(FdPool& p) : pool_(p) {}
  ~Latest() { set(-1); }
  void set(int slot) {
    std::lock_guard<std::mutex> g(mu_);
    if (slot_ >= 0) pool_.unref(slot_);
    slot_ = slot;
  }
  int get() {
    std::lock_guard<std::mutex> g(mu_);
    if (slot_ >= 0) pool_.ref(slot_);
    return slot_;
  }
  FdPool& pool() { return pool_; }

 private:
  FdPool& pool_;
  std::mutex mu_;
  int slot_ = -1;
};

// Lock-free-ish triple buffer of fixed fds: the camera writes back(), publish() makes it the newest; the reader's
// take() returns the newest fd and keeps it stable until its next take().
class TripleFd {
 public:
  bool create(int w, int h, NvBufferColorFormat fmt) {
    for (int& fd : fd_)
      if ((fd = nvbuf_create(w, h, fmt, NvBufferLayout_Pitch, NvBufferTag_CAMERA)) < 0) return false;
    return true;
  }
  ~TripleFd() {
    for (int fd : fd_)
      if (fd >= 0) NvBufferDestroy(fd);
  }
  int back() const { return fd_[back_]; }
  void publish() {
    std::lock_guard<std::mutex> g(mu_);
    std::swap(back_, mid_);
    fresh_ = true;
  }
  int take(bool* was_fresh = nullptr) {
    std::lock_guard<std::mutex> g(mu_);
    if (was_fresh) *was_fresh = fresh_;
    if (fresh_) std::swap(front_, mid_), fresh_ = false;
    return fd_[front_];
  }

 private:
  int fd_[3] = {-1, -1, -1};
  int back_ = 0, mid_ = 1, front_ = 2;
  bool fresh_ = false;
  std::mutex mu_;
};

}  // namespace vc
