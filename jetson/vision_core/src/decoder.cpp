// §11.9 sleep replay: NvVideoDecoder wrapper (MMAPI 00_video_decode pattern, L4T 32.7), non-blocking V4L2 fd.
#include "decoder.hpp"

#include <NvVideoDecoder.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace vc {

namespace {
constexpr uint32_t kInBufs = 8, kInSize = 2 << 20;  // 2 MB: a 720p IDR is ~100-300 KB at 3 Mb/s
}

bool Decoder::open(bool hevc, FrameFn fn) {
  close();
  fn_ = std::move(fn);
  dec_ = NvVideoDecoder::createVideoDecoder("replay", O_NONBLOCK);
  if (!dec_) return std::fprintf(stderr, "[replay] createVideoDecoder failed\n"), false;
  bool ok = dec_->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0) == 0 &&
            dec_->setOutputPlaneFormat(hevc ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_H264, kInSize) == 0 &&
            dec_->setFrameInputMode(0) == 0 &&  // one complete access unit per buffer (TsDemux gives exactly that)
            dec_->output_plane.setupPlane(V4L2_MEMORY_MMAP, kInBufs, true, false) == 0 &&
            dec_->output_plane.setStreamStatus(true) == 0;
  if (!ok) return std::fprintf(stderr, "[replay] decoder setup failed\n"), close(), false;
  next_in_ = 0, cap_ = false, eos_ = false;
  return true;
}

bool Decoder::setup_capture() {
  v4l2_format f;
  v4l2_crop crop;
  std::memset(&f, 0, sizeof f), std::memset(&crop, 0, sizeof crop);
  if (dec_->capture_plane.getFormat(f) < 0 || dec_->capture_plane.getCrop(crop) < 0) return false;
  w_ = int(crop.c.width), h_ = int(crop.c.height);
  int min = 0;
  if (dec_->setCapturePlaneFormat(f.fmt.pix_mp.pixelformat, f.fmt.pix_mp.width, f.fmt.pix_mp.height) < 0 ||
      dec_->getMinimumCapturePlaneBuffers(min) < 0 ||
      dec_->capture_plane.setupPlane(V4L2_MEMORY_MMAP, uint32_t(min + 4), false, false) < 0 ||
      dec_->capture_plane.setStreamStatus(true) < 0)
    return false;
  for (uint32_t i = 0; i < dec_->capture_plane.getNumBuffers(); ++i) {
    v4l2_buffer b;
    v4l2_plane planes[MAX_PLANES];
    std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
    b.index = i, b.m.planes = planes;
    if (dec_->capture_plane.qBuffer(b, nullptr) < 0) return false;
  }
  std::fprintf(stderr, "[replay] decoding %dx%d\n", w_, h_);
  return cap_ = true;
}

// Frames ready now; false once the end-of-stream frame (bytesused 0) has come out.
bool Decoder::drain() {
  if (!cap_) {
    v4l2_event ev;
    std::memset(&ev, 0, sizeof ev);
    if (dec_->dqEvent(ev, 0) == 0 && ev.type == V4L2_EVENT_RESOLUTION_CHANGE && !setup_capture())
      std::fprintf(stderr, "[replay] capture plane setup failed\n");
    if (!cap_) return true;
  }
  for (;;) {
    v4l2_buffer b;
    v4l2_plane planes[MAX_PLANES];
    std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
    b.m.planes = planes;
    NvBuffer* nb = nullptr;
    if (dec_->capture_plane.dqBuffer(b, &nb, nullptr, 0) < 0) return true;  // EAGAIN: nothing decoded yet
    if (planes[0].bytesused == 0) return false;
    uint64_t us = uint64_t(b.timestamp.tv_sec) * 1000000ull + uint64_t(b.timestamp.tv_usec);
    if (fn_) fn_(nb->planes[0].fd, w_, h_, us * 9 / 100);
    dec_->capture_plane.qBuffer(b, nullptr);
  }
}

bool Decoder::push(const uint8_t* au, size_t n, uint64_t pts90k) {
  if (!dec_ || eos_) return false;
  if (n > kInSize) return true;  // skip a corrupt oversize AU, keep going
  v4l2_buffer b;
  v4l2_plane planes[MAX_PLANES];
  std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
  b.m.planes = planes;
  NvBuffer* nb = nullptr;
  if (next_in_ < dec_->output_plane.getNumBuffers()) {
    b.index = next_in_++;
    nb = dec_->output_plane.getNthBuffer(b.index);
  } else {
    for (int i = 0; dec_->output_plane.dqBuffer(b, &nb, nullptr, 0) < 0; ++i) {
      drain();
      if (i >= 2000) return false;
      usleep(1000);
    }
  }
  std::memcpy(nb->planes[0].data, au, n);
  nb->planes[0].bytesused = uint32_t(n);
  planes[0].bytesused = uint32_t(n);
  uint64_t us = pts90k * 100 / 9;
  b.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
  b.timestamp.tv_sec = long(us / 1000000), b.timestamp.tv_usec = long(us % 1000000);
  if (dec_->output_plane.qBuffer(b, nullptr) < 0) return false;
  drain();
  return true;
}

void Decoder::finish() {
  if (!dec_ || eos_) return;
  eos_ = true;
  v4l2_buffer b;
  v4l2_plane planes[MAX_PLANES];
  std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
  b.m.planes = planes;
  NvBuffer* nb = nullptr;
  if (next_in_ < dec_->output_plane.getNumBuffers()) b.index = next_in_++;
  else if (dec_->output_plane.dqBuffer(b, &nb, nullptr, 1000) < 0) return;
  planes[0].bytesused = 0;  // empty buffer = end of stream
  dec_->output_plane.qBuffer(b, nullptr);
  for (int i = 0; i < 2000 && drain(); ++i) usleep(1000);
}

void Decoder::close() {
  if (!dec_) return;
  dec_->output_plane.setStreamStatus(false);
  if (cap_) dec_->capture_plane.setStreamStatus(false);
  delete dec_;
  dec_ = nullptr, cap_ = false;
}

}  // namespace vc
