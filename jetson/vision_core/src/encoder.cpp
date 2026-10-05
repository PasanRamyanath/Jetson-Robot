// §5.5.5 / §5.5.6 NvVideoEncoder wrapper (MMAPI 01_video_encode / 10_camera_recording pattern, L4T 32.7).
#include "encoder.hpp"

#include <NvVideoEncoder.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace vc {

Encoder::Encoder() = default;
Encoder::~Encoder() { close(); }

bool Encoder::open(const EncCfg& c, AuFn au, MvFn mv, RelFn rel) {
  c_ = c, au_ = std::move(au), mv_ = std::move(mv), rel_ = std::move(rel);
  enc_ = NvVideoEncoder::createVideoEncoder(c.name);
  if (!enc_) return std::fprintf(stderr, "[%s] createVideoEncoder failed\n", c.name), false;
  bool ok = enc_->setCapturePlaneFormat(c.hevc ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_H264, uint32_t(c.w),
                                        uint32_t(c.h), uint32_t(c.w * c.h * 3 / 2)) == 0 &&
            enc_->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, uint32_t(c.w), uint32_t(c.h)) == 0 &&
            enc_->setBitrate(uint32_t(c.bps)) == 0;
  if (!ok) return std::fprintf(stderr, "[%s] format setup failed\n", c.name), false;
  if (c.hevc) {
    enc_->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
  } else {  // constrained baseline: every WebRTC browser decodes it (teleop via MediaMTX)
    enc_->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE);
    enc_->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_1);
    // §4 item 17: POC type 2, so the decoder never holds a frame back for reordering
    if (enc_->setPocType(2) != 0) std::fprintf(stderr, "[%s] no poc-type 2\n", c.name);
  }
  enc_->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR);
  enc_->setIDRInterval(uint32_t(c.idr));
  enc_->setIFrameInterval(uint32_t(c.idr));
  enc_->setFrameRate(uint32_t(c.fps), 1);
  enc_->setInsertSpsPpsAtIdrEnabled(true);
  enc_->setHWPresetType(V4L2_ENC_HW_PRESET_ULTRAFAST);
  enc_->setMaxPerfMode(1);  // clocks NVENC up front instead of ramping per frame
  if (c.roi_dqp) {
    v4l2_enc_enable_roi_param r;
    std::memset(&r, 0, sizeof r);
    r.bEnableROI = 1;
    if (enc_->enableROI(r) != 0) std::fprintf(stderr, "[%s] no ROI encoding\n", c.name), c_.roi_dqp = 0;
  }
  if (c.mv && enc_->enableMotionVectorReporting() != 0) std::fprintf(stderr, "[%s] no MV reporting\n", c.name);
  if (enc_->output_plane.setupPlane(V4L2_MEMORY_DMABUF, uint32_t(c.nbuf), false, false) < 0 ||
      enc_->capture_plane.setupPlane(V4L2_MEMORY_MMAP, uint32_t(c.nbuf), true, false) < 0 ||
      enc_->output_plane.setStreamStatus(true) < 0 || enc_->capture_plane.setStreamStatus(true) < 0)
    return std::fprintf(stderr, "[%s] plane setup failed\n", c.name), false;
  enc_->capture_plane.setDQThreadCallback(&Encoder::on_capture);
  enc_->capture_plane.startDQThread(this);
  for (uint32_t i = 0; i < enc_->capture_plane.getNumBuffers(); ++i) {
    v4l2_buffer b;
    v4l2_plane planes[MAX_PLANES];
    std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
    b.index = i, b.m.planes = planes;
    if (enc_->capture_plane.qBuffer(b, nullptr) < 0) return false;
  }
  int n = int(enc_->output_plane.getNumBuffers());
  fd_of_.assign(size_t(n), -1);
  free_.clear();
  for (int i = n - 1; i >= 0; --i) free_.push_back(i);
  eos_ = false;
  return true;
}

bool Encoder::push(int fd, uint64_t ts_us) {
  std::lock_guard<std::mutex> g(mu_);
  if (!enc_ || eos_) return false;
  v4l2_buffer b;
  v4l2_plane planes[MAX_PLANES];
  std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
  b.m.planes = planes;
  if (!free_.empty()) {
    b.index = uint32_t(free_.back());
    free_.pop_back();
  } else {  // oldest in-flight buffer: NVENC finished it long ago at nbuf frames of depth
    NvBuffer* nb = nullptr;
    if (enc_->output_plane.dqBuffer(b, &nb, nullptr, 10) < 0) return false;
    if (rel_ && fd_of_[b.index] >= 0) rel_(fd_of_[b.index]);
  }
  fd_of_[b.index] = fd;
  planes[0].m.fd = fd;
  planes[0].bytesused = 1;  // must be non-zero for DMABUF
  b.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
  b.timestamp.tv_sec = long(ts_us / 1000000), b.timestamp.tv_usec = long(ts_us % 1000000);
  if (c_.roi_dqp) {  // per buffer (config_store = index): an empty list clears the previous frame's regions
    v4l2_enc_frame_ROI_params r;
    std::memset(&r, 0, sizeof r);
    r.num_ROI_regions = uint32_t(roi_.size());
    for (size_t i = 0; i < roi_.size(); ++i) {
      v4l2_enc_ROI_param& q = r.ROI_params[i];
      q.ROIRect.left = roi_[i].l, q.ROIRect.top = roi_[i].t;
      q.ROIRect.width = uint32_t(roi_[i].w), q.ROIRect.height = uint32_t(roi_[i].h);
      q.QPdelta = c_.roi_dqp;
    }
    enc_->setROIParams(b.index, r);
  }
  if (enc_->output_plane.qBuffer(b, nullptr) < 0) {
    fd_of_[b.index] = -1;
    free_.push_back(int(b.index));
    return false;
  }
  return true;
}

bool Encoder::on_capture(v4l2_buffer* b, NvBuffer* buf, NvBuffer*, void* arg) {
  auto* e = static_cast<Encoder*>(arg);
  if (!b || !buf) return false;
  if (buf->planes[0].bytesused == 0) return false;  // EOS
  uint64_t us = uint64_t(b->timestamp.tv_sec) * 1000000ull + uint64_t(b->timestamp.tv_usec);
  if (e->mv_ && e->c_.mv) {
    v4l2_ctrl_videoenc_outputbuf_metadata_MV m;
    std::memset(&m, 0, sizeof m);
    if (e->enc_->getMotionVectors(b->index, m) == 0 && m.pMVInfo) {
      size_t n = m.bufSize / sizeof(MVInfo);
      e->mag_.resize(n);
      for (size_t i = 0; i < n; ++i) {
        float x = float(m.pMVInfo[i].mv_x) * e->mv_scale, y = float(m.pMVInfo[i].mv_y) * e->mv_scale;
        e->mag_[i] = std::sqrt(x * x + y * y);
      }
      e->mv_(e->mag_.data(), int(n), us);
    }
  }
  if (e->au_) e->au_(buf->planes[0].data, buf->planes[0].bytesused, us, (b->flags & V4L2_BUF_FLAG_KEYFRAME) != 0);
  return e->enc_->capture_plane.qBuffer(*b, nullptr) >= 0;
}

void Encoder::set_bitrate(int bps) {
  std::lock_guard<std::mutex> g(mu_);
  if (enc_) enc_->setBitrate(uint32_t(bps));
}

void Encoder::force_idr() {
  std::lock_guard<std::mutex> g(mu_);
  if (enc_) enc_->forceIDR();
}

void Encoder::set_roi(std::vector<Rect> r) {
  std::sort(r.begin(), r.end(), [](const Rect& a, const Rect& b) { return a.w * a.h > b.w * b.h; });
  if (r.size() > 8) r.resize(8);  // V4L2_MAX_ROI_REGIONS
  for (Rect& x : r) {             // grow to the 16 px macroblock grid, inside the frame
    int l = std::max(0, x.l & ~15), t = std::max(0, x.t & ~15);
    x.w = std::min(c_.w, (x.l + x.w + 15) & ~15) - l, x.h = std::min(c_.h, (x.t + x.h + 15) & ~15) - t;
    x.l = l, x.t = t;
  }
  r.erase(std::remove_if(r.begin(), r.end(), [](const Rect& x) { return x.w <= 0 || x.h <= 0; }), r.end());
  std::lock_guard<std::mutex> g(mu_);
  roi_.swap(r);
}

void Encoder::close() {
  std::unique_lock<std::mutex> g(mu_);
  if (!enc_) return;
  eos_ = true;
  v4l2_buffer b;
  v4l2_plane planes[MAX_PLANES];
  std::memset(&b, 0, sizeof b), std::memset(planes, 0, sizeof planes);
  b.m.planes = planes;
  bool have = false;
  if (!free_.empty()) {
    b.index = uint32_t(free_.back()), free_.pop_back(), have = true;
  } else {
    NvBuffer* nb = nullptr;
    if (enc_->output_plane.dqBuffer(b, &nb, nullptr, 10) >= 0) {
      if (rel_ && fd_of_[b.index] >= 0) rel_(fd_of_[b.index]);
      fd_of_[b.index] = -1, have = true;
    }
  }
  if (have) {  // EOS: an empty buffer; the capture DQ thread exits on the matching empty bitstream buffer
    planes[0].m.fd = -1, planes[0].bytesused = 0;
    enc_->output_plane.qBuffer(b, nullptr);
    enc_->capture_plane.waitForDQThread(2000);
  }
  enc_->capture_plane.stopDQThread();
  delete enc_;  // streams off: every queued dmabuf is returned
  enc_ = nullptr;
  if (rel_)
    for (int& fd : fd_of_)
      if (fd >= 0) rel_(fd), fd = -1;
}

}  // namespace vc
