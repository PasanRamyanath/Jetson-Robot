// §5.5.5 / §5.5.6 step 5: NvVideoEncoder (V4L2 NVENC) fed with dmabuf fds: rec0/rec1 H.265, teleop H.264,
// and a 320x180 H.264 encoder whose bitstream is dropped and only its motion vectors are read.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

class NvVideoEncoder;
struct v4l2_buffer;
class NvBuffer;

namespace vc {

struct EncCfg {
  const char* name = "enc";
  bool hevc = true;
  int w = 1280, h = 720, fps = 30, bps = 3000000, idr = 60;
  bool mv = false;  // report motion vectors (bitstream is still produced; the sink may ignore it)
  int roi_dqp = 0;  // < 0: per-frame ROI QP delta for the regions given to set_roi (teleop faces, §5.3)
  int nbuf = 6;
};

struct Rect {
  int l, t, w, h;
};

class Encoder {
 public:
  using AuFn = std::function<void(const uint8_t* au, size_t n, uint64_t ts_us, bool key)>;
  using MvFn = std::function<void(const float* mag, int n, uint64_t ts_us)>;  // |mv| per macroblock, pixels
  using RelFn = std::function<void(int fd)>;                                  // NVENC is done reading `fd`

  Encoder();
  ~Encoder();
  bool open(const EncCfg& c, AuFn au, MvFn mv, RelFn rel);
  // Queues a YUV420 dmabuf. False when every output buffer is still in flight: the caller keeps ownership.
  bool push(int fd, uint64_t ts_us);
  void set_bitrate(int bps);
  void force_idr();
  void set_roi(std::vector<Rect> r);  // applied from the next pushed frame; at most 8, largest first
  void close();
  float mv_scale = 0.25f;  // MVInfo units -> pixels (quarter-pel on Tegra NVENC; verify once with a pan)

 private:
  static bool on_capture(v4l2_buffer* b, NvBuffer* buf, NvBuffer* shared, void* arg);
  EncCfg c_;
  NvVideoEncoder* enc_ = nullptr;
  AuFn au_;
  MvFn mv_;
  RelFn rel_;
  std::mutex mu_;
  std::vector<int> free_, fd_of_;
  std::vector<float> mag_;
  std::vector<Rect> roi_;
  bool eos_ = false;
};

}  // namespace vc
