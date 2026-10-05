// §5.5.1 one Argus CaptureSession per sensor with two ISP output streams: full 1280x720 (recording, faces,
// snapshots, teleop, MV) and the detector-sized stream (the ISP scales it for free). Argus headers stay in the .cpp.
#pragma once
#include <cstdint>
#include <memory>

namespace vc {

struct CamCfg {
  int sensor = 0;
  bool noir = false;
  int w = 1280, h = 720, fps = 60;
  int dnn_w = 512, dnn_h = 288;
  uint64_t max_exposure_ns = 8000000;  // motion-blur cap (§4 item 3)
  float wb[4] = {1.6f, 1.f, 1.f, 1.9f};  // NoIR manual gains (R, Geven, Godd, B); tune once under 850 nm
};

struct CamFrame {
  uint64_t fn = 0;       // capture counter since open()
  uint64_t ts_ns = 0;    // sensor timestamp
  double t = 0;          // steady clock seconds at acquire
  float lux = 0;         // NightLogic::lux_proxy
};

class ArgusCam {
 public:
  ArgusCam();
  ~ArgusCam();
  bool open(const CamCfg& c);
  // Blocks for the next capture. The DNN stream is VIC-converted into `dnn_fd` (RGBA); the full frame goes into
  // `full_fd` (YUV420) when >= 0, otherwise it is released untouched.
  bool next(CamFrame& f, int dnn_fd, int full_fd);
  // Normalised face box for auto-exposure (§5.5.1); r <= l clears. Applied on the next capture.
  void set_ae_region(float l, float t, float r, float b);
  // §15.1 idle-watch: stop/restart the repeating capture (VI + ISP go quiet). Call from the thread that calls next().
  void set_paused(bool paused);
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace vc
