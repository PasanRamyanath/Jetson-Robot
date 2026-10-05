// §3.17 NVDEC row / §11.9 step 9: NvVideoDecoder (V4L2 NVDEC) for sleep replay. Synchronous and single-threaded:
// push() Annex-B access units in decode order; decoded frames come back from inside push()/finish() as the
// decoder's own capture dmabuf (block-linear NV12), valid only during the callback: VIC-blit it out.
#pragma once
#include <cstdint>
#include <functional>

class NvVideoDecoder;

namespace vc {

class Decoder {
 public:
  using FrameFn = std::function<void(int fd, int w, int h, uint64_t pts90k)>;
  ~Decoder() { close(); }
  bool open(bool hevc, FrameFn fn);
  bool push(const uint8_t* au, size_t n, uint64_t pts90k);  // false: decoder error or no input buffer within 2 s
  void finish();                                             // EOS, then drain the remaining frames
  void close();

 private:
  bool drain();
  bool setup_capture();
  NvVideoDecoder* dec_ = nullptr;
  FrameFn fn_;
  uint32_t next_in_ = 0;
  int w_ = 0, h_ = 0;
  bool cap_ = false, eos_ = false;
};

}  // namespace vc
