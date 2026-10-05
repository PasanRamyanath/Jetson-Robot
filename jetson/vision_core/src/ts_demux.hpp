// §11.9 sleep replay: MPEG-TS -> Annex-B access units with their PTS, for NVDEC. Reads what TsMux writes (one
// program, one H.264/H.265 stream, one AU per PES) and tolerates a torn last packet from a power cut. PAT/PMT only
// locate the video PID. Portable, host-tested against TsMux.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

namespace vc {

class TsDemux {
 public:
  using AuFn = std::function<void(const uint8_t* au, size_t n, uint64_t pts90k)>;
  explicit TsDemux(AuFn fn) : fn_(std::move(fn)) {}

  // Any chunking; bytes past the last whole packet are kept for the next call.
  void feed(const uint8_t* p, size_t n) {
    while (n) {
      if (part_.empty() && n >= 188 && p[0] == 0x47) {
        packet(p), p += 188, n -= 188;
        continue;
      }
      if (part_.empty() && p[0] != 0x47) {  // resync
        ++p, --n;
        continue;
      }
      size_t k = std::min(n, size_t(188) - part_.size());
      part_.insert(part_.end(), p, p + k), p += k, n -= k;
      if (part_.size() == 188) packet(part_.data()), part_.clear();
    }
  }
  void flush() {  // the final AU has no following PES start to close it
    if (!au_.empty() && have_pts_) fn_(au_.data(), au_.size(), pts_);
    au_.clear(), have_pts_ = false;
  }
  bool hevc() const { return stype_ == 0x24; }
  int video_pid() const { return vpid_; }

 private:
  AuFn fn_;
  std::vector<uint8_t> part_, au_;
  int pmt_pid_ = -1, vpid_ = -1, stype_ = 0;
  uint64_t pts_ = 0;
  bool have_pts_ = false;

  static const uint8_t* section(const uint8_t* pl, const uint8_t* end, size_t& len) {
    if (pl >= end) return nullptr;
    const uint8_t* s = pl + 1 + pl[0];  // pointer_field
    if (s + 3 > end) return nullptr;
    len = size_t(((s[1] & 0x0f) << 8) | s[2]) + 3;
    return s + len <= end ? s : nullptr;
  }

  void packet(const uint8_t* p) {
    int pid = ((p[1] & 0x1f) << 8) | p[2];
    bool pusi = p[1] & 0x40;
    int afc = (p[3] >> 4) & 3;
    if (!(afc & 1)) return;
    const uint8_t *pl = p + 4, *end = p + 188;
    if (afc & 2) pl += 1 + p[4];
    if (pl >= end) return;
    size_t len = 0;
    if (pid == 0 && pusi) {
      const uint8_t* s = section(pl, end, len);
      if (s && s[0] == 0x00)
        for (const uint8_t* e = s + 8; e + 4 <= s + len - 4; e += 4)
          if ((e[0] << 8 | e[1]) != 0) {
            pmt_pid_ = ((e[2] & 0x1f) << 8) | e[3];
            break;
          }
    } else if (pid == pmt_pid_ && pusi) {
      const uint8_t* s = section(pl, end, len);
      if (!s || s[0] != 0x02 || len < 16) return;
      const uint8_t* e = s + 12 + (((s[10] & 0x0f) << 8) | s[11]);
      for (; e + 5 <= s + len - 4; e += 5 + (((e[3] & 0x0f) << 8) | e[4]))
        if (e[0] == 0x1b || e[0] == 0x24) {
          stype_ = e[0], vpid_ = ((e[1] & 0x1f) << 8) | e[2];
          break;
        }
    } else if (pid == vpid_) {
      if (pusi) {
        flush();
        if (end - pl < 9 || pl[0] || pl[1] || pl[2] != 1) return;
        const uint8_t* h = pl;
        if ((h[7] & 0x80) && end - pl >= 14) {
          pts_ = (uint64_t(h[9] >> 1 & 7) << 30) | (uint64_t(h[10]) << 22) | (uint64_t(h[11] >> 1) << 15) |
                 (uint64_t(h[12]) << 7) | uint64_t(h[13] >> 1);
          have_pts_ = true;
        }
        pl += 9 + h[8];
        if (pl > end) return;
      } else if (!have_pts_) {
        return;  // joined mid-PES
      }
      au_.insert(au_.end(), pl, end);
    }
  }
};

}  // namespace vc
