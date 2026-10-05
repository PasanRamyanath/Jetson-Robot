// §5.5.6 step 5: minimal single-program MPEG-TS muxer for NVENC access units (H.264 / H.265, no B-frames).
// Recording segments (.ts: seekable, survives power loss mid-file) and the teleop feed to MediaMTX (udp:5000,
// same as the DeepStream path) both use it, so vision_core needs no GStreamer/ffmpeg. Portable, host-tested.
#pragma once
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <functional>
#include <string>

namespace vc {

inline uint32_t crc32_mpeg(const uint8_t* p, size_t n) {
  uint32_t c = 0xffffffffu;
  for (size_t i = 0; i < n; ++i) {
    c ^= uint32_t(p[i]) << 24;
    for (int k = 0; k < 8; ++k) c = c & 0x80000000u ? (c << 1) ^ 0x04c11db7u : c << 1;
  }
  return c;
}

// True if the Annex-B access unit holds an IDR/IRAP picture (used when the V4L2 keyframe flag is unavailable).
inline bool is_keyframe(const uint8_t* p, size_t n, bool hevc) {
  for (size_t i = 0; i + 3 < n; ++i) {
    if (p[i] || p[i + 1] || p[i + 2] != 1) continue;
    uint8_t h = p[i + 3];
    int t = hevc ? (h >> 1) & 0x3f : h & 0x1f;
    if (hevc ? (t >= 16 && t <= 21) : t == 5) return true;
  }
  return false;
}

class TsMux {
 public:
  using Sink = std::function<void(const uint8_t*, size_t)>;
  static constexpr uint16_t kPmtPid = 0x1000, kVidPid = 0x100;

  TsMux(bool hevc, Sink sink) : hevc_(hevc), sink_(std::move(sink)) {}
  // pts90k: 90 kHz presentation time. Tables are repeated before every keyframe so any segment/joiner can start there.
  void write(const uint8_t* au, size_t n, uint64_t pts90k, bool key) {
    if (key || !tables_sent_) tables();
    uint8_t pes[19];
    size_t h = 0;
    pes[h++] = 0, pes[h++] = 0, pes[h++] = 1, pes[h++] = 0xe0;
    size_t plen = n + 8;
    pes[h++] = plen > 0xffff ? 0 : uint8_t(plen >> 8), pes[h++] = plen > 0xffff ? 0 : uint8_t(plen);
    pes[h++] = 0x80, pes[h++] = 0x80, pes[h++] = 5;  // PTS only
    pts90k &= 0x1ffffffffULL;
    put_ts(pes + h, 0x2, pts90k), h += 5;
    bool first = true;
    size_t off = 0, total = h + n;
    while (off < total) {
      uint8_t pkt[188];
      pkt[0] = 0x47, pkt[1] = uint8_t((first ? 0x40 : 0) | (kVidPid >> 8)), pkt[2] = uint8_t(kVidPid & 0xff);
      size_t p = 4, room = 184;
      uint8_t af[184];
      size_t aflen = 0;
      if (first) {  // adaptation field: PCR (+ random access on keyframes)
        af[0] = uint8_t((key ? 0x40 : 0) | 0x10);
        uint64_t pcr = pts90k >= 2700 ? pts90k - 2700 : 0;  // 30 ms decode head-room
        af[1] = uint8_t(pcr >> 25), af[2] = uint8_t(pcr >> 17), af[3] = uint8_t(pcr >> 9), af[4] = uint8_t(pcr >> 1);
        af[5] = uint8_t(((pcr & 1) << 7) | 0x7e), af[6] = 0;
        aflen = 7;
      }
      size_t left = total - off, afb = aflen ? aflen + 1 : 0;  // afb: adaptation bytes incl. its length byte
      size_t payload = std::min(left, room - afb), stuff = room - afb - payload;
      if (afb) {
        pkt[3] = uint8_t(0x30 | cc_), pkt[p++] = uint8_t(aflen + stuff);
        std::memcpy(pkt + p, af, aflen), p += aflen;
      } else if (stuff) {  // last packet of the PES: an adaptation field made only of stuffing
        pkt[3] = uint8_t(0x30 | cc_), pkt[p++] = uint8_t(stuff - 1);
        if (stuff > 1) pkt[p++] = 0, stuff -= 2;
        else stuff = 0;
      } else {
        pkt[3] = uint8_t(0x10 | cc_);
      }
      std::memset(pkt + p, 0xff, stuff), p += stuff;
      cc_ = (cc_ + 1) & 0xf;
      size_t from_hdr = off < h ? std::min(h - off, payload) : 0;
      std::memcpy(pkt + p, pes + off, from_hdr);
      std::memcpy(pkt + p + from_hdr, au + (off + from_hdr - h), payload - from_hdr);
      off += payload;
      emit(pkt);
      first = false;
    }
    flush();
  }
  bool hevc() const { return hevc_; }

 private:
  bool hevc_;
  Sink sink_;
  uint8_t cc_ = 0, cc_pat_ = 0, cc_pmt_ = 0;
  bool tables_sent_ = false;
  uint8_t out_[188 * 7];
  size_t fill_ = 0;

  static void put_ts(uint8_t* b, uint8_t tag, uint64_t v) {
    b[0] = uint8_t((tag << 4) | ((v >> 29) & 0x0e) | 1);
    b[1] = uint8_t(v >> 22), b[2] = uint8_t(((v >> 14) & 0xfe) | 1);
    b[3] = uint8_t(v >> 7), b[4] = uint8_t(((v << 1) & 0xfe) | 1);
  }
  void emit(const uint8_t* pkt) {  // 7 packets per chunk = 1316 B: one UDP datagram, MediaMTX's expectation
    std::memcpy(out_ + fill_, pkt, 188), fill_ += 188;
    if (fill_ == sizeof(out_)) flush();
  }
  void flush() {
    if (fill_) sink_(out_, fill_), fill_ = 0;
  }
  void section(uint16_t pid, uint8_t& cc, const uint8_t* sec, size_t n) {
    uint8_t pkt[188];
    std::memset(pkt, 0xff, 188);
    pkt[0] = 0x47, pkt[1] = uint8_t(0x40 | (pid >> 8)), pkt[2] = uint8_t(pid), pkt[3] = uint8_t(0x10 | cc);
    cc = (cc + 1) & 0xf;
    pkt[4] = 0;  // pointer_field
    std::memcpy(pkt + 5, sec, n);
    emit(pkt);
  }
  void tables() {
    uint8_t pat[16] = {0x00, 0xb0, 13, 0x00, 0x01, 0xc1, 0, 0, 0x00, 0x01, uint8_t(0xe0 | (kPmtPid >> 8)),
                       uint8_t(kPmtPid & 0xff)};
    uint32_t c = crc32_mpeg(pat, 12);
    pat[12] = uint8_t(c >> 24), pat[13] = uint8_t(c >> 16), pat[14] = uint8_t(c >> 8), pat[15] = uint8_t(c);
    section(0, cc_pat_, pat, 16);
    uint8_t pmt[21] = {0x02, 0xb0, 18, 0x00, 0x01, 0xc1, 0, 0, uint8_t(0xe0 | (kVidPid >> 8)), uint8_t(kVidPid & 0xff),
                       0xf0, 0, uint8_t(hevc_ ? 0x24 : 0x1b), uint8_t(0xe0 | (kVidPid >> 8)),
                       uint8_t(kVidPid & 0xff), 0xf0, 0};
    c = crc32_mpeg(pmt, 17);
    pmt[17] = uint8_t(c >> 24), pmt[18] = uint8_t(c >> 16), pmt[19] = uint8_t(c >> 8), pmt[20] = uint8_t(c);
    section(kPmtPid, cc_pmt_, pmt, 21);
    tables_sent_ = true;
  }
};

}  // namespace vc
