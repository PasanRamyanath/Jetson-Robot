// §7.2 IPC schemas for the C++ processes (vision_core, beni_face, ROS 2 bridge). Mirror of beni_common/schemas.py.
// Header-only msgpack writer/reader (no msgpack-c dependency): C++14 (host gcc-7 / nvcc 10.2) and C++17 (ROS 2).
// Every message is a map with v (schema version), t (unix seconds, float64) and src. Add a field -> bump kV.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace beni {

constexpr int kSchemaV = 1;

namespace ep {
constexpr const char* kVision = "ipc:///tmp/beni/vision.sock";           // PUB det / jpeg / motion / scene
constexpr const char* kVisionCtrl = "ipc:///tmp/beni/vision_ctrl.sock";  // REP snapshot / bitrate / interval / mode
constexpr const char* kRobotState = "ipc:///tmp/beni/robot_state.sock";  // PUB state (10 Hz)
constexpr const char* kRobotCmd = "ipc:///tmp/beni/robot_cmd.sock";      // REP goto / look_at / follow / stop / dock
constexpr const char* kFaceCtrl = "ipc:///tmp/beni/face_ctrl.sock";      // PULL expression / gaze / overlay
constexpr const char* kEvents = "ipc:///tmp/beni/events.sock";           // PUB (agent binds)
constexpr const char* kSched = "ipc:///tmp/beni/sched.sock";             // PUB admit/pause + power mode
}  // namespace ep

namespace topic {
constexpr const char* kDet = "det";
constexpr const char* kJpeg = "jpeg";
constexpr const char* kMotion = "motion";
constexpr const char* kScene = "scene";
constexpr const char* kState = "state";
constexpr const char* kReplay = "replay";  // §11.9 sleep replay results
constexpr const char* kPose = "pose";      // §4 item 39 keypoints, on demand
}  // namespace topic

inline double unix_now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- writer
class MsgWriter {
 public:
  std::string buf;
  explicit MsgWriter(size_t reserve = 512) { buf.reserve(reserve); }
  void clear() { buf.clear(); }

  MsgWriter& map(uint32_t n) { return hdr(n, 0x80, 15, 0xde, 0xdf); }
  MsgWriter& arr(uint32_t n) { return hdr(n, 0x90, 15, 0xdc, 0xdd); }
  MsgWriter& nil() { return b8(0xc0); }
  MsgWriter& b(bool v) { return b8(v ? 0xc3 : 0xc2); }
  MsgWriter& str(const char* s) { return str(s, std::strlen(s)); }
  MsgWriter& str(const std::string& s) { return str(s.data(), s.size()); }
  MsgWriter& str(const char* s, size_t n) {
    if (n < 32) b8(uint8_t(0xa0 | n));
    else if (n < 256) b8(0xd9).b8(uint8_t(n));
    else if (n < 65536) b8(0xda).be(uint16_t(n));
    else b8(0xdb).be(uint32_t(n));
    buf.append(s, n);
    return *this;
  }
  MsgWriter& bin(const void* p, size_t n) {
    if (n < 256) b8(0xc4).b8(uint8_t(n));
    else if (n < 65536) b8(0xc5).be(uint16_t(n));
    else b8(0xc6).be(uint32_t(n));
    buf.append(static_cast<const char*>(p), n);
    return *this;
  }
  MsgWriter& i(int64_t v) {
    if (v >= 0) {
      if (v < 128) return b8(uint8_t(v));
      if (v < 256) return b8(0xcc).b8(uint8_t(v));
      if (v < 65536) return b8(0xcd).be(uint16_t(v));
      if (v <= 0xffffffffLL) return b8(0xce).be(uint32_t(v));
      return b8(0xcf).be(uint64_t(v));
    }
    if (v >= -32) return b8(uint8_t(int8_t(v)));
    if (v >= -128) return b8(0xd0).b8(uint8_t(int8_t(v)));
    if (v >= -32768) return b8(0xd1).be(uint16_t(int16_t(v)));
    if (v >= -2147483648LL) return b8(0xd2).be(uint32_t(int32_t(v)));
    return b8(0xd3).be(uint64_t(v));
  }
  MsgWriter& f32(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return b8(0xca).be(u);
  }
  MsgWriter& f64(double v) {
    uint64_t u;
    std::memcpy(&u, &v, 8);
    return b8(0xcb).be(u);
  }
  // Rounded like Python's round(x, d) in beni_vision.py, so both vision paths emit identical payloads.
  MsgWriter& fr(double v, int decimals) {
    double m = decimals == 1 ? 10.0 : decimals == 2 ? 100.0 : 1000.0;
    return f64(double(int64_t(v * m + (v >= 0 ? 0.5 : -0.5))) / m);
  }
  MsgWriter& key(const char* k) { return str(k); }
  // Opens the top-level map: n_extra fields must follow.
  MsgWriter& envelope(const char* src, uint32_t n_extra) {
    map(3 + n_extra);
    key("v").i(kSchemaV);
    key("t").f64(unix_now());
    return key("src").str(src);
  }

 private:
  MsgWriter& b8(uint8_t c) {
    buf.push_back(char(c));
    return *this;
  }
  template <typename T>
  MsgWriter& be(T v) {
    char tmp[sizeof(T)];
    for (size_t k = 0; k < sizeof(T); ++k) tmp[k] = char((v >> (8 * (sizeof(T) - 1 - k))) & 0xff);
    buf.append(tmp, sizeof(T));
    return *this;
  }
  MsgWriter& hdr(uint32_t n, uint8_t fix, uint32_t fixmax, uint8_t c16, uint8_t c32) {
    if (n <= fixmax) return b8(uint8_t(fix | n));
    if (n < 65536) return b8(c16).be(uint16_t(n));
    return b8(c32).be(n);
  }
};

// ---------------------------------------------------------------- reader (small control messages)
struct Value {
  enum Type { NIL, BOOL, INT, FLOAT, STR, BIN, ARR, MAP } type = NIL;
  int64_t i = 0;
  double f = 0;
  std::string s;  // STR and BIN
  std::vector<Value> a;
  std::vector<std::pair<std::string, Value>> m;

  const Value* get(const char* k) const {
    for (const auto& kv : m)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
  double num(const char* k, double dflt = 0) const {
    const Value* v = get(k);
    return !v ? dflt : v->type == INT ? double(v->i) : v->type == FLOAT ? v->f : v->type == BOOL ? double(v->i) : dflt;
  }
  std::string text(const char* k, const std::string& dflt = "") const {
    const Value* v = get(k);
    return v && v->type == STR ? v->s : dflt;
  }
  bool has(const char* k) const { return get(k) != nullptr; }
};

class MsgReader {
 public:
  MsgReader(const void* p, size_t n) : p_(static_cast<const uint8_t*>(p)), end_(p_ + n) {}
  // Returns false on malformed or truncated input (depth-limited, so hostile input cannot blow the stack).
  bool parse(Value& out, int depth = 0) {
    if (depth > 16 || p_ >= end_) return false;
    uint8_t c = *p_++;
    if (c <= 0x7f) return set_int(out, c);
    if (c >= 0xe0) return set_int(out, int8_t(c));
    if ((c & 0xf0) == 0x80) return read_map(out, c & 0x0f, depth);
    if ((c & 0xf0) == 0x90) return read_arr(out, c & 0x0f, depth);
    if ((c & 0xe0) == 0xa0) return read_str(out, c & 0x1f, Value::STR);
    uint64_t n;
    switch (c) {
      case 0xc0: out.type = Value::NIL; return true;
      case 0xc2: case 0xc3: out.type = Value::BOOL; out.i = c == 0xc3; return true;
      case 0xc4: return be(n, 1) && read_str(out, n, Value::BIN);
      case 0xc5: return be(n, 2) && read_str(out, n, Value::BIN);
      case 0xc6: return be(n, 4) && read_str(out, n, Value::BIN);
      case 0xca: { if (!be(n, 4)) return false; uint32_t u = uint32_t(n); float f; std::memcpy(&f, &u, 4);
                   out.type = Value::FLOAT; out.f = f; return true; }
      case 0xcb: { if (!be(n, 8)) return false; double d; std::memcpy(&d, &n, 8);
                   out.type = Value::FLOAT; out.f = d; return true; }
      case 0xcc: return be(n, 1) && set_int(out, int64_t(n));
      case 0xcd: return be(n, 2) && set_int(out, int64_t(n));
      case 0xce: return be(n, 4) && set_int(out, int64_t(n));
      case 0xcf: return be(n, 8) && set_int(out, int64_t(n));
      case 0xd0: return be(n, 1) && set_int(out, int8_t(n));
      case 0xd1: return be(n, 2) && set_int(out, int16_t(n));
      case 0xd2: return be(n, 4) && set_int(out, int32_t(n));
      case 0xd3: return be(n, 8) && set_int(out, int64_t(n));
      case 0xd9: return be(n, 1) && read_str(out, n, Value::STR);
      case 0xda: return be(n, 2) && read_str(out, n, Value::STR);
      case 0xdb: return be(n, 4) && read_str(out, n, Value::STR);
      case 0xdc: return be(n, 2) && read_arr(out, n, depth);
      case 0xdd: return be(n, 4) && read_arr(out, n, depth);
      case 0xde: return be(n, 2) && read_map(out, n, depth);
      case 0xdf: return be(n, 4) && read_map(out, n, depth);
      default: return false;  // ext types are never sent
    }
  }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
  bool be(uint64_t& v, int n) {
    if (end_ - p_ < n) return false;
    v = 0;
    for (int k = 0; k < n; ++k) v = (v << 8) | *p_++;
    return true;
  }
  static bool set_int(Value& out, int64_t v) {
    out.type = Value::INT;
    out.i = v;
    out.f = double(v);
    return true;
  }
  bool read_str(Value& out, uint64_t n, Value::Type t) {
    if (uint64_t(end_ - p_) < n) return false;
    out.type = t;
    out.s.assign(reinterpret_cast<const char*>(p_), size_t(n));
    p_ += n;
    return true;
  }
  bool read_arr(Value& out, uint64_t n, int depth) {
    if (uint64_t(end_ - p_) < n) return false;
    out.type = Value::ARR;
    out.a.resize(size_t(n));
    for (auto& v : out.a)
      if (!parse(v, depth + 1)) return false;
    return true;
  }
  bool read_map(Value& out, uint64_t n, int depth) {
    if (uint64_t(end_ - p_) < 2 * n) return false;
    out.type = Value::MAP;
    out.m.resize(size_t(n));
    for (auto& kv : out.m) {
      Value k;
      if (!parse(k, depth + 1) || k.type != Value::STR || !parse(kv.second, depth + 1)) return false;
      kv.first = std::move(k.s);
    }
    return true;
  }
};

inline bool unpack(const void* p, size_t n, Value& out) { return MsgReader(p, n).parse(out); }

// Standard replies for REP sockets: {"ok": true} / {"ok": false, "err": "..."}.
inline std::string reply_ok() { MsgWriter w(8); w.map(1).key("ok").b(true); return w.buf; }
inline std::string reply_err(const std::string& e) {
  MsgWriter w(32 + e.size());
  w.map(2).key("ok").b(false).key("err").str(e);
  return w.buf;
}

}  // namespace beni
