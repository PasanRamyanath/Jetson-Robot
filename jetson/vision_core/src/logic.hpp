// §5.5 portable decision logic (host-tested): detector cadence, NVENC-MV motion gate, night hysteresis, SCRFD decode,
// head crops, 5-point similarity (Umeyama) for the CUDA face warp, and per-track face-embedding pacing.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <iterator>
#include <utility>
#include <vector>

#include "geom.hpp"

namespace vc {

// ---------------------------------------------------------------- detector cadence (§15.1 modes + §5.5.5 wake-up)
// `interval` keeps the DeepStream meaning (frames skipped at 60 fps: 3 -> 15 Hz, 59 -> 1 Hz, <0 -> off). A motion
// event forces 15 Hz for `burst_s`, so idle mode reacts within one detector period (~70 ms).
class Cadence {
 public:
  void set_interval(int n) { interval_ = n; }
  int interval() const { return interval_; }
  void motion(double t) { burst_until_ = t + burst_s; }
  bool due(uint64_t frame, double t) const {
    int n = t < burst_until_ ? std::min(interval_ < 0 ? 3 : interval_, 3) : interval_;
    return n >= 0 && frame % uint64_t(n + 1) == 0;
  }
  double burst_s = 5.0;

 private:
  int interval_ = 3;
  double burst_until_ = -1;
};

// ---------------------------------------------------------------- §5.5.5 motion from encoder MVs
// One call per MV frame with |mv| per macroblock (pixels). Fires once when >= min_blocks blocks moved more than
// px_thresh for `frames` consecutive frames, then stays quiet for `refractory_s`.
class MotionGate {
 public:
  float px_thresh = 2.f;
  int min_blocks = 6, frames = 3;
  double refractory_s = 2.0;
  int last_score = 0;
  bool feed(const float* mag, int n, double t) {
    int s = 0;
    for (int i = 0; i < n; ++i) s += mag[i] > px_thresh;
    last_score = s;
    run_ = s >= min_blocks ? run_ + 1 : 0;
    if (run_ >= frames && t - last_fire_ >= refractory_s) {
      last_fire_ = t;
      return true;
    }
    return false;
  }

 private:
  int run_ = 0;
  double last_fire_ = -1e9;
};

// ---------------------------------------------------------------- §5.5.1 night logic
// lux_proxy = 1e9 / (analog gain * ISP digital gain * exposure ns). Night after `hold_s` below `on`, day after
// `hold_s` above `off` (off > on: hysteresis). Returns +1 on entering night, -1 on leaving, 0 otherwise.
class NightLogic {
 public:
  float on = 30.f, off = 90.f;
  double hold_s = 5.0;
  bool night = false;
  float ema = -1;
  int feed(float lux, double t) {
    ema = ema < 0 ? lux : 0.9f * ema + 0.1f * lux;  // ~0.25 s at 60 fps: ignores single-frame AE hunting
    bool want = night ? !(ema > off) : ema < on;
    if (want == night) {
      since_ = -1;
      return 0;
    }
    if (since_ < 0) since_ = t;
    if (t - since_ < hold_s) return 0;
    night = want, since_ = -1;
    return night ? 1 : -1;
  }
  static float lux_proxy(float analog_gain, float isp_gain, uint64_t exp_ns) {
    double d = double(analog_gain) * isp_gain * double(exp_ns);
    return d > 0 ? float(1e9 / d) : 0.f;
  }

 private:
  double since_ = -1;
};

// ---------------------------------------------------------------- faces
struct Face {
  Box b;           // source-frame pixels
  float kps[10];   // 5 landmarks x,y (source-frame pixels): eyes, nose, mouth corners
  float score = 0;
};

// SCRFD (insightface, 2 anchors, strides 8/16/32) on an `in`x`in` crop. Per stride: scores [N], bbox [N][4] and
// kps [N][10] as distances in stride units, N = (in/stride)^2 * 2. `lb` maps crop pixels back to the frame.
inline void decode_scrfd(const float* const scores[3], const float* const bbox[3], const float* const kps[3], int in,
                         float thr, const Letterbox& lb, std::vector<Face>& out) {
  static const int strides[3] = {8, 16, 32};
  for (int k = 0; k < 3; ++k) {
    int s = strides[k], g = in / s, n = g * g * 2;
    for (int i = 0; i < n; ++i) {
      if (scores[k][i] < thr) continue;
      int cell = i / 2;
      float cx = float((cell % g) * s), cy = float((cell / g) * s);
      const float* d = bbox[k] + 4 * i;
      Face f;
      f.score = scores[k][i];
      f.b = lb.undo(Box{cx - d[0] * s, cy - d[1] * s, cx + d[2] * s, cy + d[3] * s});
      const float* p = kps[k] + 10 * i;
      for (int j = 0; j < 5; ++j) {
        f.kps[2 * j] = lb.ux(cx + p[2 * j] * s);
        f.kps[2 * j + 1] = lb.uy(cy + p[2 * j + 1] * s);
      }
      out.push_back(f);
    }
  }
  std::vector<int> keep = nms(out, 0.4f, [](const Face& f) { return f.b; }, [](const Face& f) { return f.score; });
  std::vector<Face> kept;
  for (int i : keep) kept.push_back(out[i]);
  out.swap(kept);
}

// Square head crop from a person box (both in the same pixel space): top-anchored, side = the box width capped by
// half its height, grown 20 % so SCRFD sees the whole head.
inline Box head_crop(const Box& person, float W, float H) {
  float side = std::min(person.w(), 0.5f * person.h()) * 1.2f;
  side = std::max(side, 24.f);
  Box b{person.cx() - 0.5f * side, person.y1 - 0.1f * side, person.cx() + 0.5f * side, person.y1 + 0.9f * side};
  return clip(b, W, H);
}

// ArcFace 112x112 template (§5.5.4).
constexpr float kArcface[10] = {38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f,
                                71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f};

// Least-squares similarity mapping template (112 px) -> source landmarks: the 2x3 matrix warp_faces samples with.
inline void similarity(const float* kps, float M[6], const float* tmpl = kArcface) {
  float msx = 0, msy = 0, mdx = 0, mdy = 0;
  for (int i = 0; i < 5; ++i) msx += tmpl[2 * i], msy += tmpl[2 * i + 1], mdx += kps[2 * i], mdy += kps[2 * i + 1];
  msx /= 5, msy /= 5, mdx /= 5, mdy /= 5;
  float a = 0, b = 0, var = 0;
  for (int i = 0; i < 5; ++i) {
    float sx = tmpl[2 * i] - msx, sy = tmpl[2 * i + 1] - msy, dx = kps[2 * i] - mdx, dy = kps[2 * i + 1] - mdy;
    a += sx * dx + sy * dy, b += sx * dy - sy * dx, var += sx * sx + sy * sy;
  }
  float c = a / var, s = b / var;
  M[0] = c, M[1] = -s, M[2] = mdx - (c * msx - s * msy);
  M[3] = s, M[4] = c, M[5] = mdy - (s * msx + c * msy);
}

// Which person tracks get a face pass this tick: new tracks at 5 Hz until `settle` embeddings, then 1 Hz; biggest
// boxes first, at most `max_n` (the SCRFD engine's batch).
class FacePacer {
 public:
  double fast_s = 0.2, slow_s = 1.0, ttl_s = 3.0;
  int settle = 15;
  template <typename Tracks>  // iterable of (key, box) pairs
  std::vector<int> pick(const Tracks& cands, double t, int max_n) {
    std::vector<std::pair<float, int>> due;
    for (size_t i = 0; i < cands.size(); ++i) {
      const auto& st = s_[cands[i].first];
      if (t - st.last >= (st.n < settle ? fast_s : slow_s)) due.emplace_back(-cands[i].second.area(), int(i));
    }
    std::sort(due.begin(), due.end());
    std::vector<int> out;
    for (size_t i = 0; i < due.size() && int(i) < max_n; ++i) out.push_back(due[i].second);
    return out;
  }
  void done(int64_t key, double t) {
    auto& st = s_[key];
    st.last = t, ++st.n;
  }
  void gc(double t) {
    for (auto it = s_.begin(); it != s_.end();) it = t - it->second.last > ttl_s ? s_.erase(it) : std::next(it);
  }

 private:
  struct St {
    double last = -1e9;
    int n = 0;
  };
  std::map<int64_t, St> s_;
};

inline void l2norm(float* v, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) s += double(v[i]) * v[i];
  float k = s > 0 ? float(1.0 / std::sqrt(s)) : 0.f;
  for (int i = 0; i < n; ++i) v[i] *= k;
}

// IEEE half from float (round to nearest even), for the fp16 embedding blobs the agent stores (§11.8).
inline uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
  int exp = int((x >> 23) & 0xff) - 127 + 15;
  if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
  if (exp >= 31) return uint16_t(sign | 0x7c00);
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000;
    int shift = 14 - exp;
    uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return uint16_t(sign | h);
  }
  uint32_t h = sign | (uint32_t(exp) << 10) | (mant >> 13), rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
  return uint16_t(h);
}

// ---------------------------------------------------------------- §7.6 step 4: ReID re-acquisition (follow-me)
// Live person tracks keep an EMA of their OSNet embedding. Once ByteTrack forgets a track (after its own 1.5 s lost
// buffer, so an id is never published twice), the embedding waits in a lost gallery for `keep_s`. A new track that
// matches one (cosine >= thresh) during its first `probe_s` is published under the old id, so a locked follow target
// (and the agent's identity keyed on the tid) survives occlusion.
class ReidBank {
 public:
  float thresh = 0.7f;
  double keep_s = 30.0, refresh_s = 1.0, probe_s = 3.0, probe_every_s = 0.25;
  size_t max_lost = 16;
  // Live tracks due an embedding, unembedded/new ones first: at most max_n.
  std::vector<int> due(const std::vector<int>& live, double t, int max_n) const {
    std::vector<std::pair<double, int>> d;
    for (int id : live) {
      auto it = live_.find(id);
      if (it == live_.end()) {
        d.emplace_back(-2e18, id);
      } else {
        bool probing = t - it->second.first < probe_s;
        if (t - it->second.at >= (probing ? probe_every_s : refresh_s))
          d.emplace_back((probing ? -1e18 : 0) + it->second.at, id);
      }
    }
    std::sort(d.begin(), d.end());
    std::vector<int> out;
    for (size_t i = 0; i < d.size() && int(i) < max_n; ++i) out.push_back(d[i].second);
    return out;
  }
  // `e`: L2-normalised embedding of live track `id`. Returns true when it re-acquired a lost identity.
  bool add(int id, const float* e, int dim, double t) {
    St& st = live_[id];
    if (st.first < 0) st.first = t;
    bool hit = false;
    if (t - st.first < probe_s && !alias_.count(id)) {
      int best = -1;
      float bs = thresh;
      for (size_t i = 0; i < lost_.size(); ++i) {
        float c = cos(lost_[i].emb.data(), e, dim);
        if (c >= bs) bs = c, best = int(i);
      }
      if (best >= 0) {
        alias_[id] = lost_[size_t(best)].pub;
        st.emb = lost_[size_t(best)].emb;
        lost_.erase(lost_.begin() + best);
        hit = true;
      }
    }
    if (st.emb.size() != size_t(dim)) {
      st.emb.assign(e, e + dim);
    } else {
      for (int i = 0; i < dim; ++i) st.emb[size_t(i)] = 0.8f * st.emb[size_t(i)] + 0.2f * e[i];
      l2(st.emb);
    }
    st.at = t;
    return hit;
  }
  // Once per detector tick with every id ByteTrack still holds (any state): forgotten ones move to the lost gallery.
  void sync(const std::vector<int>& live, double t) {
    for (auto it = live_.begin(); it != live_.end();) {
      if (std::find(live.begin(), live.end(), it->first) != live.end()) {
        ++it;
        continue;
      }
      if (!it->second.emb.empty()) lost_.push_back(Lost{alias(it->first), std::move(it->second.emb), t});
      alias_.erase(it->first);
      it = live_.erase(it);
    }
    for (auto it = lost_.begin(); it != lost_.end();) it = t - it->t > keep_s ? lost_.erase(it) : std::next(it);
    if (lost_.size() > max_lost) lost_.erase(lost_.begin(), lost_.begin() + long(lost_.size() - max_lost));
  }
  int alias(int id) const {
    auto it = alias_.find(id);
    return it == alias_.end() ? id : it->second;
  }
  size_t lost() const { return lost_.size(); }

 private:
  struct St {
    std::vector<float> emb;
    double first = -1, at = -1e9;
  };
  struct Lost {
    int pub;
    std::vector<float> emb;
    double t;
  };
  static float cos(const float* a, const float* b, int n) {
    float s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
  }
  static void l2(std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += double(x) * x;
    float k = s > 0 ? float(1.0 / std::sqrt(s)) : 0.f;
    for (float& x : v) x *= k;
  }
  std::map<int, St> live_;
  std::map<int, int> alias_;
  std::vector<Lost> lost_;
};

}  // namespace vc
