// §5.5 portable geometry: boxes, IoU, detector output decoding (YOLO26 end-to-end / EfficientNMS), NMS.
// Pure C++14, no CUDA: unit-tested on the host (tests/contract/cpp/vision_core_test.cpp).
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace vc {

struct Box {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float w() const { return std::max(0.f, x2 - x1); }
  float h() const { return std::max(0.f, y2 - y1); }
  float area() const { return w() * h(); }
  float cx() const { return 0.5f * (x1 + x2); }
  float cy() const { return 0.5f * (y1 + y2); }
};

struct Det {
  Box b;
  float score = 0;
  int cls = 0;
};

inline float iou(const Box& a, const Box& b) {
  float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1), ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0.f;
  float inter = iw * ih;
  return inter / (a.area() + b.area() - inter + 1e-6f);
}

inline Box clip(Box b, float W, float H) {
  b.x1 = std::min(std::max(b.x1, 0.f), W), b.x2 = std::min(std::max(b.x2, 0.f), W);
  b.y1 = std::min(std::max(b.y1, 0.f), H), b.y2 = std::min(std::max(b.y2, 0.f), H);
  return b;
}

// Network input <- source image mapping. The ISP's 512x288 stream is exactly 16:9, so scale = 1 and no padding;
// kept general for the 416x224 low-power engine and for square SCRFD crops.
struct Letterbox {
  float scale = 1, pad_x = 0, pad_y = 0;
  static Letterbox fit(float src_w, float src_h, float net_w, float net_h) {
    Letterbox l;
    l.scale = std::min(net_w / src_w, net_h / src_h);
    l.pad_x = 0.5f * (net_w - src_w * l.scale), l.pad_y = 0.5f * (net_h - src_h * l.scale);
    return l;
  }
  float ux(float x) const { return (x - pad_x) / scale; }
  float uy(float y) const { return (y - pad_y) / scale; }
  Box undo(const Box& b) const { return Box{ux(b.x1), uy(b.y1), ux(b.x2), uy(b.y2)}; }
};

// Class whitelist as a 128-bit mask (COCO has 80 classes).
struct ClassMask {
  uint64_t bits[2] = {~0ull, ~0ull};
  static ClassMask only(const std::vector<int>& ids) {
    ClassMask m;
    m.bits[0] = m.bits[1] = 0;
    for (int c : ids)
      if (c >= 0 && c < 128) m.bits[c >> 6] |= 1ull << (c & 63);
    return m;
  }
  bool has(int c) const { return c >= 0 && c < 128 && (bits[c >> 6] >> (c & 63)) & 1; }
};

// YOLO26 end-to-end head (§5.5.6 step 2): rows of x1 y1 x2 y2 score class, already one-to-one (no NMS).
inline void decode_e2e(const float* rows, int n, float thr, const ClassMask& allow, const Letterbox& lb, float W,
                       float H, std::vector<Det>& out) {
  for (int i = 0; i < n; ++i) {
    const float* r = rows + 6 * i;
    int cls = int(r[5] + 0.5f);
    if (r[4] < thr || !allow.has(cls)) continue;
    Det d;
    d.b = clip(lb.undo(Box{r[0], r[1], r[2], r[3]}), W, H), d.score = r[4], d.cls = cls;
    if (d.b.area() > 1.f) out.push_back(d);
  }
}

// YOLOv8n fallback: EfficientNMS_TRT outputs (num_dets, boxes[K][4] x1y1x2y2, scores[K], classes[K]).
inline void decode_nms(int num, const float* boxes, const float* scores, const float* classes, float thr,
                       const ClassMask& allow, const Letterbox& lb, float W, float H, std::vector<Det>& out) {
  for (int i = 0; i < num; ++i) {
    int cls = int(classes[i] + 0.5f);
    if (scores[i] < thr || !allow.has(cls)) continue;
    const float* b = boxes + 4 * i;
    Det d;
    d.b = clip(lb.undo(Box{b[0], b[1], b[2], b[3]}), W, H), d.score = scores[i], d.cls = cls;
    if (d.b.area() > 1.f) out.push_back(d);
  }
}

// Greedy NMS in place (SCRFD only; the person detector never needs it). Returns kept indices, best first.
template <typename T, typename BoxOf, typename ScoreOf>
std::vector<int> nms(const std::vector<T>& v, float thr, BoxOf box, ScoreOf score) {
  std::vector<int> idx(v.size()), keep;
  for (size_t i = 0; i < v.size(); ++i) idx[i] = int(i);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return score(v[a]) > score(v[b]); });
  std::vector<char> dead(v.size(), 0);
  for (size_t a = 0; a < idx.size(); ++a) {
    if (dead[idx[a]]) continue;
    keep.push_back(idx[a]);
    for (size_t b = a + 1; b < idx.size(); ++b)
      if (!dead[idx[b]] && iou(box(v[idx[a]]), box(v[idx[b]])) > thr) dead[idx[b]] = 1;
  }
  return keep;
}

}  // namespace vc
