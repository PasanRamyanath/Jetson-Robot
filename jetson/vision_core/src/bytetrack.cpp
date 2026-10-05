// §5.5.6 ByteTrack + Kalman (see bytetrack.hpp).
#include "bytetrack.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace vc {

namespace {
constexpr float kWPos = 1.f / 20, kWVel = 1.f / 160;

// In-place inverse of a 4x4 SPD matrix by Gauss-Jordan (innovation covariance: always well conditioned).
bool inv4(float a[4][4], float r[4][4]) {
  float m[4][8];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) m[i][j] = j < 4 ? a[i][j] : float(j - 4 == i);
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int i = c + 1; i < 4; ++i)
      if (std::fabs(m[i][c]) > std::fabs(m[piv][c])) piv = i;
    if (std::fabs(m[piv][c]) < 1e-12f) return false;
    if (piv != c)
      for (int j = 0; j < 8; ++j) std::swap(m[c][j], m[piv][j]);
    float d = 1.f / m[c][c];
    for (int j = 0; j < 8; ++j) m[c][j] *= d;
    for (int i = 0; i < 4; ++i) {
      if (i == c) continue;
      float f = m[i][c];
      for (int j = 0; j < 8; ++j) m[i][j] -= f * m[c][j];
    }
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r[i][j] = m[i][j + 4];
  return true;
}

void xyah(const Box& b, float z[4]) {
  z[0] = b.cx(), z[1] = b.cy(), z[3] = std::max(b.h(), 1.f), z[2] = b.w() / z[3];
}
}  // namespace

void Kalman::init(const Box& b) {
  xyah(b, x);
  x[4] = x[5] = x[6] = x[7] = 0;
  float h = x[3];
  float s[8] = {2 * kWPos * h, 2 * kWPos * h, 1e-2f, 2 * kWPos * h, 10 * kWVel * h, 10 * kWVel * h, 1e-5f,
                10 * kWVel * h};
  std::memset(P, 0, sizeof(P));
  for (int i = 0; i < 8; ++i) P[i][i] = s[i] * s[i];
}

void Kalman::predict(float dt) {
  if (dt <= 0) return;
  float h = x[3];
  float q[8] = {kWPos * h, kWPos * h, 1e-2f, kWPos * h, kWVel * h, kWVel * h, 1e-5f, kWVel * h};
  for (int i = 0; i < 4; ++i) x[i] += dt * x[i + 4];
  // P = F P F^T with F = [[I, dt I], [0, I]]: rows then columns.
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) P[i][j] += dt * P[i + 4][j];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) P[i][j] += dt * P[i][j + 4];
  for (int i = 0; i < 8; ++i) P[i][i] += dt * q[i] * q[i];  // process noise grows with elapsed ticks
  if (x[3] < 1) x[3] = 1;
}

void Kalman::update(const Box& b) {
  float z[4], h = x[3];
  xyah(b, z);
  float r[4] = {kWPos * h, kWPos * h, 1e-1f, kWPos * h};
  float S[4][4], Si[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) S[i][j] = P[i][j] + (i == j ? r[i] * r[i] : 0.f);
  if (!inv4(S, Si)) return;
  float K[8][4];  // K = P H^T S^-1, H^T selects P's first 4 columns
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) {
      float acc = 0;
      for (int k = 0; k < 4; ++k) acc += P[i][k] * Si[k][j];
      K[i][j] = acc;
    }
  float y[4];
  for (int i = 0; i < 4; ++i) y[i] = z[i] - x[i];
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 4; ++k) x[i] += K[i][k] * y[k];
  float HP[4][8];  // P -= K H P  (== K S K^T)
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) HP[i][j] = P[i][j];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      float acc = 0;
      for (int k = 0; k < 4; ++k) acc += K[i][k] * HP[k][j];
      P[i][j] -= acc;
    }
}

Box Kalman::box() const {
  float w = x[2] * x[3];
  return Box{x[0] - 0.5f * w, x[1] - 0.5f * x[3], x[0] + 0.5f * w, x[1] + 0.5f * x[3]};
}

std::vector<std::pair<int, int>> assign(const std::vector<float>& cost, int rows, int cols, float thresh) {
  std::vector<std::pair<int, int>> out;
  if (!rows || !cols) return out;
  bool tr = rows > cols;  // Hungarian below needs n <= m
  int n = tr ? cols : rows, m = tr ? rows : cols;
  const float big = thresh + 1.f;
  auto c = [&](int i, int j) {  // 1-based
    float v = tr ? cost[size_t(j - 1) * cols + (i - 1)] : cost[size_t(i - 1) * cols + (j - 1)];
    return v > thresh ? big : v;
  };
  const float inf = std::numeric_limits<float>::max();
  std::vector<float> u(n + 1), v(m + 1), minv(m + 1);
  std::vector<int> p(m + 1), way(m + 1);
  std::vector<char> used(m + 1);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::fill(minv.begin(), minv.end(), inf);
    std::fill(used.begin(), used.end(), 0);
    do {
      used[j0] = 1;
      int i0 = p[j0], j1 = 0;
      float delta = inf;
      for (int j = 1; j <= m; ++j)
        if (!used[j]) {
          float cur = c(i0, j) - u[i0] - v[j];
          if (cur < minv[j]) minv[j] = cur, way[j] = j0;
          if (minv[j] < delta) delta = minv[j], j1 = j;
        }
      for (int j = 0; j <= m; ++j)
        if (used[j]) u[p[j]] += delta, v[j] -= delta;
        else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  for (int j = 1; j <= m; ++j) {
    if (!p[j]) continue;
    int r = tr ? j - 1 : p[j] - 1, col = tr ? p[j] - 1 : j - 1;
    if (cost[size_t(r) * cols + col] <= thresh) out.emplace_back(r, col);
  }
  return out;
}

void ByteTracker::predict(double t) {
  if (t_ >= 0 && t > t_) {
    float dt = float((t - t_) * p_.tick_hz);
    for (auto& tr : tracks_) tr.kf.predict(dt);
  }
  if (t > t_) t_ = t;  // monotonic: the 60 Hz publisher may already be ahead of a detector tick's capture time
}

void ByteTracker::update(double t, const std::vector<Det>& dets) {
  if (t_ < 0) t_ = t;
  std::vector<int> high, low;
  for (int i = 0; i < int(dets.size()); ++i) {
    if (dets[i].score >= p_.track_thresh) high.push_back(i);
    else if (dets[i].score >= p_.low_score) low.push_back(i);
  }
  std::vector<int> pool, unconf;  // indices into tracks_
  for (int i = 0; i < int(tracks_.size()); ++i) (tracks_[i].activated ? pool : unconf).push_back(i);

  auto costs = [&](const std::vector<int>& trs, const std::vector<int>& ds, bool fuse) {
    std::vector<float> c(trs.size() * ds.size());
    for (size_t a = 0; a < trs.size(); ++a) {
      Box tb = tracks_[trs[a]].box();
      for (size_t b = 0; b < ds.size(); ++b) {
        const Det& d = dets[ds[b]];
        float s = iou(tb, d.b) * (fuse ? d.score : 1.f);
        c[a * ds.size() + b] = tracks_[trs[a]].cls == d.cls ? 1.f - s : 2.f;
      }
    }
    return c;
  };
  auto hit = [&](Track& tr, const Det& d) {
    tr.kf.update(d.b);
    tr.score = d.score, tr.t_seen = t, tr.state = Track::TRACKED, tr.activated = true, ++tr.hits;
  };
  std::vector<char> det_used(dets.size(), 0), tr_used(tracks_.size(), 0);

  // 1) high-score detections vs confirmed + lost tracks
  for (auto& pr : assign(costs(pool, high, true), int(pool.size()), int(high.size()), p_.match_thresh)) {
    hit(tracks_[pool[pr.first]], dets[high[pr.second]]);
    tr_used[pool[pr.first]] = det_used[high[pr.second]] = 1;
  }
  // 2) low-score detections vs still-unmatched *tracked* tracks (occlusion recovery: the Byte step)
  std::vector<int> rest, rest_high;
  for (int i : pool)
    if (!tr_used[i] && tracks_[i].state == Track::TRACKED) rest.push_back(i);
  for (auto& pr : assign(costs(rest, low, false), int(rest.size()), int(low.size()), p_.low_match_thresh)) {
    hit(tracks_[rest[pr.first]], dets[low[pr.second]]);
    tr_used[rest[pr.first]] = det_used[low[pr.second]] = 1;
  }
  for (int i : rest)
    if (!tr_used[i]) tracks_[i].state = Track::LOST;
  // 3) unconfirmed tracks vs the remaining high detections; unmatched unconfirmed die
  for (int i : high)
    if (!det_used[i]) rest_high.push_back(i);
  for (auto& pr : assign(costs(unconf, rest_high, true), int(unconf.size()), int(rest_high.size()),
                         p_.new_match_thresh)) {
    hit(tracks_[unconf[pr.first]], dets[rest_high[pr.second]]);
    tr_used[unconf[pr.first]] = det_used[rest_high[pr.second]] = 1;
  }
  std::vector<char> dead(tracks_.size(), 0);
  for (int i : unconf)
    if (!tr_used[i]) dead[i] = 1;
  for (size_t i = 0; i < tracks_.size(); ++i)
    if (tracks_[i].state == Track::LOST && t - tracks_[i].t_seen > p_.max_lost_s) dead[i] = 1;
  std::vector<Track> keep;
  keep.reserve(tracks_.size() + rest_high.size());
  for (size_t i = 0; i < tracks_.size(); ++i)
    if (!dead[i]) keep.push_back(tracks_[i]);
  // 4) new tracks from confident leftovers (confirmed immediately only on the very first tick)
  for (int i : rest_high) {
    if (det_used[i] || dets[i].score < p_.high_thresh) continue;
    Track tr;
    tr.id = next_id_++, tr.cls = dets[i].cls, tr.score = dets[i].score, tr.t_seen = t, tr.hits = 1;
    tr.state = Track::TRACKED, tr.activated = first_;
    tr.kf.init(dets[i].b);
    keep.push_back(tr);
  }
  tracks_.swap(keep);
  first_ = false;
}

}  // namespace vc
