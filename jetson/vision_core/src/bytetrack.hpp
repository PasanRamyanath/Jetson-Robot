// §5.5.6 step 3: ByteTrack (Zhang et al. 2022; port of the MIT reference) with a Kalman filter that predicts at the
// 60 fps camera rate and associates on detector ticks (15 Hz, or 1 Hz in idle). Time is in seconds; velocities are
// per 1/15 s so the filter behaves the same whatever the detector interval. Portable C++14, host-tested.
#pragma once
#include <vector>

#include "geom.hpp"

namespace vc {

struct Kalman {  // state x y a h vx vy va vh (a = w/h), DeepSORT parameterisation
  float x[8], P[8][8];
  void init(const Box& b);
  void predict(float dt);  // dt in detector ticks (1 = 1/15 s)
  void update(const Box& b);
  Box box() const;
};

struct Track {
  int id = 0, cls = 0;
  float score = 0;
  enum State { NEW, TRACKED, LOST } state = NEW;
  bool activated = false;
  double t_seen = 0;  // last associated detection
  int hits = 0;
  Kalman kf;
  Box box() const { return kf.box(); }
};

struct ByteTrackParams {
  float track_thresh = 0.5f, high_thresh = 0.6f, match_thresh = 0.8f, low_match_thresh = 0.5f;
  float new_match_thresh = 0.7f, low_score = 0.1f;
  double max_lost_s = 1.5;
  float tick_hz = 15.f;
};

class ByteTracker {
 public:
  explicit ByteTracker(ByteTrackParams p = ByteTrackParams(), int first_id = 1) : p_(p), next_id_(first_id) {}
  // Every camera frame: move all live tracks to time t.
  void predict(double t);
  // Detector tick (call predict(t) first): associate, spawn, retire.
  void update(double t, const std::vector<Det>& dets);
  // Confirmed, currently tracked tracks (what gets published).
  template <typename F>
  void each_active(F f) const {
    for (const auto& tr : tracks_)
      if (tr.state == Track::TRACKED && tr.activated) f(tr);
  }
  const std::vector<Track>& tracks() const { return tracks_; }
  void reset() { tracks_.clear(), t_ = -1; }

 private:
  ByteTrackParams p_;
  std::vector<Track> tracks_;
  int next_id_;
  double t_ = -1;
  bool first_ = true;
};

// Min-cost assignment with a gate: returns pairs (row, col) with cost <= thresh. Rectangular, O(n^2 m).
std::vector<std::pair<int, int>> assign(const std::vector<float>& cost, int rows, int cols, float thresh);

}  // namespace vc
