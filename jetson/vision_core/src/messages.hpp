// §5.5.6 step 6: the vision.sock payloads, byte-compatible with beni_vision.py (§5.4.4) so the agent, the scheduler
// and the ROS bridge cannot tell the two vision paths apart. bbox = [left, top, w, h] in 512x288 detector pixels.
#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "schemas.hpp"
#include "geom.hpp"

namespace vc {

constexpr int kGiePerson = 1, kGieFace = 2;
constexpr float kDetW = 512, kDetH = 288;

struct Obj {
  int tid = 0, cls = 0, gie = kGiePerson, parent = -1;
  float conf = 0;
  Box b;                 // detector pixels
  std::string emb;       // fp16 L2-normalised embedding bytes (faces only)
};

struct Frame {
  int cam = 0;
  int64_t fn = 0, ts = 0;  // frame number, sensor timestamp (ns)
  std::vector<Obj> objs;
};

inline void write_obj(beni::MsgWriter& w, const Obj& o, bool full = true) {
  bool face = full && o.gie == kGieFace;
  w.map(5 + (face ? 1 : 0) + (full && !o.emb.empty() ? 1 : 0) - (full ? 0 : 1));
  w.key("tid").i(o.tid).key("cls").i(o.cls);
  if (full) w.key("gie").i(o.gie);
  w.key("conf").fr(o.conf, 3);
  w.key("bbox").arr(4).fr(o.b.x1, 1).fr(o.b.y1, 1).fr(o.b.w(), 1).fr(o.b.h(), 1);
  if (face) w.key("parent").i(o.parent);
  if (full && !o.emb.empty()) w.key("emb").bin(o.emb.data(), o.emb.size());
}

// Returns false (nothing written) when every frame is empty: silence means "nobody" to the agent.
inline bool write_det(beni::MsgWriter& w, const std::vector<Frame>& frames) {
  bool any = false;
  for (const auto& f : frames) any |= !f.objs.empty();
  if (!any) return false;
  w.clear();
  w.envelope("vision", 1).key("frames").arr(uint32_t(frames.size()));
  for (const auto& f : frames) {
    w.map(4).key("cam").i(f.cam).key("fn").i(f.fn).key("ts").i(f.ts).key("objs").arr(uint32_t(f.objs.size()));
    for (const auto& o : f.objs) write_obj(w, o);
  }
  return true;
}

inline void write_jpeg(beni::MsgWriter& w, int cam, const beni::Value* req, const void* jpg, size_t n,
                       const std::vector<Obj>& objs) {
  w.clear();
  w.envelope("vision", 4).key("cam").i(cam).key("req");
  if (req && req->type == beni::Value::STR) w.str(req->s);
  else if (req && req->type == beni::Value::INT) w.i(req->i);
  else w.nil();
  w.key("jpeg").bin(jpg, n).key("objs").arr(uint32_t(objs.size()));
  for (const auto& o : objs) write_obj(w, o, false);
}

inline void write_motion(beni::MsgWriter& w, int cam, int score) {
  w.clear();
  w.envelope("vision", 2).key("cam").i(cam).key("score").i(score);
}

// §11.9 replay topic: one sampled frame of a recording (`pts` s of media time from the segment start; `fn` its
// decode index), then {done: true} per segment. `jpeg` (320x180) rides along the first time a track shows a face.
inline void write_replay(beni::MsgWriter& w, const std::string& path, int cam, double pts, int64_t fn,
                         const std::vector<Obj>& objs, const std::string& jpeg) {
  w.clear();
  w.envelope("vision", 6).key("path").str(path).key("cam").i(cam).key("pts").fr(pts, 2).key("fn").i(fn);
  w.key("objs").arr(uint32_t(objs.size()));
  for (const auto& o : objs) write_obj(w, o);
  w.key("jpeg");
  if (jpeg.empty()) w.nil();
  else w.bin(jpeg.data(), jpeg.size());
}

inline void write_replay_done(beni::MsgWriter& w, const std::string& path, int cam, int64_t frames, bool ok) {
  w.clear();
  w.envelope("vision", 5).key("path").str(path).key("cam").i(cam).key("done").b(true).key("frames").i(frames);
  w.key("ok").b(ok);
}

// §4 item 39 pose topic: per person track, 18 keypoints (COCO order + neck) flattened as x, y, conf; x/y in the
// same 512x288 detector pixels as the det boxes, conf 0 where a part was not found.
struct Pose {
  int tid = 0;
  std::vector<float> kps;  // 54
};

inline void write_pose(beni::MsgWriter& w, int cam, int64_t fn, const std::vector<Pose>& people) {
  w.clear();
  w.envelope("vision", 3).key("cam").i(cam).key("fn").i(fn).key("people").arr(uint32_t(people.size()));
  for (const auto& p : people) {
    w.map(2).key("tid").i(p.tid).key("kps").arr(uint32_t(p.kps.size()));
    for (size_t i = 0; i < p.kps.size(); ++i) w.fr(p.kps[i], i % 3 == 2 ? 2 : 1);
  }
}

inline void write_scene(beni::MsgWriter& w, int cam, bool night, float lux) {
  w.clear();
  w.envelope("vision", 3).key("cam").i(cam).key("night").b(night).key("lux").fr(lux, 1);
}

// ---------------------------------------------------------------- vision_ctrl REP
struct Cmd {
  enum Op { BAD, PING, SNAPSHOT, BITRATE, INTERVAL, AE_REGION, IR, IDR, THUMB, REPLAY, POSE, CAM } op = BAD;
  int cam = 0, n = 0;
  float l = 0, t = 0, r = 1, b = 1;  // normalised AE region; r <= l clears it
  beni::Value req;
  std::string path;  // replay
  std::string err;
};

inline Cmd parse_cmd(const void* p, size_t n) {
  Cmd c;
  beni::Value v;
  if (!beni::unpack(p, n, v) || v.type != beni::Value::MAP) {
    c.err = "bad msgpack";
    return c;
  }
  std::string op = v.text("op");
  c.cam = int(v.num("cam", 0));
  if (c.cam < 0 || c.cam > 1) {
    c.err = "bad cam";
    return c;
  }
  if (op == "ping") c.op = Cmd::PING;
  else if (op == "snapshot") {
    c.op = Cmd::SNAPSHOT;
    if (const beni::Value* r = v.get("req")) c.req = *r;
  } else if (op == "bitrate") {
    c.op = Cmd::BITRATE, c.n = int(v.num("bps", 0));
    if (c.n < 100000 || c.n > 20000000) c.op = Cmd::BAD, c.err = "bps out of range";
  } else if (op == "interval") {
    c.op = Cmd::INTERVAL, c.n = int(v.num("n", 3));
  } else if (op == "ae_region") {
    c.op = Cmd::AE_REGION, c.l = float(v.num("l", 0)), c.t = float(v.num("t", 0)), c.r = float(v.num("r", 0)),
    c.b = float(v.num("b", 0));
  } else if (op == "idr") {
    c.op = Cmd::IDR;  // §5.3: a WebRTC viewer joined; don't make it wait up to a GOP for a keyframe
  } else if (op == "thumb") {  // §3.17 memory: episodic keyframe (no box) or a square face crop, n px wide
    c.op = Cmd::THUMB, c.n = int(v.num("n", 320)), c.l = float(v.num("l", 0)), c.t = float(v.num("t", 0)),
    c.r = float(v.num("r", 1)), c.b = float(v.num("b", 1));
    if (const beni::Value* r = v.get("req")) c.req = *r;
    if (c.n < 32 || c.n > 1024) c.op = Cmd::BAD, c.err = "n out of range";
  } else if (op == "replay") {  // §11.9: queue a recording segment (path), stop (stop: 1), or just report
    c.op = Cmd::REPLAY, c.path = v.text("path"), c.n = int(v.num("stride", 6)), c.r = float(v.num("stop", 0));
    if (c.n < 1 || c.n > 300) c.op = Cmd::BAD, c.err = "bad stride";
  } else if (op == "pose") {  // §4 item 39: keypoints on this camera for `secs` (0 stops)
    c.op = Cmd::POSE, c.r = float(v.num("secs", 10));
    if (c.r < 0 || c.r > 120) c.op = Cmd::BAD, c.err = "bad secs";
  } else if (op == "cam") {  // §15.1: pause (on: 0) / resume a camera's capture
    c.op = Cmd::CAM, c.n = int(v.num("on", 1));
  } else if (op == "ir") {
    c.op = Cmd::IR, c.n = int(v.num("duty", -1));  // 0..100, -1 = automatic (night logic)
  } else {
    c.err = "unknown op";
  }
  return c;
}

// Thumbnail geometry in W x H source pixels: the whole frame -> n x (n*H/W), or a normalised box grown to a centred
// square (clamped inside the frame) -> n x n. Everything even, as VIC and NVJPG want for YUV420.
struct ThumbGeom {
  int x, y, cw, ch, w, h;
};

inline ThumbGeom thumb_geom(float l, float t, float r, float b, int W, int H, int n) {
  n &= ~1;
  if (r <= l || b <= t || (r - l >= 0.999f && b - t >= 0.999f)) return ThumbGeom{0, 0, W, H, n, (n * H / W) & ~1};
  int s = int(std::max((r - l) * float(W), (b - t) * float(H)));
  s = std::min(std::max(s, 16), std::min(W, H)) & ~1;
  int x = int(0.5f * (l + r) * float(W)) - s / 2, y = int(0.5f * (t + b) * float(H)) - s / 2;
  x = std::min(std::max(x, 0), W - s) & ~1, y = std::min(std::max(y, 0), H - s) & ~1;
  return ThumbGeom{x, y, s, s, n, n};
}

// §3.17 sched.sock: true when the scheduler admits background GPU work (`admit.gpu` and not `pause`).
inline bool parse_sched(const void* p, size_t n, bool& gpu_ok) {
  beni::Value v;
  if (!beni::unpack(p, n, v) || v.type != beni::Value::MAP) return false;
  const beni::Value* a = v.get("admit");
  gpu_ok = v.num("pause", 0) == 0 && a && a->num("gpu", 0) != 0;
  return true;
}

}  // namespace vc
