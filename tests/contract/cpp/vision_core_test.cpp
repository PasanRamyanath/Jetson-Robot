// Host tests for vision_core's portable parts (§5.5). `vc_test` runs the checks; `vc_test det` prints a det message;
// `vc_test ctrl` parses a msgpack control request from stdin and prints it.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "bytetrack.hpp"
#include "logic.hpp"
#include "messages.hpp"
#include "ts_demux.hpp"
#include "ts_mux.hpp"

using namespace vc;

static int fails = 0;
#define CHECK(c)                                                  \
  do {                                                            \
    if (!(c)) std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c), ++fails; \
  } while (0)

static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

static void test_decode() {
  float rows[] = {10, 20, 110, 220, 0.9f, 0, 0, 0, 5, 5, 0.2f, 0, 50, 50, 60, 60, 0.8f, 41, 480, -5, 700, 100, 0.5f, 0};
  std::vector<Det> out;
  decode_e2e(rows, 4, 0.35f, ClassMask::only({0}), Letterbox(), kDetW, kDetH, out);
  CHECK(out.size() == 2);  // low score and disallowed class dropped
  CHECK(out[0].cls == 0 && near(out[0].b.x2, 110, 1e-4f));
  CHECK(near(out[1].b.x1, 480, 1e-4f) && near(out[1].b.x2, 512, 1e-4f) && near(out[1].b.y1, 0, 1e-4f));
  Letterbox lb = Letterbox::fit(1280, 720, 320, 320);  // 0.25 scale, 70 px bars top/bottom
  CHECK(near(lb.scale, 0.25f, 1e-6f) && near(lb.pad_y, 70, 1e-4f) && near(lb.uy(70), 0, 1e-4f));
  float boxes[] = {0, 0, 10, 10, 1, 1, 50, 50}, scores[] = {0.9f, 0.4f}, classes[] = {0, 0};
  out.clear();
  decode_nms(2, boxes, scores, classes, 0.5f, ClassMask(), Letterbox(), kDetW, kDetH, out);
  CHECK(out.size() == 1);
}

static void test_assign() {
  std::vector<float> c = {4, 1, 3, 2, 0, 5, 3, 2, 2};
  auto m = assign(c, 3, 3, 10);
  float tot = 0;
  for (auto& p : m) tot += c[p.first * 3 + p.second];
  CHECK(m.size() == 3 && near(tot, 5, 1e-5f));
  std::vector<float> r = {0.1f, 0.9f, 0.95f, 0.2f, 0.95f, 0.99f};  // 3 tracks x 2 dets, third track gated out
  m = assign(r, 3, 2, 0.8f);
  CHECK(m.size() == 2);
  for (auto& p : m) CHECK(p.first == p.second);
  CHECK(assign(r, 0, 2, 0.8f).empty());
}

static Det det(float x, float y, float w, float h, float s = 0.9f, int cls = 0) {
  Det d;
  d.b = Box{x, y, x + w, y + h}, d.score = s, d.cls = cls;
  return d;
}

static void test_bytetrack() {
  ByteTracker bt;
  int id = -1;
  float maxerr = 0;
  // Two people walking towards each other at 60 fps, detector every 4th frame; B is partly occluded (low score)
  // around the crossing. IDs must survive; between ticks the Kalman prediction must follow the motion.
  for (int f = 0; f < 240; ++f) {
    double t = f / 60.0;
    float ax = 20 + 1.5f * f, bx = 460 - 1.5f * f;
    bt.predict(t);
    if (f % 4 == 0) {
      std::vector<Det> d = {det(ax, 60, 50, 150)};
      bool crossing = std::fabs(ax - bx) < 60;
      d.push_back(det(bx, 70, 50, 150, crossing ? 0.3f : 0.85f));
      bt.update(t, d);
    }
    int n = 0;
    bt.each_active([&](const Track& tr) {
      ++n;
      if (tr.id == 1 && f > 20) maxerr = std::max(maxerr, std::fabs(tr.box().x1 - ax));
    });
    if (f == 0) {
      CHECK(n == 2);  // the first tick confirms immediately
      bt.each_active([&](const Track& tr) { if (tr.box().x1 < 100) id = tr.id; });
    }
  }
  CHECK(id == 1);
  CHECK(maxerr < 8.f);  // px, at 90 px/s with a 15 Hz detector
  int ids = 0;
  bt.each_active([&](const Track& tr) { ids += tr.id; });
  CHECK(ids == 3);  // still tracks 1 and 2: no swaps into new IDs through the occlusion
  // Disappear for 1 s -> lost, then back near the predicted spot -> same ID. Gone for 2 s -> retired.
  ByteTracker b2;
  b2.predict(0), b2.update(0, {det(100, 100, 40, 100)});
  int n = 0;
  for (double t = 1 / 60.0; t < 1.0; t += 1 / 60.0) b2.predict(t), b2.update(t, {});
  b2.each_active([&](const Track&) { ++n; });
  CHECK(n == 0 && b2.tracks().size() == 1);
  b2.predict(1.0), b2.update(1.0, {det(102, 100, 40, 100)});
  b2.each_active([&](const Track& tr) { CHECK(tr.id == 1); ++n; });
  CHECK(n == 1);
  for (double t = 1.0; t < 3.1; t += 0.25) b2.predict(t), b2.update(t, {});
  CHECK(b2.tracks().empty());
  // A new object after the first tick needs a second hit before it is published.
  b2.predict(4.0), b2.update(4.0, {det(300, 100, 40, 100)});
  n = 0;
  b2.each_active([&](const Track&) { ++n; });
  CHECK(n == 0);
  b2.predict(4.07), b2.update(4.07, {det(301, 100, 40, 100)});
  b2.each_active([&](const Track& tr) { CHECK(tr.id == 2); ++n; });
  CHECK(n == 1);
  // A class change never continues a track.
  b2.predict(4.13), b2.update(4.13, {det(301, 100, 40, 100, 0.9f, 56)});
  n = 0;
  b2.each_active([&](const Track& tr) { n += tr.cls == 56; });
  CHECK(n == 0);
}

static void test_faces() {
  // A known similarity (scale 1.7, 20 deg, shift) applied to the template must be recovered exactly.
  float c = 1.7f * std::cos(0.349f), s = 1.7f * std::sin(0.349f), kps[10], M[6];
  for (int i = 0; i < 5; ++i) {
    float x = kArcface[2 * i], y = kArcface[2 * i + 1];
    kps[2 * i] = c * x - s * y + 400, kps[2 * i + 1] = s * x + c * y + 120;
  }
  similarity(kps, M);
  CHECK(near(M[0], c, 1e-4f) && near(M[1], -s, 1e-4f) && near(M[2], 400, 1e-2f));
  CHECK(near(M[3], s, 1e-4f) && near(M[4], c, 1e-4f) && near(M[5], 120, 1e-2f));
  // SCRFD: one anchor fires at stride 16, cell (3, 2), anchor 1, on a 320 crop letterboxed from a 160x160 region.
  const int in = 320;
  std::vector<float> sc[3], bb[3], kp[3];
  for (int k = 0; k < 3; ++k) {
    int g = in / (8 << k), n = g * g * 2;
    sc[k].assign(n, 0.f), bb[k].assign(4 * n, 0.f), kp[k].assign(10 * n, 0.f);
  }
  int i = (2 * 20 + 3) * 2 + 1;
  sc[1][i] = 0.8f;
  sc[1][i - 1] = 0.7f;  // the twin anchor on the same cell: removed by NMS
  float d[4] = {1, 1, 2, 2};
  for (int j = 0; j < 4; ++j) bb[1][4 * i + j] = d[j], bb[1][4 * (i - 1) + j] = d[j];
  for (int j = 0; j < 10; ++j) kp[1][10 * i + j] = 0.5f;
  const float* S[3] = {sc[0].data(), sc[1].data(), sc[2].data()};
  const float* B[3] = {bb[0].data(), bb[1].data(), bb[2].data()};
  const float* K[3] = {kp[0].data(), kp[1].data(), kp[2].data()};
  std::vector<Face> faces;
  Letterbox lb;
  lb.scale = 2, lb.pad_x = -2 * 600, lb.pad_y = -2 * 100;  // crop origin (600, 100) in the frame, 2x upscale
  decode_scrfd(S, B, K, in, 0.5f, lb, faces);
  CHECK(faces.size() == 1 && near(faces[0].score, 0.8f, 1e-6f));
  if (!faces.empty()) {  // anchor centre (48, 32) in crop px; box 16 px each side up-left, 32 px down-right
    CHECK(near(faces[0].b.x1, 600 + 16, 1e-3f) && near(faces[0].b.y2, 100 + 32, 1e-3f));
    CHECK(near(faces[0].kps[0], 600 + 28, 1e-3f));
  }
  Box h = head_crop(Box{100, 50, 180, 250}, 512, 288);
  CHECK(near(h.w(), 96, 1e-3f) && h.y1 >= 40 && h.cx() == 140);
  FacePacer fp;
  std::vector<std::pair<int64_t, Box>> cand = {{1, Box{0, 0, 10, 10}}, {2, Box{0, 0, 50, 50}}, {3, Box{0, 0, 5, 5}}};
  auto pick = fp.pick(cand, 0, 2);
  CHECK(pick.size() == 2 && pick[0] == 1 && pick[1] == 0);
  fp.done(2, 0), fp.done(1, 0);
  pick = fp.pick(cand, 0.1, 4);
  CHECK(pick.size() == 1 && pick[0] == 2);
  CHECK(fp.pick(cand, 0.25, 4).size() == 3);
}

static void test_gates() {
  MotionGate mg;
  std::vector<float> still(240, 0.5f), moving(240, 0.5f);
  for (int i = 0; i < 10; ++i) moving[i] = 5;
  bool fired = false;
  for (int f = 0; f < 10; ++f) fired |= mg.feed(still.data(), 240, f / 15.0);
  CHECK(!fired);
  CHECK(!mg.feed(moving.data(), 240, 1.0) && !mg.feed(moving.data(), 240, 1.07) && mg.feed(moving.data(), 240, 1.13));
  CHECK(!mg.feed(moving.data(), 240, 1.2));  // refractory
  CHECK(mg.last_score == 10);
  NightLogic nl;
  int ev = 0;
  for (int f = 0; f < 60 * 4; ++f) ev += nl.feed(10, f / 60.0);
  CHECK(ev == 0 && !nl.night);
  for (int f = 240; f < 60 * 7; ++f) ev += nl.feed(10, f / 60.0);
  CHECK(ev == 1 && nl.night);
  for (int f = 420; f < 60 * 14; ++f) ev += nl.feed(60, f / 60.0);  // between on and off: stays night
  CHECK(nl.night);
  for (int f = 840; f < 60 * 21; ++f) ev += nl.feed(500, f / 60.0);
  CHECK(!nl.night && ev == 0);
  CHECK(near(NightLogic::lux_proxy(2, 1, 10000000), 50, 1e-3f));
  Cadence cd;
  int runs = 0;
  for (uint64_t f = 0; f < 60; ++f) runs += cd.due(f, 0);
  CHECK(runs == 15);
  cd.set_interval(59), runs = 0;
  for (uint64_t f = 0; f < 60; ++f) runs += cd.due(f, 0);
  CHECK(runs == 1);
  cd.motion(10), runs = 0;
  for (uint64_t f = 0; f < 60; ++f) runs += cd.due(f, 10.5);
  CHECK(runs == 15);
  cd.set_interval(-1), runs = 0;
  for (uint64_t f = 0; f < 60; ++f) runs += cd.due(f, 20);
  CHECK(runs == 0);
  CHECK(f2h(1.f) == 0x3c00 && f2h(-2.f) == 0xc000 && f2h(65504.f) == 0x7bff && f2h(1e6f) == 0x7c00);
  CHECK(f2h(0.1f) == 0x2e66 && f2h(1e-8f) == 0 && f2h(6e-5f) == 0x03ef && f2h(0.f) == 0);
}

static void test_sched() {
  bool ok = false;
  beni::MsgWriter w;
  w.map(2).str("admit").map(2).str("gpu").b(true).str("cpu").b(false).str("pause").b(false);
  CHECK(parse_sched(w.buf.data(), w.buf.size(), ok) && ok);
  w.clear(), w.map(2).str("admit").map(1).str("gpu").b(true).str("pause").b(true);
  CHECK(parse_sched(w.buf.data(), w.buf.size(), ok) && !ok);
  w.clear(), w.map(1).str("pause").b(false);
  CHECK(parse_sched(w.buf.data(), w.buf.size(), ok) && !ok);
  const char not_map[1] = {1};
  CHECK(!parse_sched(not_map, 1, ok));
}

static void test_reid() {
  float a[3] = {1, 0, 0}, b[3] = {0, 1, 0}, a2[3] = {0.95f, 0.3122f, 0};
  ReidBank rb;
  std::vector<int> live = {5, 6};
  CHECK(rb.due(live, 0, 1).size() == 1);
  CHECK(!rb.add(5, a, 3, 0) && !rb.add(6, b, 3, 0));
  CHECK(rb.due(live, 0.1, 4).empty() && rb.due(live, 0.3, 4).size() == 2);  // probing at 4 Hz
  rb.add(5, a, 3, 4.0), rb.add(6, b, 3, 4.0);
  CHECK(rb.due(live, 4.5, 4).empty() && rb.due(live, 5.1, 4).size() == 2);  // then 1 Hz refresh
  rb.sync({6}, 5.0);  // ByteTrack forgot 5 (occluded)
  CHECK(rb.lost() == 1);
  CHECK(!rb.add(9, b, 3, 6.0) && rb.alias(9) == 9);  // a different person is not merged
  CHECK(rb.add(8, a2, 3, 6.0) && rb.alias(8) == 5 && rb.lost() == 0);  // the same person returns as tid 5
  rb.sync({6, 8, 9}, 7.0);
  rb.sync({6, 9}, 8.0);  // lost again: re-published under 5, not 8
  CHECK(rb.lost() == 1 && rb.add(12, a, 3, 9.0) && rb.alias(12) == 5);
  rb.sync({6, 9, 12}, 9.5);
  rb.add(20, b, 3, 10.0);
  rb.sync({20}, 10.5);  // 6, 9 and 12 (=5) forgotten; 20 is past probing by 14 s, so it keeps its own id
  CHECK(rb.lost() == 3 && !rb.add(20, a, 3, 14.0) && rb.alias(20) == 20);
  rb.sync({}, 21.0);
  rb.sync({}, 60.0);  // gallery expiry
  CHECK(rb.lost() == 0);
}

static std::string emb_bytes() {
  float v[4] = {1, 2, 2, 4};
  l2norm(v, 4);
  std::string e;
  for (float x : v) {
    uint16_t h = f2h(x);
    e.push_back(char(h & 0xff)), e.push_back(char(h >> 8));  // little-endian like numpy on the Nano
  }
  return e;
}

void test_demux() {
  for (bool hevc : {true, false}) {
    std::string ts;
    TsMux mux(hevc, [&](const uint8_t* p, size_t n) { ts.append((const char*)p, n); });
    std::vector<std::string> aus;
    for (int i = 0; i < 5; ++i) {
      std::string a = std::string("\0\0\0\1", 4) + char(hevc ? (i ? 0x02 : 0x26) : (i ? 0x41 : 0x65)) +
                      std::string(size_t(37 + i * 211), char('a' + i));
      aus.push_back(a);
      mux.write((const uint8_t*)a.data(), a.size(), 90000 + uint64_t(i) * 3000, i == 0);
    }
    std::vector<std::pair<std::string, uint64_t>> got;
    TsDemux dm([&](const uint8_t* p, size_t n, uint64_t pts) {
      got.emplace_back(std::string((const char*)p, n), pts);
    });
    for (size_t o = 0; o < ts.size(); o += 101)  // odd chunking: packets straddle feed() calls
      dm.feed((const uint8_t*)ts.data() + o, std::min<size_t>(101, ts.size() - o));
    dm.flush();
    CHECK(dm.hevc() == hevc && dm.video_pid() == TsMux::kVidPid);
    CHECK(got.size() == aus.size());
    for (size_t i = 0; i < got.size() && i < aus.size(); ++i)
      CHECK(got[i].first == aus[i] && got[i].second == 90000 + i * 3000);
  }
}

void test_thumb() {
  ThumbGeom k = thumb_geom(0, 0, 1, 1, 1280, 720, 320);
  CHECK(k.x == 0 && k.cw == 1280 && k.w == 320 && k.h == 180);
  ThumbGeom f = thumb_geom(0.5f, 0.5f, 0.55f, 0.6f, 1280, 720, 128);  // 64 x 72 px box -> 72 px square
  CHECK(f.cw == 72 && f.ch == 72 && f.w == 128 && f.h == 128 && f.x == 636 && f.y == 360);
  ThumbGeom e = thumb_geom(0.95f, 0.9f, 1.f, 1.f, 1280, 720, 128);   // clamped inside the frame
  CHECK(e.x + e.cw <= 1280 && e.y + e.ch <= 720 && e.x % 2 == 0 && e.y % 2 == 0);
}

int main(int argc, char** argv) {
#ifdef _WIN32  // msgpack/TS bytes on stdin/stdout: no CRLF translation
  _setmode(_fileno(stdin), _O_BINARY), _setmode(_fileno(stdout), _O_BINARY);
#endif
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "det") {
    Frame f0, f1;
    f0.cam = 0, f0.fn = 42, f0.ts = 1234567890123LL;
    Obj p;
    p.tid = 7, p.cls = 0, p.conf = 0.91234f, p.b = Box{10.04f, 20, 60.5f, 170};
    Obj fc;
    fc.tid = 7, fc.gie = kGieFace, fc.parent = 7, fc.conf = 0.8f, fc.b = Box{20, 25, 40, 50}, fc.emb = emb_bytes();
    f0.objs = {p, fc};
    f1.cam = 1, f1.fn = 43;
    beni::MsgWriter w;
    CHECK(!write_det(w, {f1}));
    write_det(w, {f0, f1});
    std::fwrite(w.buf.data(), 1, w.buf.size(), stdout);
    return fails ? 1 : 0;
  }
  if (mode == "ts") {  // three H.265 access units: IDR (multi-packet), P, P sized to need 1 byte of stuffing
    TsMux mux(true, [](const uint8_t* p, size_t n) { std::fwrite(p, 1, n, stdout); });
    const std::string sc("\0\0\0\1", 4);
    std::string idr = sc + "\x26\x01" + std::string(1000, 'k');
    std::string p1 = sc + "\x02\x01" + std::string(50, 'p');
    std::string p2 = sc + "\x02\x01" + std::string(155, 'q');  // 14 B PES header + 161 = 175 of 176: 1 B stuffing
    CHECK(is_keyframe((const uint8_t*)idr.data(), idr.size(), true));
    CHECK(!is_keyframe((const uint8_t*)p1.data(), p1.size(), true));
    mux.write((const uint8_t*)idr.data(), idr.size(), 90000, true);
    mux.write((const uint8_t*)p1.data(), p1.size(), 93000, false);
    mux.write((const uint8_t*)p2.data(), p2.size(), 96000, false);
    return fails ? 1 : 0;
  }
  if (mode == "replay") {  // one sampled frame with a face + keyframe, then the segment summary
    Obj p;
    p.tid = 3, p.cls = 0, p.conf = 0.8f, p.b = Box{10, 20, 60, 170};
    Obj fc;
    fc.tid = -1, fc.gie = kGieFace, fc.parent = 3, fc.conf = 0.7f, fc.b = Box{20, 25, 40, 50}, fc.emb = emb_bytes();
    beni::MsgWriter w;
    write_replay(w, "/ssd/beni/rec/cam1_x.ts", 1, 12.345, 72, {p, fc}, std::string("\xff\xd8jpg", 5));
    std::string a = w.buf;
    write_replay_done(w, "/ssd/beni/rec/cam1_x.ts", 1, 900, true);
    uint32_t n = uint32_t(a.size());
    std::fwrite(&n, 4, 1, stdout), std::fwrite(a.data(), 1, a.size(), stdout);
    std::fwrite(w.buf.data(), 1, w.buf.size(), stdout);
    return fails ? 1 : 0;
  }
  if (mode == "pose") {  // two tracks; the second found only its nose
    Pose a, b;
    a.tid = 4, b.tid = 1000002;
    for (int k = 0; k < 18; ++k) a.kps.insert(a.kps.end(), {10.f + k, 20.5f + k, 0.5f});
    b.kps.assign(54, 0.f), b.kps[0] = 100.04f, b.kps[1] = 50.f, b.kps[2] = 0.333f;
    beni::MsgWriter w;
    write_pose(w, 1, 4242, {a, b});
    std::fwrite(w.buf.data(), 1, w.buf.size(), stdout);
    return fails ? 1 : 0;
  }
  if (mode == "ctrl") {
    std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    Cmd c = parse_cmd(in.data(), in.size());
    std::printf("op=%d cam=%d n=%d l=%.2f r=%.2f req=%lld err=%s path=%s\n", int(c.op), c.cam, c.n, c.l, c.r,
                (long long)c.req.i, c.err.c_str(), c.path.c_str());
    return 0;
  }
  test_decode();
  test_assign();
  test_bytetrack();
  test_faces();
  test_gates();
  test_reid();
  test_sched();
  test_thumb();
  test_demux();
  if (fails) return 1;
  std::puts("ok");
  return 0;
}
