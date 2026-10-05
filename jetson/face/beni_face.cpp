// beni_face (§3.11): Beni's face on the 5" 800x480 HDMI LCD, composed by the display controller (no X, no GPU).
//   plane 0  expression clips: Annex-B H.264 in /ssd/face/<name>.h264 (one AU per frame, IDR at frame 0) fed from RAM
//            through appsrc -> NVDEC. A clip change cross-cuts to the new clip's IDR on the next frame tick (<100 ms).
//   plane 1  overlay: pupils (gaze) + status icons, 400x240 RGBA drawn on the CPU only when something moved, upscaled
//            by VIC and alpha-blended by the DC.
//   plane 2  optional picture-in-picture of cam0 ("show me what you see"), pulled from mediamtx only while shown.
// Inputs (ZMQ): PULL face_ctrl {op: expression|gaze|overlay|pip}, SUB events (voice_state, brain, person_seen,
// privacy), robot_state (battery, charging, mic_muted), vision det (gie-2 faces on cam0 -> gaze), sched (mode ->
// sleepy).
// Budget (§14.1): CPU 0, 3-6 %, 40-60 MB. C++14 (host gcc-7), GStreamer 1.14 + libzmq 4.2 from L4T 32.7.
#include <dirent.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <sys/stat.h>
#include <zmq.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "schemas.hpp"

namespace {

constexpr int kW = 800, kH = 480;      // LCD / clip size
constexpr int kOW = 400, kOH = 240;    // overlay canvas (VIC upscales 2x)
constexpr int kFaceGie = 2;            // beni_vision FACE_DET_GIE
constexpr float kMuxW = 512, kMuxH = 288;

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

double mono() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void note(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void note(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fputs("beni_face: ", stderr);
  std::vfprintf(stderr, fmt, ap);
  std::fputc('\n', stderr);
  va_end(ap);
}

// ---------------------------------------------------------------- clips
struct Clip {
  std::string name, data;
  std::vector<std::pair<size_t, size_t>> au;  // (offset, size) per frame
  double fps = 30;
  bool eyes = false;                         // sidecar present: pupils are drawn on plane 1
  float lcx = 0, lcy = 0, rcx = 0, rcy = 0;  // eye centres (800x480 px)
  float hw = 0, hh = 0, pr = 0;              // eye half-width, half-height, pupil radius
  std::vector<float> lid;                    // per-frame eyelid openness, 0 closed .. 1 open
  float open(size_t i) const { return lid.empty() ? 1.f : lid[std::min(i, lid.size() - 1)]; }
};

// Splits an Annex-B stream into access units: a new AU starts at an AUD, SEI/SPS/PPS or a slice with
// first_mb_in_slice == 0, once the current AU already holds a VCL NAL (H.264 7.4.1.2.3).
std::vector<std::pair<size_t, size_t>> split_aus(const std::string& d) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(d.data());
  const size_t n = d.size();
  std::vector<std::pair<size_t, size_t>> out;
  size_t start = 0;
  bool vcl = false, any = false;
  for (size_t i = 0; i + 3 < n; ++i) {
    if (p[i] || p[i + 1] || p[i + 2] != 1) continue;
    size_t pre = (i > 0 && p[i - 1] == 0) ? i - 1 : i, hdr = i + 3;
    int t = p[hdr] & 0x1f;
    bool slice = t == 1 || t == 5;
    bool first = slice && hdr + 1 < n && (p[hdr + 1] & 0x80);
    if (!any) start = pre, any = true;
    if (vcl && (t == 9 || t == 6 || t == 7 || t == 8 || first)) {
      out.emplace_back(start, pre - start);
      start = pre;
      vcl = false;
    }
    vcl = vcl || slice;
    i = hdr;
  }
  if (vcl) out.emplace_back(start, n - start);
  return out;
}

bool read_file(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

// <name>.eyes sidecar (written by assets/make_clips.py):
//   fps 30
//   eyes <lcx> <lcy> <rcx> <rcy> <half_w> <half_h> <pupil_r>
//   lid <open0> <open1> ...
void read_sidecar(const std::string& path, Clip& c) {
  std::ifstream f(path);
  std::string line, k;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    if (!(ss >> k)) continue;
    if (k == "fps") ss >> c.fps;
    else if (k == "eyes") c.eyes = bool(ss >> c.lcx >> c.lcy >> c.rcx >> c.rcy >> c.hw >> c.hh >> c.pr);
    else if (k == "lid") for (float v; ss >> v;) c.lid.push_back(v);
  }
  if (!(c.fps > 1 && c.fps <= 60)) c.fps = 30;
}

std::map<std::string, std::unique_ptr<Clip>> load_clips(const std::string& dir) {
  std::map<std::string, std::unique_ptr<Clip>> clips;
  DIR* d = opendir(dir.c_str());
  if (!d) return clips;
  while (dirent* e = readdir(d)) {
    std::string fn = e->d_name;
    if (fn.size() < 6 || fn.compare(fn.size() - 5, 5, ".h264") != 0) continue;
    std::unique_ptr<Clip> c(new Clip);
    c->name = fn.substr(0, fn.size() - 5);
    if (!read_file(dir + "/" + fn, c->data)) continue;
    c->au = split_aus(c->data);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(c->data.data());
    bool idr = false;
    if (!c->au.empty())
      for (size_t i = c->au[0].first; i + 3 < c->au[0].first + c->au[0].second && !idr; ++i)
        idr = !p[i] && !p[i + 1] && p[i + 2] == 1 && (p[i + 3] & 0x1f) == 5;
    if (!idr) {
      note("%s: no IDR in the first access unit, skipped", fn.c_str());
      continue;
    }
    read_sidecar(dir + "/" + c->name + ".eyes", *c);
    note("clip %s: %lu frames @ %.0f fps, %lu KB%s", c->name.c_str(), static_cast<unsigned long>(c->au.size()), c->fps,
         static_cast<unsigned long>(c->data.size() / 1024), c->eyes ? ", eyes" : "");
    std::string name = c->name;
    clips[name] = std::move(c);
  }
  closedir(d);
  return clips;
}

// ---------------------------------------------------------------- overlay canvas (RGBA, straight alpha)
struct Rgba {
  uint8_t r, g, b, a;
};

class Canvas {
 public:
  std::vector<uint8_t> px = std::vector<uint8_t>(size_t(kOW) * kOH * 4, 0);
  void clear() { std::fill(px.begin(), px.end(), 0); }

  void blend(int x, int y, Rgba c, float cov) {
    if (x < 0 || y < 0 || x >= kOW || y >= kOH || cov <= 0) return;
    uint8_t* d = &px[(size_t(y) * kOW + x) * 4];
    float a = std::min(1.f, cov) * c.a / 255.f, da = d[3] / 255.f, oa = a + da * (1 - a);
    if (oa <= 0) return;
    const uint8_t s[3] = {c.r, c.g, c.b};
    for (int k = 0; k < 3; ++k) d[k] = uint8_t((s[k] * a + d[k] * da * (1 - a)) / oa + 0.5f);
    d[3] = uint8_t(oa * 255 + 0.5f);
  }
  // Anti-aliased disc, optionally clipped by an ellipse (ex, ey, erx, ery) with a soft edge.
  void disc(float cx, float cy, float r, Rgba c, const float* clip = nullptr) {
    for (int y = int(cy - r - 1); y <= int(cy + r + 1); ++y)
      for (int x = int(cx - r - 1); x <= int(cx + r + 1); ++x) {
        float cov = r + 0.5f - std::hypot(x + 0.5f - cx, y + 0.5f - cy);
        if (cov <= 0) continue;
        if (clip) {
          if (clip[3] <= 0.5f) return;
          float u = (x + 0.5f - clip[0]) / clip[2], v = (y + 0.5f - clip[1]) / clip[3];
          cov = std::min(cov, (1 - std::sqrt(u * u + v * v)) * std::min(clip[2], clip[3]) + 0.5f);
        }
        blend(x, y, c, cov);
      }
  }
  // Thick anti-aliased segment (capsule).
  void line(float x0, float y0, float x1, float y1, float w, Rgba c) {
    float dx = x1 - x0, dy = y1 - y0, l2 = std::max(1e-6f, dx * dx + dy * dy), r = w / 2;
    for (int y = int(std::min(y0, y1) - r - 1); y <= int(std::max(y0, y1) + r + 1); ++y)
      for (int x = int(std::min(x0, x1) - r - 1); x <= int(std::max(x0, x1) + r + 1); ++x) {
        float px_ = x + 0.5f, py = y + 0.5f;
        float t = std::max(0.f, std::min(1.f, ((px_ - x0) * dx + (py - y0) * dy) / l2));
        blend(x, y, c, r + 0.5f - std::hypot(px_ - x0 - t * dx, py - y0 - t * dy));
      }
  }
  void rect(int x, int y, int w, int h, Rgba c) {
    for (int j = y; j < y + h; ++j)
      for (int i = x; i < x + w; ++i) blend(i, j, c, 1);
  }
  void frame(int x, int y, int w, int h, Rgba c) {
    rect(x, y, w, 1, c), rect(x, y + h - 1, w, 1, c), rect(x, y, 1, h, c), rect(x + w - 1, y, 1, h, c);
  }
};

const Rgba kWhite{235, 235, 240, 255}, kGrey{150, 150, 160, 230}, kRed{230, 60, 50, 255}, kAmber{240, 170, 30, 255},
    kGreen{80, 200, 90, 255}, kPupil{18, 20, 32, 255}, kGlint{255, 255, 255, 235};

// ---------------------------------------------------------------- gstreamer helpers
GstElement* launch(const std::string& desc) {
  GError* err = nullptr;
  GstElement* p = gst_parse_launch(desc.c_str(), &err);
  if (err) {
    note("pipeline error: %s\n  %s", err->message, desc.c_str());
    g_error_free(err);
    if (p) gst_object_unref(p);
    return nullptr;
  }
  if (gst_element_set_state(p, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    note("cannot start: %s", desc.c_str());
    gst_element_set_state(p, GST_STATE_NULL);
    gst_object_unref(p);
    return nullptr;
  }
  return p;
}

void stop(GstElement*& p) {
  if (!p) return;
  gst_element_set_state(p, GST_STATE_NULL);
  gst_object_unref(p);
  p = nullptr;
}

GstAppSrc* appsrc_of(GstElement* p, const char* caps) {
  GstElement* e = gst_bin_get_by_name(GST_BIN(p), "src");
  GstCaps* c = gst_caps_from_string(caps);
  g_object_set(e, "caps", c, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", FALSE, "max-bytes",
               guint64(4 << 20), nullptr);
  gst_caps_unref(c);
  GstAppSrc* s = GST_APP_SRC(e);
  gst_object_unref(e);  // the pipeline keeps it alive
  return s;
}

// True if the pipeline posted an error/EOS since the last call.
bool failed(GstElement* p) {
  if (!p) return false;
  GstBus* bus = gst_element_get_bus(p);
  bool bad = false;
  while (GstMessage* m = gst_bus_pop_filtered(bus, GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS))) {
    if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR) {
      GError* e = nullptr;
      gchar* dbg = nullptr;
      gst_message_parse_error(m, &e, &dbg);
      note("%s: %s (%s)", GST_OBJECT_NAME(GST_MESSAGE_SRC(m)), e ? e->message : "?", dbg ? dbg : "");
      if (e) g_error_free(e);
      g_free(dbg);
    }
    bad = true;
    gst_message_unref(m);
  }
  gst_object_unref(bus);
  return bad;
}

// ---------------------------------------------------------------- the face
struct Opts {
  std::string dir = "/ssd/face", pip_url = "rtsp://127.0.0.1:8554/teleop";
  int conn = 0, gaze_cam = 0, lag = 2;
  bool overlay = true;
};

class Face {
 public:
  explicit Face(const Opts& o) : o_(o), rng_(std::random_device{}()) {}

  int run() {
    clips_ = load_clips(o_.dir);
    if (clips_.empty()) note("no clips in %s (run assets/make_clips.py): overlay only", o_.dir.c_str());
    void* ctx = zmq_ctx_new();
    zmq_ctx_set(ctx, ZMQ_IO_THREADS, 1);
    void* ctrl = sock(ctx, ZMQ_PULL, beni::ep::kFaceCtrl, nullptr, true);
    void* ev = sock(ctx, ZMQ_SUB, beni::ep::kEvents, "event", false);
    void* st = sock(ctx, ZMQ_SUB, beni::ep::kRobotState, beni::topic::kState, false);
    void* vis = sock(ctx, ZMQ_SUB, beni::ep::kVision, beni::topic::kDet, false);
    void* sch = sock(ctx, ZMQ_SUB, beni::ep::kSched, "sched", false);
    zmq_pollitem_t items[] = {{ctrl, 0, ZMQ_POLLIN, 0}, {ev, 0, ZMQ_POLLIN, 0}, {st, 0, ZMQ_POLLIN, 0},
                              {vis, 0, ZMQ_POLLIN, 0}, {sch, 0, ZMQ_POLLIN, 0}};
    t0_ = mono();
    double next = t0_, retry = 0, ovl_retry = 0, wifi_t = 0;
    while (!g_stop) {
      double now = mono();
      zmq_poll(items, 5, std::max(0, int((next - now) * 1000)));
      for (int k = 0; k < 5; ++k)
        if (items[k].revents & ZMQ_POLLIN) drain(items[k].socket, k);
      now = mono();
      if (now < next) continue;
      if (failed(video_)) stop(video_), vsrc_ = nullptr;
      if (failed(ovl_)) stop(ovl_), osrc_ = nullptr;
      if (failed(pip_)) stop(pip_), pip_on_ = false;
      if (!video_ && now >= retry) start_video(), retry = now + 2;
      if (o_.overlay && !ovl_ && now >= ovl_retry) start_overlay(), ovl_retry = now + 30;
      if (now >= wifi_t) read_wifi(), wifi_t = now + 5;
      tick(now);
      next += cur_ ? 1.0 / cur_->fps : 1.0 / 30;
      if (next < now - 0.25) next = now;  // stalled (e.g. SIGSTOP): don't burst
    }
    stop(pip_), stop(ovl_), stop(video_);
    vsrc_ = osrc_ = nullptr;
    for (void* s : {ctrl, ev, st, vis, sch}) zmq_close(s);
    zmq_ctx_term(ctx);
    return 0;
  }

 private:
  Opts o_;
  std::mt19937 rng_;
  std::map<std::string, std::unique_ptr<Clip>> clips_;
  const Clip* cur_ = nullptr;
  size_t idx_ = 0;
  uint64_t frames_ = 0;
  double t0_ = 0;
  GstElement *video_ = nullptr, *ovl_ = nullptr, *pip_ = nullptr;
  GstAppSrc *vsrc_ = nullptr, *osrc_ = nullptr;
  bool pip_on_ = false;
  // state
  std::string voice_ = "idle", mode_ = "active", transient_;
  double transient_until_ = 0, happy_t_ = -1e9;
  int battery_ = -1, wifi_ = -1;  // wifi: -1 no wireless iface, 0 disconnected, else link quality %
  bool charging_ = false, mic_muted_ = false, cloud_off_ = false, privacy_ = false;
  // gaze (-1..1, screen space: +x right, +y down as seen by the viewer)
  float gx_ = 0, gy_ = 0, tx_ = 0, ty_ = 0;
  double face_t_ = -1e9, gaze_hold_ = 0, saccade_t_ = 0;
  Canvas cv_;
  std::string ovl_key_;

  void* sock(void* ctx, int type, const char* ep, const char* topic, bool bind) {
    void* s = zmq_socket(ctx, type);
    int hwm = 20, linger = 0;
    zmq_setsockopt(s, ZMQ_RCVHWM, &hwm, sizeof hwm);
    zmq_setsockopt(s, ZMQ_LINGER, &linger, sizeof linger);
    if (topic) zmq_setsockopt(s, ZMQ_SUBSCRIBE, topic, std::strlen(topic));
    if (bind) {
      std::string path = std::string(ep).substr(6);  // ipc://
      std::remove(path.c_str());                     // stale socket from a crashed run
      if (zmq_bind(s, ep) != 0) note("bind %s: %s", ep, zmq_strerror(zmq_errno()));
    } else {
      zmq_connect(s, ep);
    }
    return s;
  }

  // Reads everything queued; state streams (robot_state, det) only parse their newest message.
  void drain(void* s, int k) {
    std::string last;
    bool have = false;
    for (;;) {
      zmq_msg_t m;
      zmq_msg_init(&m);
      if (zmq_msg_recv(&m, s, ZMQ_DONTWAIT) < 0) {
        zmq_msg_close(&m);
        break;
      }
      bool more = zmq_msg_more(&m);
      std::string part(static_cast<const char*>(zmq_msg_data(&m)), zmq_msg_size(&m));
      zmq_msg_close(&m);
      if (more) continue;  // topic frame
      if (k == 2 || k == 3) {
        last.swap(part), have = true;
      } else {
        handle(k, part);
      }
    }
    if (have) handle(k, last);
  }

  void handle(int k, const std::string& buf) {
    beni::Value v;
    if (!beni::unpack(buf.data(), buf.size(), v) || v.type != beni::Value::MAP) return;
    double now = mono();
    if (k == 0) on_ctrl(v, now);
    else if (k == 1) on_event(v, now);
    else if (k == 2) on_state(v);
    else if (k == 3) on_det(v, now);
    else if (k == 4) mode_ = v.text("mode", mode_);
  }

  void on_ctrl(const beni::Value& v, double now) {
    std::string op = v.text("op");
    if (op == "expression") {
      std::string name = v.text("name");
      double hold = v.num("hold", 4);
      transient_ = name == "auto" ? "" : name;
      transient_until_ = hold > 0 ? now + hold : 1e18;
    } else if (op == "gaze") {
      tx_ = clampf(float(v.num("x")), -1, 1), ty_ = clampf(float(v.num("y")), -1, 1);
      gaze_hold_ = now + v.num("hold", 2);
    } else if (op == "overlay") {
      if (v.has("mic_muted")) mic_muted_ = v.num("mic_muted") != 0;
      if (v.has("cloud_offline")) cloud_off_ = v.num("cloud_offline") != 0;
    } else if (op == "pip") {
      bool on = v.num("on", 1) != 0;
      if (v.has("url")) o_.pip_url = v.text("url");
      if (on && !pip_) start_pip();
      if (!on) stop(pip_);
      pip_on_ = pip_ != nullptr;
    }
  }

  void on_event(const beni::Value& v, double now) {
    std::string name = v.text("name");
    if (name == "voice_state") {
      voice_ = v.text("state", "idle");
      if (voice_ == "listening") transient_.clear();  // a wake always shows attention
    } else if (name == "brain") {
      cloud_off_ = v.num("online") == 0;
    } else if (name == "privacy") {  // §12.3 camera privacy mode: the vision unit is stopped, eyes stay closed
      privacy_ = v.num("on") != 0;
    } else if (name == "mic_muted") {  // the agent's GPIO mute button (the ESP32 switch comes via robot_state)
      mic_muted_ = v.num("muted") != 0;
    } else if (name == "person_seen" && voice_ == "idle" && now >= transient_until_ && now - happy_t_ > 30) {
      transient_ = "happy", transient_until_ = now + 2.5, happy_t_ = now;
    }
  }

  void on_state(const beni::Value& v) {
    const beni::Value* b = v.get("battery");
    battery_ = b && b->type == beni::Value::INT ? int(b->i) : -1;
    charging_ = v.num("charging") != 0;
    if (v.has("mic_muted")) mic_muted_ = v.num("mic_muted") != 0;
  }

  // Largest gie-2 face on the gaze camera -> gaze target (mirrored: the face looks back at the person).
  void on_det(const beni::Value& v, double now) {
    const beni::Value* fr = v.get("frames");
    if (!fr || fr->type != beni::Value::ARR) return;
    float best = 0, fx = 0, fy = 0;
    for (const auto& f : fr->a) {
      if (int(f.num("cam", -1)) != o_.gaze_cam) continue;
      const beni::Value* objs = f.get("objs");
      if (!objs || objs->type != beni::Value::ARR) continue;
      for (const auto& o : objs->a) {
        const beni::Value* bb = o.get("bbox");
        if (int(o.num("gie")) != kFaceGie || !bb || bb->type != beni::Value::ARR || bb->a.size() < 4) continue;
        auto n = [&](int i) { return float(bb->a[i].type == beni::Value::INT ? bb->a[i].i : bb->a[i].f); };
        float area = n(2) * n(3);
        if (area > best) best = area, fx = n(0) + n(2) / 2, fy = n(1) + n(3) / 2;
      }
    }
    if (best <= 0 || now < gaze_hold_) return;
    tx_ = clampf(1 - 2 * fx / kMuxW, -1, 1), ty_ = clampf(2 * fy / kMuxH - 1, -1, 1);
    face_t_ = now;
  }

  static float clampf(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }

  void read_wifi() {
    std::ifstream f("/proc/net/wireless");
    std::string line;
    int n = 0;
    wifi_ = -1;
    while (std::getline(f, line))
      if (++n > 2) {  // " wlan0: 0000   54.  -56.  -256 ..."
        std::istringstream ss(line);
        std::string iface, status;
        float link = 0;
        if (ss >> iface >> status >> link) wifi_ = std::max(wifi_, int(link * 100 / 70));
      }
  }

  // ---------------------------------------------------------------- pipelines
  void start_video() {
    char d[512];
    std::snprintf(d, sizeof d,
                  "appsrc name=src ! h264parse ! nvv4l2decoder enable-max-performance=1 disable-dpb=1 ! "
                  "nvdrmvideosink conn-id=%d plane-id=0 set-mode=1 sync=false",
                  o_.conn);
    video_ = launch(d);
    vsrc_ = video_ ? appsrc_of(video_, "video/x-h264,stream-format=byte-stream,alignment=au") : nullptr;
    cur_ = nullptr;  // restart at an IDR
  }

  void start_overlay() {
    char d[512];
    std::snprintf(d, sizeof d,
                  "appsrc name=src ! nvvidconv ! video/x-raw(memory:NVMM),format=RGBA,width=%d,height=%d ! "
                  "nvdrmvideosink conn-id=%d plane-id=1 set-mode=0 sync=false",
                  kW, kH, o_.conn);
    ovl_ = launch(d);
    char caps[128];
    std::snprintf(caps, sizeof caps, "video/x-raw,format=RGBA,width=%d,height=%d,framerate=0/1", kOW, kOH);
    osrc_ = ovl_ ? appsrc_of(ovl_, caps) : nullptr;
    ovl_key_.clear();
    if (!ovl_) note("overlay plane unavailable: no pupils/icons (retrying in 30 s)");
  }

  void start_pip() {
    char d[768];
    std::snprintf(d, sizeof d,
                  "rtspsrc location=%s latency=100 protocols=tcp ! rtph264depay ! h264parse ! "
                  "nvv4l2decoder enable-max-performance=1 ! nvvidconv ! "
                  "video/x-raw(memory:NVMM),format=NV12,width=320,height=180 ! "
                  "nvdrmvideosink conn-id=%d plane-id=2 offset-x=464 offset-y=284 set-mode=0 sync=false",
                  o_.pip_url.c_str(), o_.conn);
    pip_ = launch(d);
  }

  // ---------------------------------------------------------------- per frame
  const Clip* pick(double now) {
    std::string want;
    if (!transient_.empty() && now < transient_until_) want = transient_;
    else if (voice_ == "listening") want = "listening";
    else if (voice_ == "thinking") want = "thinking";
    else if (voice_ == "speaking" || voice_ == "offline_reply") want = "talking";
    else if (privacy_) want = "closed";
    else if (mode_ == "idle" || mode_ == "critical") want = "sleepy";
    else want = "idle_blink";
    for (const std::string& n : {want, std::string("idle_blink")}) {
      auto it = clips_.find(n);
      if (it != clips_.end()) return it->second.get();
    }
    return clips_.empty() ? nullptr : clips_.begin()->second.get();
  }

  void tick(double now) {
    const Clip* want = pick(now);
    if (want != cur_) cur_ = want, idx_ = 0;  // cross-cut at the new clip's IDR
    if (vsrc_ && cur_ && !cur_->au.empty() && gst_app_src_get_current_level_bytes(vsrc_) < (512u << 10)) {
      const auto& a = cur_->au[idx_];
      GstBuffer* b = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY, const_cast<char*>(cur_->data.data()),
                                                 cur_->data.size(), a.first, a.second, nullptr, nullptr);
      GST_BUFFER_PTS(b) = GST_BUFFER_DTS(b) = GstClockTime(frames_ * GST_SECOND / cur_->fps);
      GST_BUFFER_DURATION(b) = GstClockTime(GST_SECOND / cur_->fps);
      gst_app_src_push_buffer(vsrc_, b);  // clips are never freed, so wrapping without a copy is safe
      ++frames_;
      idx_ = (idx_ + 1) % cur_->au.size();
    }
    if (osrc_) overlay(now);
  }

  void gaze(double now) {
    if (now - face_t_ > 1.5 && now >= gaze_hold_ && now >= saccade_t_) {  // nobody: idle saccades
      std::uniform_real_distribution<float> u(-1, 1), dt(1.5f, 4.f);
      bool centre = u(rng_) < -0.4f;
      tx_ = centre ? 0 : 0.5f * u(rng_), ty_ = centre ? 0 : 0.3f * u(rng_);
      saccade_t_ = now + dt(rng_);
    }
    float k = now - face_t_ < 0.5 ? 0.35f : 0.5f;  // saccades snap, tracking follows smoothly
    gx_ += (tx_ - gx_) * k, gy_ += (ty_ - gy_) * k;
  }

  void overlay(double now) {
    gaze(now);
    bool batt = battery_ >= 0 && (battery_ <= 25 || charging_), wifi = wifi_ >= 0 && wifi_ < 40;
    float open = 0, s = float(kOW) / kW;
    float pl[2] = {0, 0}, pr_[2] = {0, 0};
    if (cur_ && cur_->eyes) {
      open = cur_->open(idx_ >= size_t(o_.lag) ? idx_ - o_.lag : 0);
      float mx = cur_->hw - cur_->pr * 1.1f, my = (cur_->hh - cur_->pr) * 0.6f;
      pl[0] = (cur_->lcx + gx_ * mx) * s, pl[1] = (cur_->lcy + gy_ * my) * s;
      pr_[0] = (cur_->rcx + gx_ * mx) * s, pr_[1] = (cur_->rcy + gy_ * my) * s;
    }
    char key[160];
    std::snprintf(key, sizeof key, "%d,%d,%d,%d,%d|%d,%d,%d,%d,%d,%d", int(pl[0] * 2), int(pl[1] * 2),
                  int(pr_[0] * 2), int(pr_[1] * 2), int(open * 40), batt ? battery_ : -1, charging_, wifi ? wifi_ : -1,
                  mic_muted_, cloud_off_, pip_on_);
    if (ovl_key_ == key) return;  // nothing moved: the DC keeps scanning out the last buffer
    ovl_key_ = key;
    cv_.clear();
    if (open > 0.05f) {
      for (int e = 0; e < 2; ++e) {
        float cx = (e ? cur_->rcx : cur_->lcx) * s, cy = (e ? cur_->rcy : cur_->lcy) * s;
        float clip[4] = {cx, cy, cur_->hw * s, cur_->hh * s * open};
        const float* p = e ? pr_ : pl;
        float r = cur_->pr * s;
        cv_.disc(p[0], p[1], r, kPupil, clip);
        cv_.disc(p[0] - r * 0.35f, p[1] - r * 0.4f, r * 0.28f, kGlint, clip);
      }
    }
    int x = kOW - 8;  // status icons, right to left, only when noteworthy
    if (batt) x = icon_battery(x - 22, 6);
    if (wifi) x = icon_wifi(x - 16, 6);
    if (mic_muted_) x = icon_mic(x - 12, 5);
    if (cloud_off_) x = icon_cloud(x - 20, 6);
    GstBuffer* b = gst_buffer_new_allocate(nullptr, cv_.px.size(), nullptr);
    gst_buffer_fill(b, 0, cv_.px.data(), cv_.px.size());
    GST_BUFFER_PTS(b) = GstClockTime((now - t0_) * GST_SECOND);
    gst_app_src_push_buffer(osrc_, b);
  }

  int icon_battery(int x, int y) {
    Rgba c = battery_ <= 10 ? kRed : battery_ <= 25 ? kAmber : kGreen;
    cv_.frame(x, y, 20, 11, kWhite);
    cv_.rect(x + 20, y + 3, 2, 5, kWhite);
    cv_.rect(x + 2, y + 2, std::max(1, 16 * std::min(battery_, 100) / 100), 7, c);
    if (charging_) {  // bolt
      cv_.line(x + 11, y + 1, x + 8, y + 6, 1.6f, kAmber);
      cv_.line(x + 8, y + 6, x + 12, y + 5, 1.6f, kAmber);
      cv_.line(x + 12, y + 5, x + 9, y + 10, 1.6f, kAmber);
    }
    return x - 6;
  }
  int icon_wifi(int x, int y) {
    int bars = wifi_ <= 0 ? 0 : wifi_ < 20 ? 1 : 2;
    for (int i = 0; i < 3; ++i) cv_.rect(x + i * 5, y + 8 - i * 4, 3, 3 + i * 4, i < bars ? kWhite : kGrey);
    if (wifi_ == 0) cv_.line(x, y, x + 13, y + 11, 1.5f, kRed);
    return x - 6;
  }
  int icon_mic(int x, int y) {
    cv_.line(x + 5, y + 2, x + 5, y + 7, 5, kWhite);
    cv_.line(x + 5, y + 10, x + 5, y + 13, 1.2f, kWhite);
    cv_.line(x, y + 13, x + 11, y, 1.6f, kRed);
    return x - 6;
  }
  int icon_cloud(int x, int y) {
    cv_.disc(x + 6, y + 7, 4, kGrey), cv_.disc(x + 11, y + 5, 5, kGrey), cv_.disc(x + 15, y + 8, 3.5f, kGrey);
    cv_.rect(x + 6, y + 8, 10, 4, kGrey);
    cv_.line(x + 2, y + 12, x + 18, y, 1.6f, kRed);
    return x - 6;
  }
};

}  // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--dir") o.dir = val();
    else if (a == "--conn") o.conn = std::atoi(val().c_str());
    else if (a == "--gaze-cam") o.gaze_cam = std::atoi(val().c_str());
    else if (a == "--lag") o.lag = std::atoi(val().c_str());
    else if (a == "--pip-url") o.pip_url = val();
    else if (a == "--no-overlay") o.overlay = false;
    else {
      std::fprintf(stderr, "usage: beni_face [--dir /ssd/face] [--conn 0] [--gaze-cam 0] [--lag 2] "
                           "[--pip-url rtsp://...] [--no-overlay]\n");
      return 2;
    }
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  mkdir("/tmp/beni", 0775);
  return Face(o).run();
}
