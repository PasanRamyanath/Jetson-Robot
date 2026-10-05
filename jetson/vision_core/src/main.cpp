// §5.5 vision_core: the Phase-3 replacement for beni_vision.py. Same vision.sock / vision_ctrl.sock contract.
// Threads (§5.5.6): cam0, cam1 (capture, recording, teleop, MV feed; CPU 1), infer (detector + faces; CPUs 1-3),
// pub (60 Hz det from the trackers + queued jpeg/motion/scene; CPU 1), ctrl (REP; CPU 1). NVENC/NVJPG/VIC run the
// pixel work; the CPU only handles fds, boxes and bytes.
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <zmq.h>

#include <NvJpegEncoder.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "argus_cam.hpp"
#include "bytetrack.hpp"
#include "decoder.hpp"
#include "detector.hpp"
#include "egl_cuda_map.hpp"
#include "encoder.hpp"
#include "logic.hpp"
#include "messages.hpp"
#include "nvbuf.hpp"
#include "nvtx.hpp"
#include "ts_demux.hpp"
#include "ts_mux.hpp"

namespace vc {
namespace {

std::atomic<bool> g_run{true}, g_fail{false};
void on_signal(int) { g_run = false; }

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void pin(std::initializer_list<int> cpus) {
  cpu_set_t s;
  CPU_ZERO(&s);
  for (int c : cpus) CPU_SET(c, &s);
  pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

struct Args {
  int cams = 2;
  bool infer = true, rec = true, teleop = true, mv = true, reid = true, replay = true, pose = true;
  std::string engines = "/ssd/beni/engines", det = "yolo26n_512x288_b2_fp16.engine", det_fallback =
      "yolov8n_512x288_b2_fp16.engine", scrfd = "scrfd_b4_fp16.engine", embed = "mobilefacenet_b8_fp16.engine",
                                    osnet = "osnet_b4_fp16.engine", trtpose = "trtpose_r18_b4_fp16.engine";
  std::string rec_dir = "/ssd/beni/rec", ir_pwm = "/sys/class/pwm/pwmchip0/pwm0";
  int rec_bps = 3000000, teleop_bps = 2000000, seg_s = 300, ir_max = 80;
  double rec_keep_gb = 60, min_free_gb = 8;
  float det_thresh = 0.1f;  // ByteTrack wants the low-score boxes too (§5.5.6 step 3)
};

bool parse(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--cams") a.cams = std::max(1, std::min(2, std::atoi(val())));
    else if (k == "--no-infer") a.infer = false;
    else if (k == "--no-rec") a.rec = false;
    else if (k == "--no-teleop") a.teleop = false;
    else if (k == "--no-mv") a.mv = false;
    else if (k == "--no-reid") a.reid = false;
    else if (k == "--no-replay") a.replay = false;
    else if (k == "--no-pose") a.pose = false;
    else if (k == "--engines") a.engines = val();
    else if (k == "--det") {  // §6.4: yolo26n | yolov8n [_416] (build_all.sh names) or an engine file name
      a.det = val();
      if (a.det == "yolo26n" || a.det == "yolov8n") a.det += "_512x288_b2_fp16.engine";
      else if (a.det == "yolo26n_416" || a.det == "yolov8n_416") a.det += "_b2_fp16.engine";
    } else if (k == "--rec-dir") a.rec_dir = val();
    else if (k == "--rec-bps") a.rec_bps = std::atoi(val());
    else if (k == "--rec-keep-gb") a.rec_keep_gb = std::atof(val());
    else if (k == "--teleop-bps") a.teleop_bps = std::atoi(val());
    else if (k == "--ir-pwm") a.ir_pwm = val();
    else if (k == "--ir-max") a.ir_max = std::atoi(val());
    else return std::fprintf(stderr, "usage: vision_core [--cams N] [--no-{infer,rec,teleop,mv,reid,replay,pose}] "
                                     "[--engines DIR] [--det yolo26n|yolov8n[_416]|FILE] [--rec-dir DIR] [--rec-bps N] "
                                     "[--rec-keep-gb G] [--teleop-bps N] [--ir-pwm SYSFS] [--ir-max PCT]\n"), false;
  }
  // The DNN stream size follows the --det name, so the fallback must have the same input size.
  if (a.det.find("416") != std::string::npos) a.det_fallback = "yolov8n_416_b2_fp16.engine";
  return true;
}

bool exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

// pgie_yolo26n.txt filter-out-class-ids: vehicles, outdoor and sports classes are dropped at the source.
ClassMask allowed_classes() {
  const int drop[] = {4,  6,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 33, 34, 35,
                      36, 37, 38, 40, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 78, 79};
  std::vector<int> keep;
  for (int c = 0; c < 80; ++c)
    if (std::find(std::begin(drop), std::end(drop), c) == std::end(drop)) keep.push_back(c);
  return ClassMask::only(keep);
}

// ---------------------------------------------------------------- IR LED ring on header pin 32 (§5.5.1)
class IrLed {
 public:
  explicit IrLed(const std::string& dir) : dir_(dir) {
    ok_ = put("period", kPeriod) && put("duty_cycle", 0) && put("enable", 1);
    if (!ok_) std::fprintf(stderr, "[ir] %s not writable (pwm0 exported? see beni-vision-core.service)\n", dir.c_str());
  }
  void set(int pct) {
    pct = std::max(0, std::min(100, pct));
    if (!ok_ || pct == cur_) return;
    cur_ = pct;
    put("duty_cycle", int64_t(kPeriod) * pct / 100);
  }
  int duty() const { return cur_ < 0 ? 0 : cur_; }

 private:
  static constexpr int kPeriod = 50000;  // ns: 20 kHz, inaudible, no camera banding
  bool put(const char* f, int64_t v) {
    FILE* fp = std::fopen((dir_ + "/" + f).c_str(), "w");
    if (!fp) return false;
    bool ok = std::fprintf(fp, "%lld", static_cast<long long>(v)) > 0;
    return std::fclose(fp) == 0 && ok;
  }
  std::string dir_;
  bool ok_ = false;
  int cur_ = -1;
};

// ---------------------------------------------------------------- recording: 5-min .ts segments + size cap
void prune(const std::string& dir, double keep_gb, double min_free_gb) {
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  struct F {
    std::string p;
    time_t m;
    uint64_t s;
  };
  std::vector<F> fs;
  uint64_t total = 0;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.compare(0, 3, "cam") || n.size() < 4 || n.compare(n.size() - 3, 3, ".ts")) continue;
      struct stat st;
      std::string p = dir + "/" + n;
      if (::stat(p.c_str(), &st) == 0) fs.push_back(F{p, st.st_mtime, uint64_t(st.st_size)}), total += st.st_size;
    }
    closedir(d);
  }
  std::sort(fs.begin(), fs.end(), [](const F& a, const F& b) { return a.m < b.m; });
  struct statvfs v;
  double free_gb = statvfs(dir.c_str(), &v) == 0 ? double(v.f_bavail) * v.f_frsize / 1e9 : 1e9;
  for (size_t i = 0; i + 2 < fs.size() && (total / 1e9 > keep_gb || free_gb < min_free_gb); ++i) {
    if (::unlink(fs[i].p.c_str()) == 0) total -= fs[i].s, free_gb += fs[i].s / 1e9;
  }
}

class Recorder {
 public:
  Recorder(const Args& a, int cam) : a_(a), cam_(cam), mux_(true, [this](const uint8_t* p, size_t n) {
                                        if (f_) std::fwrite(p, 1, n, f_);
                                      }) {}
  ~Recorder() {
    if (f_) std::fclose(f_);
  }
  // Encoder capture thread. Rotates on the first keyframe after seg_s (IDR every 2 s at 30 fps).
  void on_au(const uint8_t* au, size_t n, uint64_t ts_us, bool key) {
    key = key || is_keyframe(au, n, true);
    double t = now_s();
    if (key && (!f_ || t - opened_ >= a_.seg_s)) open(t);
    if (f_) mux_.write(au, n, ts_us * 9 / 100, key);
  }

 private:
  void open(double t) {
    if (f_) std::fclose(f_);
    char name[64];
    time_t w = time(nullptr);
    struct tm tmv;
    localtime_r(&w, &tmv);
    std::strftime(name, sizeof name, "%Y%m%d-%H%M%S", &tmv);
    std::string p = a_.rec_dir + "/cam" + std::to_string(cam_) + "_" + name + ".ts";
    f_ = std::fopen(p.c_str(), "wb");
    if (f_) std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
    opened_ = t;
    std::string dir = a_.rec_dir;
    double keep = a_.rec_keep_gb, fr = a_.min_free_gb;
    std::thread([dir, keep, fr] { prune(dir, keep, fr); }).detach();
  }
  const Args& a_;
  int cam_;
  TsMux mux_;
  FILE* f_ = nullptr;
  double opened_ = 0;
};

// ---------------------------------------------------------------- publisher (the only thread touching the PUB socket)
class Publisher {
 public:
  explicit Publisher(void* sock) : sock_(sock) {}
  void post(const char* topic, std::string data) {
    std::lock_guard<std::mutex> g(mu_);
    if (q_.size() < 32) q_.emplace_back(topic, std::move(data));
    cv_.notify_one();
  }
  void wait_until(double t) {
    std::unique_lock<std::mutex> g(mu_);
    double d = t - now_s();
    if (q_.empty() && d > 0) cv_.wait_for(g, std::chrono::duration<double>(d));
  }
  void drain() {
    std::deque<std::pair<const char*, std::string>> q;
    {
      std::lock_guard<std::mutex> g(mu_);
      q.swap(q_);
    }
    for (auto& m : q) send(m.first, m.second);
  }
  void send(const char* topic, const std::string& data) {  // non-blocking: a slow subscriber never stalls vision
    if (zmq_send(sock_, topic, std::strlen(topic), ZMQ_SNDMORE | ZMQ_DONTWAIT) < 0) return;
    zmq_send(sock_, data.data(), data.size(), ZMQ_DONTWAIT);
  }
  size_t backlog() {
    std::lock_guard<std::mutex> g(mu_);
    return q_.size();
  }

 private:
  void* sock_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::pair<const char*, std::string>> q_;
};

// ---------------------------------------------------------------- per-camera state
struct Cam {
  int id = 0;
  ArgusCam argus;
  TripleFd dnn;
  FdPool full;
  std::unique_ptr<Latest> latest;
  std::unique_ptr<Recorder> rec;
  Encoder rec_enc;
  std::atomic<uint64_t> fn{0}, ts{0};
  std::mutex trk_mu;  // tracker, pending faces, last objs
  ByteTracker tracker;
  std::vector<Obj> faces, last_objs;
  FacePacer pacer;
  ReidBank reid;  // guarded by trk_mu
  int rgba_fd = -1;  // full-res RGBA for the face chain (VIC-converted on face ticks only)
  NightLogic night;
  std::atomic<bool> gray{false};
  std::atomic<double> manual_ae_until{0};  // written by ctrl, read by infer
  double auto_ae_at = -1e9, face_seen = -1e9;
  bool auto_ae = false;
  std::atomic<double> pose_until{0};  // §4 item 39: keypoints wanted until (ctrl {op: pose}); read by infer
  double pose_at = -1e9;              // infer thread only
  std::atomic<bool> pause_req{false}, paused{false};  // §15.1 idle-watch: ctrl asks, the cam thread applies
  explicit Cam(int i) : id(i), tracker(ByteTrackParams(), i * 1000000 + 1) {}
};

struct App {
  Args a;
  std::vector<std::unique_ptr<Cam>> cams;
  std::unique_ptr<Publisher> pub;
  std::mutex cad_mu;
  Cadence cadence;
  std::atomic<int> base_interval{3}, ir_override{-1};
  std::atomic<bool> night{false};
  std::mutex frame_mu;
  std::condition_variable frame_cv;
  // teleop + MV
  FdPool tele_pool, mv_pool;
  Encoder tele_enc, mv_enc;
  std::unique_ptr<TsMux> tele_mux;
  int udp = -1;
  sockaddr_in udp_to{};
  MotionGate motion;
  std::unique_ptr<IrLed> ir;
  bool infer_ok = false;
  std::atomic<bool> pose_ok{false};
  class Replayer* replay = nullptr;  // null with --no-replay / --no-infer
  std::atomic<bool> bg_gpu{true};  // §3.17 scheduler admission; assumed true while no scheduler is publishing
  std::atomic<double> sched_t{0};

  bool bg_allowed() const { return bg_gpu || now_s() - sched_t > 10; }

  // §5.3 teleop: finer QP on every head in the side-by-side picture (heads in 1280x720 camera pixels)
  std::mutex roi_mu;
  std::vector<Box> tele_heads[2];
  void set_tele_heads(int cam, std::vector<Box> heads) {
    std::vector<Rect> r;
    {
      std::lock_guard<std::mutex> g(roi_mu);
      tele_heads[cam].swap(heads);
      int n = int(cams.size()), half = n > 1 ? tele_pool.w() / 2 : tele_pool.w();
      float sx = float(half) / 1280, sy = float(tele_pool.h()) / 720;
      for (int i = 0; i < n && i < 2; ++i)
        for (const Box& b : tele_heads[i])
          r.push_back(Rect{int(b.x1 * sx) + i * half, int(b.y1 * sy), int(b.w() * sx), int(b.h() * sy)});
    }
    tele_enc.set_roi(std::move(r));
  }

  void set_cadence() {
    std::lock_guard<std::mutex> g(cad_mu);
    int n = base_interval;
    cadence.set_interval(night && n >= 0 ? std::max(n, 7) : n);  // §5.5.1: 7.5 Hz at night, MV wake-ups on top
  }
};

// ---------------------------------------------------------------- camera thread (§5.5.6 step 1)
void compose_teleop(App& app, int slot0, uint64_t ts_us) {
  int t = app.tele_pool.acquire();
  if (t < 0) return;
  int srcs[2] = {app.cams[0]->full.fd(slot0), -1};
  int s1 = app.cams.size() > 1 ? app.cams[1]->latest->get() : -1;
  NvBufferCompositeParams p;
  std::memset(&p, 0, sizeof p);
  p.composite_flag = NVBUFFER_COMPOSITE | NVBUFFER_COMPOSITE_FILTER;
  p.input_buf_count = s1 >= 0 ? 2 : 1;
  p.session = vic_session();
  int W = app.tele_pool.w(), H = app.tele_pool.h(), half = s1 >= 0 ? W / 2 : W;
  for (uint32_t i = 0; i < p.input_buf_count; ++i) {
    p.composite_filter[i] = NvBufferTransform_Filter_Smart;
    p.src_comp_rect[i].top = 0, p.src_comp_rect[i].left = 0;
    p.src_comp_rect[i].width = 1280, p.src_comp_rect[i].height = 720;
    p.dst_comp_rect[i].top = 0, p.dst_comp_rect[i].left = uint32_t(i * half);
    p.dst_comp_rect[i].width = uint32_t(half), p.dst_comp_rect[i].height = uint32_t(H);
    p.dst_comp_rect_alpha[i] = 1.f;
  }
  if (s1 >= 0) srcs[1] = app.cams[1]->full.fd(s1);
  bool ok = NvBufferComposite(srcs, app.tele_pool.fd(t), &p) == 0;
  if (s1 >= 0) app.cams[1]->full.unref(s1);
  if (!ok || !app.tele_enc.push(app.tele_pool.fd(t), ts_us)) app.tele_pool.unref(t);
}

void cam_loop(App& app, Cam& c) {
  pin({1});
  CamFrame f;
  int fails = 0;
  float ir_duty = 0;
  double scene_at = 0;
  for (uint64_t k = 0; g_run; ++k) {
    if (c.pause_req != c.paused) {
      c.argus.set_paused(c.paused = c.pause_req.load());
      std::fprintf(stderr, "[cam%d] %s\n", c.id, c.paused ? "paused" : "resumed");
    }
    if (c.paused) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    int slot = k % 2 == 0 ? c.full.acquire() : -1;  // 30 fps full-res path from the 60 fps capture
    if (!c.argus.next(f, c.dnn.back(), slot >= 0 ? c.full.fd(slot) : -1)) {
      if (slot >= 0) c.full.unref(slot);
      if (++fails > 5) {
        std::fprintf(stderr, "[cam%d] capture failed; exiting for a clean restart\n", c.id);
        g_fail = true, g_run = false;
      }
      continue;
    }
    fails = 0;
    c.dnn.publish();
    c.fn = f.fn, c.ts = f.ts_ns;
    if (c.id == 0 || app.cams[0]->paused) {  // the infer clock: CAM0, or CAM1 while CAM0 is paused
      std::lock_guard<std::mutex> g(app.frame_mu);
      app.frame_cv.notify_one();
    }
    uint64_t us = f.ts_ns / 1000;
    if (slot >= 0) {
      if (c.rec) {
        c.full.ref(slot);
        if (!c.rec_enc.push(c.full.fd(slot), us)) c.full.unref(slot);
      }
      if (c.id == 0 && app.a.teleop) compose_teleop(app, slot, us);
      if (c.id == 1 && app.a.mv && k % 4 == 0) {  // §5.5.5: 320x180 @ 15 fps, CAM1
        int m = app.mv_pool.acquire();
        if (m >= 0 && !(vic_blit(c.full.fd(slot), app.mv_pool.fd(m)) && app.mv_enc.push(app.mv_pool.fd(m), us)))
          app.mv_pool.unref(m);
      }
      c.latest->set(slot);
    }
    if (c.id == int(app.cams.size()) - 1 && app.cams.size() > 1) {  // NoIR CAM1 drives night mode + IR ring
      int r = c.night.feed(f.lux, f.t);
      if (r) {
        app.night = r > 0, c.gray = r > 0;
        app.set_cadence();
        std::fprintf(stderr, "[scene] night=%d lux=%.1f\n", int(app.night), double(c.night.ema));
      }
      if (r || f.t - scene_at > 10) {
        beni::MsgWriter w;
        write_scene(w, c.id, c.night.night, c.night.ema);
        app.pub->post(beni::topic::kScene, w.buf);
        scene_at = f.t;
      }
      if (app.ir) {
        int ov = app.ir_override;
        float target = float(ov >= 0 ? ov : c.night.night ? app.a.ir_max : 0);
        ir_duty += std::max(-2.f, std::min(2.f, target - ir_duty));  // ramp ~0.7 s at 60 fps
        app.ir->set(int(ir_duty + 0.5f));
      }
    }
  }
}

// ---------------------------------------------------------------- low-priority stream workers (§7.6, §4 item 39)
// OSNet ReID and TRT-Pose share this shape: one job per camera in flight (VIC-blit that camera's full frame, run `Net`
// on person boxes, hand the per-box vectors to `done` on this thread). Runs beside the foreground detector.
template <class Net>
class BgWorker {
 public:
  using Done = std::function<void(Cam&, const std::vector<int>&, std::vector<std::vector<float>>&)>;
  BgWorker(EglCudaMapper& egl, cudaStream_t s, Done done) : egl_(egl), s_(s), done_(std::move(done)) {}
  ~BgWorker() { stop(); }
  bool start(const std::string& engine, size_t ncams) {
    if (!net_.load(engine)) return false;
    for (size_t i = 0; i < ncams && i < 2; ++i)
      rgba_[i] = nvbuf_create(1280, 720, NvBufferColorFormat_ABGR32, NvBufferLayout_Pitch, NvBufferTag_NONE);
    th_ = std::thread(&BgWorker::loop, this);
    return true;
  }
  int max_batch() const { return net_.max_batch(); }
  // Takes over one reference on `slot` of c.full. Dropped (and released) while this camera's last job is running.
  void post(Cam& c, int slot, std::vector<int> ids, std::vector<Box> boxes) {
    std::lock_guard<std::mutex> g(mu_);
    Job& j = jobs_[c.id];
    if (j.c || !run_) {
      c.full.unref(slot);
      return;
    }
    j.c = &c, j.slot = slot, j.ids = std::move(ids), j.boxes = std::move(boxes);
    cv_.notify_one();
  }
  void stop() {
    {
      std::lock_guard<std::mutex> g(mu_);
      run_ = false;
      cv_.notify_one();
    }
    if (th_.joinable()) th_.join();
    for (Job& j : jobs_)
      if (j.c && j.slot >= 0) j.c->full.unref(j.slot), j.c = nullptr;
    for (int& fd : rgba_)
      if (fd >= 0) NvBufferDestroy(fd), fd = -1;
  }

 private:
  struct Job {
    Cam* c = nullptr;
    int slot = -1;
    std::vector<int> ids;
    std::vector<Box> boxes;
  };
  void loop() {
    pin({2, 3});
    std::vector<std::vector<float>> res;
    for (int k = 0;; k ^= 1) {
      Job j;
      {
        std::unique_lock<std::mutex> g(mu_);
        cv_.wait(g, [&] { return !run_ || jobs_[0].c || jobs_[1].c; });
        if (!run_) return;
        if (!jobs_[k].c) k ^= 1;  // alternate cameras when both are waiting
        j = jobs_[k];
      }
      Cam& c = *j.c;
      bool ok = rgba_[k] >= 0 && vic_blit(c.full.fd(j.slot), rgba_[k]);
      c.full.unref(j.slot);
      if (ok && net_.run(egl_.map(rgba_[k]), j.boxes, s_, res)) done_(c, j.ids, res);
      std::lock_guard<std::mutex> g(mu_);
      jobs_[k] = Job();
    }
  }
  EglCudaMapper& egl_;
  cudaStream_t s_;
  Done done_;
  Net net_;
  int rgba_[2] = {-1, -1};
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  Job jobs_[2];
  bool run_ = true;
};

// TRT-Pose crop: the person box grown to a centred square (+10 %) so the 224x224 input keeps the body's aspect.
Box pose_box(const Box& b) {
  float s = 1.1f * std::max(b.w(), b.h()), cx = 0.5f * (b.x1 + b.x2), cy = 0.5f * (b.y1 + b.y2);
  return clip(Box{cx - 0.5f * s, cy - 0.5f * s, cx + 0.5f * s, cy + 0.5f * s}, 1280, 720);
}

// ---------------------------------------------------------------- infer thread (§5.5.6 steps 2-4)
void infer_loop(App& app) {
  pin({1, 2, 3});
  EglCudaMapper egl;
  Streams st;
  Detector det;
  std::string p = app.a.engines + "/" + app.a.det;
  if (!exists(p)) p = app.a.engines + "/" + app.a.det_fallback;
  if (!det.load(p, egl, st.fg)) {
    std::fprintf(stderr, "[infer] no detector (%s): cameras/recording only\n", p.c_str());
    return;
  }
  std::fprintf(stderr, "[infer] detector %s\n", p.c_str());
  FaceChain faces;
  if (!faces.load(app.a.engines + "/" + app.a.scrfd, app.a.engines + "/" + app.a.embed))
    std::fprintf(stderr, "[infer] face chain disabled\n");
  for (auto& c : app.cams) {
    c->rgba_fd = nvbuf_create(1280, 720, NvBufferColorFormat_ABGR32, NvBufferLayout_Pitch, NvBufferTag_NONE);
  }
  std::unique_ptr<BgWorker<ReidNet>> reid;
  if (app.a.reid) {
    reid.reset(new BgWorker<ReidNet>(egl, st.bg, [](Cam& c, const std::vector<int>& ids,
                                                    std::vector<std::vector<float>>& embs) {
      double t = now_s();
      std::lock_guard<std::mutex> g(c.trk_mu);
      for (size_t i = 0; i < embs.size(); ++i)
        if (c.reid.add(ids[i], embs[i].data(), int(embs[i].size()), t))
          std::fprintf(stderr, "[reid] cam%d track %d re-acquired as %d\n", c.id, ids[i], c.reid.alias(ids[i]));
    }));
    if (!reid->start(app.a.engines + "/" + app.a.osnet, app.cams.size())) {
      std::fprintf(stderr, "[infer] ReID disabled (no %s)\n", app.a.osnet.c_str());
      reid.reset();
    }
  }
  std::unique_ptr<BgWorker<PoseNet>> pose;
  if (app.a.pose) {
    App* ap = &app;
    pose.reset(new BgWorker<PoseNet>(egl, st.bg, [ap](Cam& c, const std::vector<int>& ids,
                                                      std::vector<std::vector<float>>& kps) {
      std::vector<Pose> people(kps.size());
      {
        std::lock_guard<std::mutex> g(c.trk_mu);
        for (size_t i = 0; i < kps.size(); ++i) people[i].tid = c.reid.alias(ids[i]);
      }
      for (size_t i = 0; i < kps.size(); ++i) {
        for (size_t k = 0; k + 2 < kps[i].size(); k += 3) kps[i][k] *= 0.4f, kps[i][k + 1] *= 0.4f;  // -> det px
        people[i].kps = std::move(kps[i]);
      }
      beni::MsgWriter w;
      write_pose(w, c.id, int64_t(c.fn.load()), people);
      ap->pub->post(beni::topic::kPose, w.buf);
    }));
    if (!pose->start(app.a.engines + "/" + app.a.trtpose, app.cams.size())) {
      std::fprintf(stderr, "[infer] pose disabled (no %s)\n", app.a.trtpose.c_str());
      pose.reset();
    }
  }
  app.pose_ok = bool(pose);
  app.infer_ok = true;
  const ClassMask allow = allowed_classes();
  bool gray[2] = {false, false};
  uint64_t last_fn = ~0ull;
  std::vector<Det> dets;
  while (g_run) {
    {
      std::unique_lock<std::mutex> g(app.frame_mu);
      app.frame_cv.wait_for(g, std::chrono::milliseconds(50));
    }
    const Cam& clk = app.cams[0]->paused && app.cams.size() > 1 ? *app.cams[1] : *app.cams[0];
    uint64_t fn = clk.fn;
    double t = now_s();
    bool due;
    {
      std::lock_guard<std::mutex> g(app.cad_mu);
      due = fn != last_fn && app.cadence.due(fn, t);
    }
    if (!due) continue;
    last_fn = fn;
    int nb = std::min(det.batch(), int(app.cams.size()));
    NVTX_RANGE("infer_tick");
    for (int b = 0; b < nb; ++b) {
      Cam& c = *app.cams[size_t(b)];
      if (c.gray != gray[b]) det.set_gray(b, gray[b] = c.gray);
      if (!c.paused) vic_blit(c.dnn.take(), det.staging_fd(b));
    }
    if (!det.run(st.fg)) continue;
    for (int b = 0; b < nb; ++b) {
      Cam& c = *app.cams[size_t(b)];
      if (c.paused) continue;
      dets.clear();
      det.decode(b, app.a.det_thresh, allow, dets);
      std::vector<std::pair<int64_t, Box>> cands;
      std::vector<int> held, people, rids;
      std::vector<Box> bodies, rboxes, tele_heads;
      int base = app.base_interval;
      // §15.1: background tier only in active daytime mode, and only while the scheduler admits it (§3.17)
      bool reid_on = reid && base >= 0 && base <= 3 && !app.night && app.bg_allowed();
      {
        NVTX_RANGE("track");
        std::lock_guard<std::mutex> g(c.trk_mu);
        c.tracker.predict(t);
        c.tracker.update(t, dets);
        c.tracker.each_active([&](const Track& tr) {
          if (tr.cls != 0) return;
          const Box& b = tr.box();
          Box full = clip(Box{b.x1 * 2.5f, b.y1 * 2.5f, b.x2 * 2.5f, b.y2 * 2.5f}, 1280, 720);
          Box h = head_crop(full, 1280, 720);
          if (h.w() >= 64 && h.h() >= 64) cands.emplace_back(tr.id, h);  // smaller heads can't reach 80 px faces
          if (h.w() >= 24) tele_heads.push_back(h);
          if (full.h() >= 96 && full.w() >= 32) people.push_back(tr.id), bodies.push_back(full);
        });
        for (const Track& tr : c.tracker.tracks()) held.push_back(tr.id);
        c.reid.sync(held, t);
        if (reid_on)
          for (int id : c.reid.due(people, t, reid->max_batch()))
            rids.push_back(id), rboxes.push_back(bodies[size_t(std::find(people.begin(), people.end(), id) -
                                                               people.begin())]);
      }
      if (app.a.teleop) app.set_tele_heads(c.id, std::move(tele_heads));
      if (!rids.empty()) {
        int slot = c.latest->get();
        if (slot >= 0) reid->post(c, slot, std::move(rids), std::move(rboxes));
      }
      // §4 item 39: on demand only (the agent asks around a turn or an arrival), 5 Hz, largest people first
      if (pose && !people.empty() && t < c.pose_until && t - c.pose_at >= 0.2 && app.bg_allowed()) {
        std::vector<size_t> ord(people.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return bodies[a].area() > bodies[b].area(); });
        ord.resize(std::min(ord.size(), size_t(pose->max_batch())));
        std::vector<int> pids;
        std::vector<Box> pboxes;
        for (size_t i : ord) pids.push_back(people[i]), pboxes.push_back(pose_box(bodies[i]));
        int slot = c.latest->get();
        if (slot >= 0) pose->post(c, slot, std::move(pids), std::move(pboxes)), c.pose_at = t;
      }
      if (!faces.ok() || cands.empty()) continue;
      std::vector<int> pick = c.pacer.pick(cands, t, faces.max_batch());
      c.pacer.gc(t);
      if (pick.empty()) continue;
      int slot = c.latest->get();
      if (slot < 0) continue;
      bool blit = vic_blit(c.full.fd(slot), c.rgba_fd);
      c.full.unref(slot);
      if (!blit) continue;
      std::vector<std::pair<int, Box>> heads;
      for (int i : pick) heads.emplace_back(int(cands[size_t(i)].first), cands[size_t(i)].second);
      std::vector<FaceResult> res;
      {
        NVTX_RANGE("faces");
        faces.run(egl.map(c.rgba_fd), heads, st.fg, res);
      }
      for (auto& h : heads) c.pacer.done(h.first, t);
      std::vector<Obj> objs;
      const Face* best = nullptr;
      for (auto& r : res) {
        Obj o;
        o.tid = -1, o.cls = 0, o.gie = kGieFace, o.parent = r.parent, o.conf = r.face.score;
        o.b = Box{r.face.b.x1 * 0.4f, r.face.b.y1 * 0.4f, r.face.b.x2 * 0.4f, r.face.b.y2 * 0.4f};
        o.emb.resize(r.emb.size() * 2);
        for (size_t i = 0; i < r.emb.size(); ++i) {
          uint16_t h = f2h(r.emb[i]);
          std::memcpy(&o.emb[2 * i], &h, 2);
        }
        objs.push_back(std::move(o));
        if (!best || r.face.b.area() > best->b.area()) best = &r.face;
      }
      {
        std::lock_guard<std::mutex> g(c.trk_mu);
        for (auto& o : objs) o.parent = c.reid.alias(o.parent), c.faces.push_back(std::move(o));
      }
      if (c.id == 0 && t >= c.manual_ae_until) {  // §5.5.1: expose for the primary face, 2 Hz at most
        if (best && t - c.auto_ae_at >= 0.5) {
          const Box& fb = best->b;
          float mx = 0.25f * fb.w(), my = 0.25f * fb.h();
          c.argus.set_ae_region((fb.x1 - mx) / 1280, (fb.y1 - my) / 720, (fb.x2 + mx) / 1280, (fb.y2 + my) / 720);
          c.auto_ae = true, c.auto_ae_at = t;
        }
        if (best) c.face_seen = t;
      }
    }
    Cam& c0 = *app.cams[0];
    if (c0.auto_ae && t - c0.face_seen > 3 && t >= c0.manual_ae_until) c0.argus.set_ae_region(0, 0, 0, 0),
        c0.auto_ae = false;
  }
  app.pose_ok = false;
  if (pose) pose->stop();
  if (reid) reid->stop();
  for (auto& c : app.cams)
    if (c->rgba_fd >= 0) NvBufferDestroy(c->rgba_fd), c->rgba_fd = -1;
}

// ---------------------------------------------------------------- pub thread: det at 60 Hz (§5.5.6 steps 3 and 6)
void pub_loop(App& app) {
  pin({1});
  beni::MsgWriter w(4096);
  std::vector<Frame> frames(app.cams.size());
  double next = now_s();
  while (g_run) {
    app.pub->wait_until(next);
    app.pub->drain();
    double t = now_s();
    if (t < next) continue;
    next = std::max(next + 1.0 / 60, t - 0.05);
    for (size_t i = 0; i < app.cams.size(); ++i) {
      Cam& c = *app.cams[i];
      Frame& f = frames[i];
      f.cam = c.id, f.fn = int64_t(c.fn.load()), f.ts = int64_t(c.ts.load());
      f.objs.clear();
      std::lock_guard<std::mutex> g(c.trk_mu);
      if (c.paused) {  // no frames: don't extrapolate stale tracks
        c.last_objs.clear();
        continue;
      }
      c.tracker.predict(t);
      c.tracker.each_active([&](const Track& tr) {
        Obj o;
        o.tid = c.reid.alias(tr.id), o.cls = tr.cls,
        o.gie = kGiePerson, o.conf = tr.score, o.b = clip(tr.box(), kDetW, kDetH);
        if (o.b.area() >= 1.f) f.objs.push_back(o);
      });
      c.last_objs = f.objs;
      for (auto& o : c.faces) f.objs.push_back(std::move(o));  // each embedding is published exactly once
      c.faces.clear();
    }
    if (write_det(w, frames)) app.pub->send(beni::topic::kDet, w.buf);
  }
}

// ---------------------------------------------------------------- sleep replay (\u00a711.9 step 9, \u00a715.1)
// {op: replay, path} queues recording segments. One thread demuxes each (TsDemux), decodes it on NVDEC, samples every
// `stride`-th frame into its own detector + face chain on the low-priority stream (engines load on the first job and
// are freed after a minute idle), with ByteTrack + FacePacer over media time. Results go out on the `replay` topic;
// the agent names the faces and clusters strangers. Pauses while the scheduler withholds the background tier and
// whenever the publisher has a backlog, so the live topics never queue behind it.
class Replayer {
 public:
  explicit Replayer(App& app) : app_(app), th_(&Replayer::loop, this) {}
  ~Replayer() {
    {
      std::lock_guard<std::mutex> g(mu_);
      run_ = false;
      cv_.notify_one();
    }
    th_.join();
  }
  void queue(const std::string& path, int stride) {
    std::lock_guard<std::mutex> g(mu_);
    if (q_.size() < 256) q_.emplace_back(path, stride);
    cv_.notify_one();
  }
  void stop() {
    std::lock_guard<std::mutex> g(mu_);
    q_.clear();
    abort_ = true;
  }
  int pending() {
    std::lock_guard<std::mutex> g(mu_);
    return int(q_.size()) + int(busy_);
  }

 private:
  struct Slot {
    int rgba = -1, small = -1;  // 1280x720 RGBA for the face chain; 320x180 YUV420 for the keyframe jpeg
    double pts = 0;
    int64_t fn = 0;
  };
  bool load() {
    if (det_) return true;
    egl_.reset(new EglCudaMapper());
    det_.reset(new Detector());
    std::string p = app_.a.engines + "/" + app_.a.det;
    if (!exists(p)) p = app_.a.engines + "/" + app_.a.det_fallback;
    if (!det_->load(p, *egl_, st_.bg)) return unload(), false;
    faces_.reset(new FaceChain());
    if (!faces_->load(app_.a.engines + "/" + app_.a.scrfd, app_.a.engines + "/" + app_.a.embed)) faces_.reset();
    slots_.resize(size_t(det_->batch()));
    for (Slot& s : slots_) {
      s.rgba = nvbuf_create(1280, 720, NvBufferColorFormat_ABGR32, NvBufferLayout_Pitch, NvBufferTag_NONE);
      s.small = nvbuf_create(320, 180, NvBufferColorFormat_YUV420, NvBufferLayout_Pitch, NvBufferTag_NONE);
    }
    jpg_.reset(NvJPEGEncoder::createJPEGEncoder("replayjpg"));
    std::fprintf(stderr, "[replay] engines loaded (batch %d, faces %d)\n", det_->batch(), int(bool(faces_)));
    return true;
  }
  void unload() {
    faces_.reset(), jpg_.reset();
    if (egl_) egl_->clear();  // unregister the EGL images before their NvBuffers (det_'s staging fds) go away
    det_.reset();
    for (Slot& s : slots_) {
      if (s.rgba >= 0) NvBufferDestroy(s.rgba);
      if (s.small >= 0) NvBufferDestroy(s.small);
    }
    slots_.clear(), egl_.reset();
  }
  void loop() {
    pin({2, 3});
    double idle_since = now_s();
    for (;;) {
      std::pair<std::string, int> job;
      {
        std::unique_lock<std::mutex> g(mu_);
        cv_.wait_for(g, std::chrono::seconds(5), [&] { return !run_ || !q_.empty(); });
        if (!run_ || !g_run) break;
        if (q_.empty()) {
          if (det_ && now_s() - idle_since > 60) unload(), std::fprintf(stderr, "[replay] idle: engines freed\n");
          continue;
        }
        job = q_.front();
        q_.pop_front();
        busy_ = true, abort_ = false;
      }
      frames_ = 0;
      bool ok = load() && segment(job.first, job.second);
      beni::MsgWriter w;
      write_replay_done(w, job.first, cam_of(job.first), frames_, ok);
      app_.pub->post(beni::topic::kReplay, w.buf);
      std::lock_guard<std::mutex> g(mu_);
      busy_ = false, idle_since = now_s();
    }
    unload();
  }
  static int cam_of(const std::string& path) {  // Recorder names segments <rec_dir>/cam<N>_<time>.ts
    size_t i = path.rfind("/cam");
    return i != std::string::npos && i + 4 < path.size() && path[i + 4] == '1' ? 1 : 0;
  }
  bool stopping() {
    std::lock_guard<std::mutex> g(mu_);
    return abort_ || !run_ || !g_run;
  }
  bool segment(const std::string& path, int stride) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    path_ = path, cam_ = cam_of(path), used_ = 0, stride_ = stride, pts0_ = -1;
    tracker_.reset();
    pacer_ = FacePacer();
    shot_.clear();
    for (int b = 0; b < det_->batch(); ++b) det_->set_gray(b, false);
    Decoder dec;
    bool opened = false, ok = true;
    TsDemux dm([&](const uint8_t* au, size_t n, uint64_t pts) {
      if (!opened) opened = dec.open(dm.hevc(), [this](int fd, int, int, uint64_t p) { on_frame(fd, p); });
      if (opened && ok) ok = dec.push(au, n, pts);
    });
    std::vector<uint8_t> buf(1 << 20);
    size_t n;
    while (ok && (n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
      dm.feed(buf.data(), n);
      while (!stopping() && (!app_.bg_allowed() || app_.pub->backlog() > 8))
        std::this_thread::sleep_for(std::chrono::milliseconds(200));  // yield to the live tier and the publisher
      if (stopping()) ok = false;
    }
    std::fclose(f);
    if (ok) dm.flush();
    if (opened) dec.finish();
    if (used_) process();
    dec.close();
    std::fprintf(stderr, "[replay] %s: %lld frames%s\n", path.c_str(), (long long)frames_, ok ? "" : " (stopped)");
    return ok && opened;
  }
  void on_frame(int fd, uint64_t pts90k) {  // decoder thread context (inside push/finish); fd valid until return
    int64_t k = frames_++;
    if (k % stride_) return;
    if (pts0_ < 0) pts0_ = double(pts90k) / 90000;
    Slot& s = slots_[size_t(used_)];
    if (!vic_blit(fd, det_->staging_fd(used_)) || !vic_blit(fd, s.rgba) || !vic_blit(fd, s.small)) return;
    s.pts = double(pts90k) / 90000 - pts0_, s.fn = k;
    if (++used_ == det_->batch()) process();
  }
  void process() {
    int n = used_;
    used_ = 0;
    NVTX_RANGE("replay_batch");
    if (!det_->run(st_.bg)) return;
    beni::MsgWriter w;
    for (int b = 0; b < n; ++b) {
      Slot& s = slots_[size_t(b)];
      dets_.clear();
      det_->decode(b, app_.a.det_thresh, allow_, dets_);
      tracker_.predict(s.pts);
      tracker_.update(s.pts, dets_);
      std::vector<Obj> objs;
      std::vector<std::pair<int64_t, Box>> cands;
      tracker_.each_active([&](const Track& tr) {
        Obj o;
        o.tid = tr.id, o.cls = tr.cls, o.gie = kGiePerson, o.conf = tr.score, o.b = clip(tr.box(), kDetW, kDetH);
        objs.push_back(o);
        if (tr.cls != 0) return;
        const Box& bb = o.b;
        Box h = head_crop(clip(Box{bb.x1 * 2.5f, bb.y1 * 2.5f, bb.x2 * 2.5f, bb.y2 * 2.5f}, 1280, 720), 1280, 720);
        if (h.w() >= 64 && h.h() >= 64) cands.emplace_back(tr.id, h);
      });
      std::string jpeg;
      if (faces_ && faces_->ok() && !cands.empty()) {
        std::vector<int> pick = pacer_.pick(cands, s.pts, faces_->max_batch());
        pacer_.gc(s.pts);
        std::vector<std::pair<int, Box>> heads;
        for (int i : pick) heads.emplace_back(int(cands[size_t(i)].first), cands[size_t(i)].second);
        std::vector<FaceResult> res;
        if (!heads.empty()) faces_->run(egl_->map(s.rgba), heads, st_.bg, res);
        for (auto& h : heads) pacer_.done(h.first, s.pts);
        bool fresh = false;
        for (auto& r : res) {
          Obj o;
          o.tid = -1, o.cls = 0, o.gie = kGieFace, o.parent = r.parent, o.conf = r.face.score;
          o.b = Box{r.face.b.x1 * 0.4f, r.face.b.y1 * 0.4f, r.face.b.x2 * 0.4f, r.face.b.y2 * 0.4f};
          o.emb.resize(r.emb.size() * 2);
          for (size_t i = 0; i < r.emb.size(); ++i) {
            uint16_t hv = f2h(r.emb[i]);
            std::memcpy(&o.emb[2 * i], &hv, 2);
          }
          objs.push_back(std::move(o));
          fresh |= shot_.insert(r.parent).second;
        }
        if (fresh && jpg_) {
          unsigned char* p = nullptr;  // libjpeg mallocs; ~12 KB, once per track
          unsigned long sz = 0;
          if (jpg_->encodeFromFd(s.small, JCS_YCbCr, &p, sz, 80) == 0 && p) jpeg.assign((const char*)p, sz);
          std::free(p);
        }
      }
      if (objs.empty()) continue;
      write_replay(w, path_, cam_, s.pts, s.fn, objs, jpeg);
      app_.pub->post(beni::topic::kReplay, w.buf);
    }
  }
  App& app_;
  Streams st_;
  std::unique_ptr<EglCudaMapper> egl_;
  std::unique_ptr<Detector> det_;
  std::unique_ptr<FaceChain> faces_;
  std::unique_ptr<NvJPEGEncoder> jpg_;
  std::vector<Slot> slots_;
  const ClassMask allow_ = allowed_classes();
  ByteTracker tracker_{ByteTrackParams(), 1};
  FacePacer pacer_;
  std::set<int> shot_;
  std::vector<Det> dets_;
  std::string path_;
  int cam_ = 0, used_ = 0, stride_ = 6;
  int64_t frames_ = 0;
  double pts0_ = -1;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::pair<std::string, int>> q_;
  bool run_ = true, busy_ = false, abort_ = false;
  std::thread th_;  // last: the worker starts once everything above exists
};

// ---------------------------------------------------------------- ctrl thread (REP)
// \u00a73.17: follow the scheduler's background-tier admission (sched.sock).
void sched_loop(App& app, void* sub) {
  std::vector<char> topic(16), in(4096);
  while (g_run) {
    if (zmq_recv(sub, topic.data(), topic.size(), 0) < 0) continue;  // RCVTIMEO: re-check g_run
    int more = 0;
    size_t sz = sizeof more;
    zmq_getsockopt(sub, ZMQ_RCVMORE, &more, &sz);
    if (!more) continue;
    int n = zmq_recv(sub, in.data(), in.size(), 0);
    bool ok = false;
    if (n > 0 && n <= int(in.size()) && parse_sched(in.data(), size_t(n), ok)) app.bg_gpu = ok, app.sched_t = now_s();
  }
}

// A snapshot of a paused camera (§15.1 idle-watch) resumes it and waits for fresh frames (Argus restarts in ~0.2 s).
void wake(Cam& c) {
  c.pause_req = false;
  uint64_t f0 = c.fn;
  for (int i = 0; i < 60 && (c.paused || c.fn < f0 + 4); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
}

void ctrl_loop(App& app, void* rep) {
  pin({1});
  std::unique_ptr<NvJPEGEncoder> jpg(NvJPEGEncoder::createJPEGEncoder("snap"));
  const int SW = 1024, SH = 576;  // same as the DeepStream path
  int snap_fd = nvbuf_create(SW, SH, NvBufferColorFormat_YUV420, NvBufferLayout_Pitch, NvBufferTag_NONE);
  std::vector<unsigned char> jbuf(size_t(SW) * SH * 3 / 2);
  std::map<std::pair<int, int>, int> thumb_fd;  // by output size; a handful at most
  beni::MsgWriter w, out(1 << 18);
  std::vector<char> in(1 << 16);
  while (g_run) {
    int n = zmq_recv(rep, in.data(), in.size(), 0);
    if (n < 0) continue;  // RCVTIMEO: re-check g_run
    Cmd cmd = parse_cmd(in.data(), size_t(std::min<int>(n, int(in.size()))));
    std::string err = cmd.err;
    w.clear();
    bool ok = err.empty();
    Cam* c = cmd.cam < int(app.cams.size()) ? app.cams[size_t(cmd.cam)].get() : nullptr;
    if (ok && !c && cmd.op != Cmd::PING) ok = false, err = "camera not running";
    if (ok) switch (cmd.op) {
        case Cmd::PING:
          break;
        case Cmd::SNAPSHOT:
        case Cmd::THUMB: {
          bool thumb = cmd.op == Cmd::THUMB;
          ThumbGeom tg = thumb ? thumb_geom(cmd.l, cmd.t, cmd.r, cmd.b, 1280, 720, cmd.n)
                               : ThumbGeom{0, 0, 1280, 720, SW, SH};
          int dst = snap_fd;
          if (thumb) {
            auto it = thumb_fd.find(std::make_pair(tg.w, tg.h));
            if (it == thumb_fd.end()) {
              if (thumb_fd.size() >= 4) {
                for (auto& kv : thumb_fd) NvBufferDestroy(kv.second);
                thumb_fd.clear();
              }
              int fd = nvbuf_create(tg.w, tg.h, NvBufferColorFormat_YUV420, NvBufferLayout_Pitch, NvBufferTag_NONE);
              if (fd >= 0) it = thumb_fd.emplace(std::make_pair(tg.w, tg.h), fd).first;
            }
            dst = it == thumb_fd.end() ? -1 : it->second;
          }
          if (c->paused || c->pause_req) wake(*c);
          int slot = c->latest->get();
          ok = slot >= 0 && dst >= 0 && jpg;
          if (ok) ok = vic_crop(c->full.fd(slot), dst, tg.x, tg.y, tg.cw, tg.ch);
          if (slot >= 0) c->full.unref(slot);
          unsigned char* p = jbuf.data();
          unsigned long sz = jbuf.size();
          if (ok) ok = jpg->encodeFromFd(dst, JCS_YCbCr, &p, sz, thumb ? 80 : 85) == 0;
          if (ok) {
            std::vector<Obj> objs;
            if (!thumb) {
              std::lock_guard<std::mutex> g(c->trk_mu);
              objs = c->last_objs;
            }
            write_jpeg(out, cmd.cam, &cmd.req, p, sz, objs);
            app.pub->post(beni::topic::kJpeg, out.buf);
          } else {
            err = "snapshot failed";
          }
          if (p != jbuf.data()) std::free(p);  // jpeg_mem_dest outgrew our buffer and malloc'd its own
          break;
        }
        case Cmd::BITRATE:
          if (app.a.teleop) app.tele_enc.set_bitrate(cmd.n);
          else ok = false, err = "teleop off";
          break;
        case Cmd::REPLAY:
          if (!app.replay) {
            ok = false, err = "replay off";
          } else if (cmd.r != 0) {
            app.replay->stop();
          } else if (!cmd.path.empty()) {  // only finished recordings: nothing outside rec_dir, no traversal
            const std::string& p = cmd.path;
            bool safe = p.compare(0, app.a.rec_dir.size() + 1, app.a.rec_dir + "/") == 0 &&
                        p.find("..") == std::string::npos && p.size() > 3 && p.compare(p.size() - 3, 3, ".ts") == 0;
            if (safe) app.replay->queue(p, cmd.n);
            else ok = false, err = "not a recording";
          }
          break;
        case Cmd::POSE:
          if (app.pose_ok) c->pose_until = cmd.r > 0 ? now_s() + cmd.r : 0;
          else ok = false, err = "pose off";
          break;
        case Cmd::IDR:
          app.cams[0]->pause_req = false;  // a viewer joined: teleop shows CAM0
          if (app.a.teleop) app.tele_enc.force_idr();
          else ok = false, err = "teleop off";
          break;
        case Cmd::INTERVAL:
          app.base_interval = cmd.n;
          app.set_cadence();
          break;
        case Cmd::AE_REGION:
          c->argus.set_ae_region(cmd.l, cmd.t, cmd.r, cmd.b);
          c->manual_ae_until = cmd.r > cmd.l ? now_s() + 10 : 0;
          break;
        case Cmd::IR:
          app.ir_override = std::min(cmd.n, 100);
          break;
        case Cmd::CAM:
          if (cmd.n == 0 && app.cams.size() < 2) ok = false, err = "only camera";
          else c->pause_req = cmd.n == 0;
          break;
        default:
          ok = false, err = "bad op";
      }
    if (ok) w.map(cmd.op == Cmd::PING ? 3 : cmd.op == Cmd::REPLAY ? 2 : 1).key("ok").b(true);
    else w.map(2).key("ok").b(false).key("err").str(err);
    if (ok && cmd.op == Cmd::PING) w.key("infer").b(app.infer_ok).key("core").b(true);
    if (ok && cmd.op == Cmd::REPLAY) w.key("pending").i(app.replay->pending());
    zmq_send(rep, w.buf.data(), w.buf.size(), 0);
  }
  if (snap_fd >= 0) NvBufferDestroy(snap_fd);
  for (auto& kv : thumb_fd) NvBufferDestroy(kv.second);
}

int run(int argc, char** argv) {
  App app;
  if (!parse(argc, argv, app.a)) return 2;
  std::signal(SIGINT, on_signal), std::signal(SIGTERM, on_signal);
  ::mkdir("/tmp/beni", 0775);
  if (app.a.rec) ::mkdir(app.a.rec_dir.c_str(), 0775);

  void* ctx = zmq_ctx_new();
  zmq_ctx_set(ctx, ZMQ_IO_THREADS, 1);
  void* pub = zmq_socket(ctx, ZMQ_PUB);
  void* rep = zmq_socket(ctx, ZMQ_REP);
  int hwm = 20, to = 200;
  zmq_setsockopt(pub, ZMQ_SNDHWM, &hwm, sizeof hwm);
  zmq_setsockopt(rep, ZMQ_RCVTIMEO, &to, sizeof to);
  if (zmq_bind(pub, beni::ep::kVision) || zmq_bind(rep, beni::ep::kVisionCtrl))
    return std::fprintf(stderr, "zmq bind: %s\n", zmq_strerror(zmq_errno())), 1;
  app.pub.reset(new Publisher(pub));
  void* sub = zmq_socket(ctx, ZMQ_SUB);
  zmq_setsockopt(sub, ZMQ_RCVTIMEO, &to, sizeof to);
  zmq_setsockopt(sub, ZMQ_SUBSCRIBE, "sched", 5);
  zmq_connect(sub, beni::ep::kSched);  // ipc connect succeeds before the scheduler binds

  for (int i = 0; i < app.a.cams; ++i) {
    std::unique_ptr<Cam> c(new Cam(i));
    CamCfg cc;
    cc.sensor = i, cc.noir = i == 1;
    // DNN stream size = detector input: 512x288 by default; the 416x224 engine works by renaming --det only if
    // the ISP stream matches, so keep them in step here.
    if (app.a.det.find("416") != std::string::npos) cc.dnn_w = 416, cc.dnn_h = 224;
    if (!c->dnn.create(cc.dnn_w, cc.dnn_h, NvBufferColorFormat_ABGR32) ||
        !c->full.create(12, 1280, 720, NvBufferColorFormat_YUV420, NvBufferLayout_BlockLinear, NvBufferTag_CAMERA))
      return std::fprintf(stderr, "[cam%d] NvBuffer allocation failed\n", i), 1;
    c->latest.reset(new Latest(c->full));
    if (!c->argus.open(cc)) return 1;
    if (app.a.rec) {
      c->rec.reset(new Recorder(app.a, i));
      Cam* cp = c.get();
      EncCfg e;
      e.name = i ? "rec1" : "rec0", e.bps = app.a.rec_bps;
      auto au = [cp](const uint8_t* p, size_t n, uint64_t us, bool key) { cp->rec->on_au(p, n, us, key); };
      if (!c->rec_enc.open(e, au, nullptr, [cp](int fd) { cp->full.unref(cp->full.slot_of(fd)); }))
        return 1;
    }
    app.cams.push_back(std::move(c));
  }
  if (app.a.teleop) {  // §5.5.6 step 5: both cameras side by side, H.264 -> MPEG-TS -> MediaMTX (udp:5000)
    int W = app.cams.size() > 1 ? 1280 : 960, H = app.cams.size() > 1 ? 360 : 540;
    if (!app.tele_pool.create(6, W, H, NvBufferColorFormat_YUV420, NvBufferLayout_BlockLinear, NvBufferTag_VIDEO_ENC))
      return 1;
    app.udp = socket(AF_INET, SOCK_DGRAM, 0);
    app.udp_to.sin_family = AF_INET, app.udp_to.sin_port = htons(5000);
    app.udp_to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    App* ap = &app;
    app.tele_mux.reset(new TsMux(false, [ap](const uint8_t* p, size_t n) {
      sendto(ap->udp, p, n, MSG_DONTWAIT, reinterpret_cast<const sockaddr*>(&ap->udp_to), sizeof ap->udp_to);
    }));
    EncCfg e;
    e.name = "teleop", e.hevc = false, e.w = W, e.h = H, e.bps = app.a.teleop_bps, e.idr = 30, e.roi_dqp = -6;
    if (!app.tele_enc.open(e, [ap](const uint8_t* p, size_t n, uint64_t us, bool key) {
          ap->tele_mux->write(p, n, us * 9 / 100, key || is_keyframe(p, n, false));
        }, nullptr, [ap](int fd) { ap->tele_pool.unref(ap->tele_pool.slot_of(fd)); }))
      return 1;
  }
  if (app.a.mv && app.cams.size() > 1) {  // §5.5.5
    if (!app.mv_pool.create(6, 320, 180, NvBufferColorFormat_YUV420, NvBufferLayout_Pitch, NvBufferTag_VIDEO_ENC))
      return 1;
    App* ap = &app;
    EncCfg e;
    e.name = "mv", e.hevc = false, e.w = 320, e.h = 180, e.fps = 15, e.bps = 200000, e.idr = 150, e.mv = true;
    if (!app.mv_enc.open(e, nullptr,
                         [ap](const float* mag, int n, uint64_t) {
                           double t = now_s();
                           if (!ap->motion.feed(mag, n, t)) return;
                           {
                             std::lock_guard<std::mutex> g(ap->cad_mu);
                             ap->cadence.motion(t);  // detector back to 15 Hz within one period
                           }
                           beni::MsgWriter w;
                           write_motion(w, 1, ap->motion.last_score);
                           ap->pub->post(beni::topic::kMotion, w.buf);
                         },
                         [ap](int fd) { ap->mv_pool.unref(ap->mv_pool.slot_of(fd)); }))
      app.a.mv = false;
  }
  if (app.cams.size() > 1) app.ir.reset(new IrLed(app.a.ir_pwm));
  app.set_cadence();

  std::unique_ptr<Replayer> replayer;
  if (app.a.infer && app.a.replay) replayer.reset(new Replayer(app)), app.replay = replayer.get();

  std::vector<std::thread> th;
  for (auto& c : app.cams) th.emplace_back(cam_loop, std::ref(app), std::ref(*c));
  if (app.a.infer) th.emplace_back(infer_loop, std::ref(app));
  th.emplace_back(pub_loop, std::ref(app));
  th.emplace_back(ctrl_loop, std::ref(app), rep);
  th.emplace_back(sched_loop, std::ref(app), sub);
  std::fprintf(stderr, "vision_core: %d cam(s), infer=%d rec=%d teleop=%d mv=%d\n", app.a.cams, int(app.a.infer),
               int(app.a.rec), int(app.a.teleop), int(app.a.mv));
  for (auto& t : th) t.join();
  app.replay = nullptr, replayer.reset();

  for (auto& c : app.cams) c->rec_enc.close();  // flushes the open segment; returns every in-flight fd
  app.tele_enc.close(), app.mv_enc.close();
  if (app.ir) app.ir->set(0);
  for (auto& c : app.cams) c->latest->set(-1), c->argus.close();
  if (app.udp >= 0) ::close(app.udp);
  zmq_close(pub), zmq_close(rep), zmq_close(sub);
  zmq_ctx_term(ctx);
  return g_fail ? 1 : 0;
}

}  // namespace
}  // namespace vc

int main(int argc, char** argv) { return vc::run(argc, argv); }
