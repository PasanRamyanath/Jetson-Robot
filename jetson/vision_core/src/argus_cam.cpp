// §5.5.1 Argus capture (L4T 32.7 libargus): 720p60 sensor mode, exposure cap, 50 Hz antibanding, NoIR manual AWB,
// AE regions and per-frame metadata -> lux proxy. Follows MMAPI samples 13_multi_camera / 10_camera_recording.
#include "argus_cam.hpp"

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>

#include "logic.hpp"

using namespace Argus;
using namespace EGLStream;

namespace vc {
namespace {

CameraProvider* provider() {  // one provider per process (nvargus-daemon client)
  static UniqueObj<CameraProvider> p(CameraProvider::create());
  return p.get();
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

ICaptureMetadata* meta_of(const UniqueObj<Frame>& f) {
  auto* am = interface_cast<IArgusCaptureMetadata>(f);
  return am ? interface_cast<ICaptureMetadata>(am->getMetadata()) : nullptr;
}

constexpr uint64_t kTimeoutNs = 1000000000ull;

}  // namespace

struct ArgusCam::Impl {
  CamCfg c;
  UniqueObj<CaptureSession> session;
  UniqueObj<OutputStream> full, dnn;
  UniqueObj<FrameConsumer> cfull, cdnn;
  UniqueObj<Request> req;
  ICaptureSession* is = nullptr;
  IAutoControlSettings* ac = nullptr;
  IFrameConsumer *ifull = nullptr, *idnn = nullptr;
  std::mutex mu;
  bool ae_dirty = false;
  std::vector<AcRegion> ae;
  uint64_t fn = 0;

  OutputStream* stream(uint32_t w, uint32_t h) {
    UniqueObj<OutputStreamSettings> s(is->createOutputStreamSettings(STREAM_TYPE_EGL));
    auto* i = interface_cast<IEGLOutputStreamSettings>(s);
    if (!i) return nullptr;
    i->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
    i->setResolution(Size2D<uint32_t>(w, h));
    i->setMetadataEnable(true);
    return is->createOutputStream(s.get());
  }
};

ArgusCam::ArgusCam() : p_(new Impl) {}
ArgusCam::~ArgusCam() { close(); }

bool ArgusCam::open(const CamCfg& c) {
  Impl& p = *p_;
  p.c = c;
  auto* ip = interface_cast<ICameraProvider>(provider());
  if (!ip) return std::fprintf(stderr, "[cam%d] no Argus provider (nvargus-daemon?)\n", c.sensor), false;
  std::vector<CameraDevice*> devs;
  ip->getCameraDevices(&devs);
  if (c.sensor >= int(devs.size())) return std::fprintf(stderr, "[cam%d] sensor not found\n", c.sensor), false;

  auto* props = interface_cast<ICameraProperties>(devs[size_t(c.sensor)]);
  std::vector<SensorMode*> modes;
  props->getBasicSensorModes(&modes);
  SensorMode* mode = nullptr;
  uint64_t want_ns = 1000000000ull / uint64_t(c.fps), best = ~0ull;
  for (SensorMode* m : modes) {  // exact 1280x720 whose min frame duration fits 60 fps, closest to it
    auto* sm = interface_cast<ISensorMode>(m);
    Size2D<uint32_t> r = sm->getResolution();
    uint64_t dmin = sm->getFrameDurationRange().min();
    if (r.width() != uint32_t(c.w) || r.height() != uint32_t(c.h) || dmin > want_ns + 100000) continue;
    uint64_t d = want_ns - std::min(dmin, want_ns);
    if (d < best) best = d, mode = m;
  }
  if (!mode) return std::fprintf(stderr, "[cam%d] no %dx%d@%d mode\n", c.sensor, c.w, c.h, c.fps), false;

  p.session.reset(ip->createCaptureSession(devs[size_t(c.sensor)]));
  p.is = interface_cast<ICaptureSession>(p.session);
  if (!p.is) return false;
  p.full.reset(p.stream(uint32_t(c.w), uint32_t(c.h)));
  p.dnn.reset(p.stream(uint32_t(c.dnn_w), uint32_t(c.dnn_h)));
  if (!p.full.get() || !p.dnn.get()) return false;
  p.cfull.reset(FrameConsumer::create(p.full.get()));
  p.cdnn.reset(FrameConsumer::create(p.dnn.get()));
  p.ifull = interface_cast<IFrameConsumer>(p.cfull);
  p.idnn = interface_cast<IFrameConsumer>(p.cdnn);
  if (!p.ifull || !p.idnn) return false;

  p.req.reset(p.is->createRequest(CAPTURE_INTENT_VIDEO_RECORD));
  auto* ir = interface_cast<IRequest>(p.req);
  if (!ir) return false;
  ir->enableOutputStream(p.full.get());
  ir->enableOutputStream(p.dnn.get());
  // §4 item 4: TNR fast on both; edge enhancement off for the DNN stream, on for the full (recording/teleop) stream
  for (int k = 0; k < 2; ++k) {
    InterfaceProvider* ss = ir->getStreamSettings(k ? p.dnn.get() : p.full.get());
    if (auto* dn = interface_cast<IDenoiseSettings>(ss)) dn->setDenoiseMode(DENOISE_MODE_FAST);
    if (auto* ee = interface_cast<IEdgeEnhanceSettings>(ss)) ee->setEdgeEnhanceMode(k ? EDGE_ENHANCE_MODE_OFF
                                                                                      : EDGE_ENHANCE_MODE_FAST);
  }
  auto* src = interface_cast<ISourceSettings>(ir->getSourceSettings());
  auto* sm = interface_cast<ISensorMode>(mode);
  src->setSensorMode(mode);
  src->setFrameDurationRange(Range<uint64_t>(want_ns));
  Range<uint64_t> er = sm->getExposureTimeRange();
  uint64_t lo = std::max<uint64_t>(er.min(), 100000), hi = std::max(lo, std::min(er.max(), c.max_exposure_ns));
  src->setExposureTimeRange(Range<uint64_t>(lo, hi));
  p.ac = interface_cast<IAutoControlSettings>(ir->getAutoControlSettings());
  p.ac->setAeAntibandingMode(AE_ANTIBANDING_MODE_50HZ);  // Sri Lanka mains
  if (c.noir) {
    p.ac->setAwbMode(AWB_MODE_MANUAL);
    p.ac->setWbGains(BayerTuple<float>(c.wb[0], c.wb[1], c.wb[2], c.wb[3]));
  }
  if (p.is->repeat(p.req.get()) != STATUS_OK) return std::fprintf(stderr, "[cam%d] repeat failed\n", c.sensor), false;
  return true;
}

bool ArgusCam::next(CamFrame& f, int dnn_fd, int full_fd) {
  Impl& p = *p_;
  {
    std::lock_guard<std::mutex> g(p.mu);
    if (p.ae_dirty) {
      p.ac->setAeRegions(p.ae);
      p.is->repeat(p.req.get());  // re-submits the request with the new settings
      p.ae_dirty = false;
    }
  }
  UniqueObj<Frame> fd(p.idnn->acquireFrame(kTimeoutNs));
  UniqueObj<Frame> ff(p.ifull->acquireFrame(kTimeoutNs));
  if (!fd.get() || !ff.get()) return false;
  ICaptureMetadata *md = meta_of(fd), *mf = meta_of(ff);
  // Both streams come from the same captures; after a hiccup, drop the older frame until the ids line up again.
  for (int guard = 0; md && mf && md->getCaptureId() != mf->getCaptureId() && guard < 4; ++guard) {
    if (md->getCaptureId() < mf->getCaptureId()) fd.reset(p.idnn->acquireFrame(kTimeoutNs)), md = meta_of(fd);
    else ff.reset(p.ifull->acquireFrame(kTimeoutNs)), mf = meta_of(ff);
    if (!fd.get() || !ff.get()) return false;
  }
  f.t = now_s();
  f.fn = p.fn++;
  if (mf) {
    f.ts_ns = mf->getSensorTimestamp();
    f.lux = NightLogic::lux_proxy(mf->getSensorAnalogGain(), mf->getIspDigitalGain(), mf->getSensorExposureTime());
  }
  auto* idf = interface_cast<IFrame>(fd);
  auto* nd = idf ? interface_cast<NV::IImageNativeBuffer>(idf->getImage()) : nullptr;
  if (!nd || nd->copyToNvBuffer(dnn_fd) != STATUS_OK) return false;
  if (full_fd >= 0) {
    auto* iff = interface_cast<IFrame>(ff);
    auto* nf = iff ? interface_cast<NV::IImageNativeBuffer>(iff->getImage()) : nullptr;
    if (!nf || nf->copyToNvBuffer(full_fd) != STATUS_OK) return false;
  }
  return true;
}

void ArgusCam::set_ae_region(float l, float t, float r, float b) {
  Impl& p = *p_;
  std::lock_guard<std::mutex> g(p.mu);
  p.ae.clear();
  if (r > l && b > t) {
    auto cl = [](float v) { return std::min(std::max(v, 0.f), 1.f); };
    p.ae.push_back(AcRegion(uint32_t(cl(l) * p.c.w), uint32_t(cl(t) * p.c.h), uint32_t(cl(r) * p.c.w),
                            uint32_t(cl(b) * p.c.h), 1.0f));
  }
  p.ae_dirty = p.ac != nullptr;
}

void ArgusCam::set_paused(bool paused) {
  Impl& p = *p_;
  std::lock_guard<std::mutex> g(p.mu);
  if (!p.is) return;
  if (paused) p.is->stopRepeat();
  else p.is->repeat(p.req.get());
}

void ArgusCam::close() {
  Impl& p = *p_;
  if (p.is) {
    p.is->stopRepeat();
    p.is->waitForIdle();
  }
  p.cfull.reset(), p.cdnn.reset();
  p.full.reset(), p.dnn.reset();
  p.req.reset(), p.session.reset();
  p.is = nullptr, p.ac = nullptr;
}

}  // namespace vc
