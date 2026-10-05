// §5.5.6 steps 2 and 4: detector (CUDA graph, batch = both cameras) and the face chain.
#include "detector.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "kernels.hpp"
#include "nvbuf.hpp"

namespace vc {

float h2f(uint16_t h) {
  uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
  if (e == 0) {
    if (!m) {
      x = s;
    } else {  // subnormal
      e = 127 - 15 + 1;
      while (!(m & 0x400)) m <<= 1, --e;
      x = s | (e << 23) | ((m & 0x3ff) << 13);
    }
  } else {
    x = s | (e == 31 ? 0x7f800000u | (m << 13) : ((e + 127 - 15) << 23) | (m << 13));
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

void to_float(const TrtEngine::Binding& b, size_t first, size_t n, float* out) {
  switch (b.dtype) {
    case nvinfer1::DataType::kHALF: {
      auto* p = static_cast<const uint16_t*>(b.host) + first;
      for (size_t i = 0; i < n; ++i) out[i] = h2f(p[i]);
      break;
    }
    case nvinfer1::DataType::kINT32: {
      auto* p = static_cast<const int32_t*>(b.host) + first;
      for (size_t i = 0; i < n; ++i) out[i] = float(p[i]);
      break;
    }
    default:
      std::memcpy(out, static_cast<const float*>(b.host) + first, n * sizeof(float));
  }
}

namespace {

const TrtEngine::Binding* first_of(const TrtEngine& e, std::initializer_list<const char*> names) {
  for (const char* n : names)
    if (const TrtEngine::Binding* b = e.find(n)) return b;
  return nullptr;
}

const TrtEngine::Binding* first_io(const TrtEngine& e, bool input) {
  for (const auto& b : e.bindings())
    if (b.input == input) return &b;
  return nullptr;
}

}  // namespace

// ---------------------------------------------------------------- Detector

Detector::~Detector() {
  if (gray_dev_) cudaFree(gray_dev_);
  for (int fd : stage_fd_) NvBufferDestroy(fd);
}

bool Detector::load(const std::string& path, EglCudaMapper& egl, cudaStream_t s) {
  if (!eng_.load(path)) return false;
  const TrtEngine::Binding* in = first_io(eng_, true);
  if (!in || in->max.nbDims != 4) return std::fprintf(stderr, "[det] expected NCHW input\n"), false;
  batch_ = in->max.d[0], h_ = in->max.d[2], w_ = in->max.d[3];
  if ((num_ = eng_.find("num_dets"))) {
    kind_ = EFFNMS;
    boxes_ = first_of(eng_, {"det_boxes", "boxes"}), scores_ = first_of(eng_, {"det_scores", "scores"});
    classes_ = first_of(eng_, {"det_classes", "classes"});
  } else if ((boxes_ = eng_.find("boxes"))) {
    kind_ = SPLIT, scores_ = eng_.find("scores"), classes_ = eng_.find("classes");
  } else {
    rows_ = first_io(eng_, false);
    if (!rows_ || rows_->max.nbDims != 3 || rows_->max.d[2] != 6)
      return std::fprintf(stderr, "[det] unknown output layout\n"), false;
    n_ = rows_->max.d[1];
    kind_ = n_ <= 1000 ? E2E : ROWS;  // YOLO26 top-k (300) vs raw anchors (3024 at 512x288)
  }
  if (kind_ == EFFNMS || kind_ == SPLIT) {
    if (!boxes_ || !scores_ || !classes_) return std::fprintf(stderr, "[det] missing NMS outputs\n"), false;
    n_ = boxes_->max.d[1];
  }
  for (int b = 0; b < batch_; ++b) {
    int fd = nvbuf_create(w_, h_, NvBufferColorFormat_ABGR32, NvBufferLayout_Pitch, NvBufferTag_NONE);
    if (fd < 0) return false;
    stage_fd_.push_back(fd);
    stage_.push_back(egl.map(fd));
    if (!stage_.back().ptr) return std::fprintf(stderr, "[det] EGL map failed\n"), false;
  }
  cudaMalloc(&gray_dev_, sizeof(int) * size_t(batch_));
  cudaMemset(gray_dev_, 0, sizeof(int) * size_t(batch_));
  size_t plane = size_t(3) * size_t(w_) * size_t(h_) * in->elem;
  auto* dev = static_cast<char*>(in->dev);
  bool half = in->half;
  auto pre = [this, dev, plane, half](cudaStream_t st) {
    for (int b = 0; b < batch_; ++b)
      preprocess(stage_[size_t(b)].ptr, stage_[size_t(b)].pitch, w_, h_, dev + size_t(b) * plane, half, kYolo,
                 gray_dev_ + b, st);
  };
  if (!eng_.capture(s, pre)) eng_.run(s, pre);  // leaves pre_ set: launch() falls back to per-tick enqueue
  std::fprintf(stderr, "[det] %s: %dx%d b%d kind=%d n=%d\n", path.c_str(), w_, h_, batch_, int(kind_), n_);
  return true;
}

void Detector::set_gray(int b, bool on) {
  int v = on ? 1 : 0;
  if (b < batch_) cudaMemcpy(gray_dev_ + b, &v, sizeof v, cudaMemcpyHostToDevice);
}

bool Detector::run(cudaStream_t s) { return eng_.launch(s) && cudaStreamSynchronize(s) == cudaSuccess; }

void Detector::decode(int b, float thr, const ClassMask& allow, std::vector<Det>& out) const {
  std::vector<Det> d;
  Letterbox id;  // decode in network pixels, rescale below (the ISP stream fills the net input exactly)
  std::vector<float> buf;
  if (kind_ == E2E || kind_ == ROWS) {
    buf.resize(size_t(n_) * 6);
    to_float(*rows_, size_t(b) * size_t(n_) * 6, buf.size(), buf.data());
    decode_e2e(buf.data(), n_, thr, allow, id, float(w_), float(h_), d);
  } else {
    const size_t n = size_t(n_);
    std::vector<float> bx(n * 4), sc(n), cl(n);
    to_float(*boxes_, size_t(b) * bx.size(), bx.size(), bx.data());
    to_float(*scores_, size_t(b) * sc.size(), sc.size(), sc.data());
    to_float(*classes_, size_t(b) * cl.size(), cl.size(), cl.data());
    float num = float(n_);
    if (kind_ == EFFNMS) to_float(*num_, size_t(b), 1, &num);
    decode_nms(std::min(int(num), n_), bx.data(), sc.data(), cl.data(), thr, allow, id, float(w_), float(h_), d);
  }
  if (kind_ == ROWS || kind_ == SPLIT) {
    std::vector<int> keep = nms(d, 0.45f, [](const Det& x) { return x.b; }, [](const Det& x) { return x.score; });
    std::vector<Det> k;
    for (int i : keep) k.push_back(d[size_t(i)]);
    d.swap(k);
  }
  float sx = 512.f / float(w_), sy = 288.f / float(h_);
  for (Det& x : d) {
    x.b.x1 *= sx, x.b.x2 *= sx, x.b.y1 *= sy, x.b.y2 *= sy;
    out.push_back(x);
  }
}

// ---------------------------------------------------------------- FaceChain

bool FaceChain::load(const std::string& scrfd, const std::string& embed) {
  if (!det_.load(scrfd) || !emb_.load(embed)) return false;
  din_ = first_io(det_, true), ein_ = first_io(emb_, true), eout_ = first_io(emb_, false);
  if (!din_ || din_->max.nbDims != 4 || !ein_ || !eout_) return false;
  in_ = din_->max.d[2];
  int batch = din_->max.d[0];
  for (const auto& b : det_.bindings()) {  // classify SCRFD heads by channel count and anchor count
    if (b.input) continue;
    if (batch > 1 && b.max.nbDims != 3)  // the batch-1 export flattens H,W before N: per-crop slices would be wrong
      return std::fprintf(stderr, "[face] SCRFD b%d needs batched outputs [B,N,C] (insightface batched export)\n",
                          batch), false;
    int c = b.max.d[b.max.nbDims - 1];
    size_t per = volume_per(b, batch, c);
    int kind = c == 1 ? 0 : c == 4 ? 1 : c == 10 ? 2 : -1;
    for (int k = 0; k < 3 && kind >= 0; ++k) {
      int g = in_ / (8 << k);
      if (per == size_t(g * g * 2)) heads_[kind][k] = &b;
    }
  }
  for (auto& row : heads_)
    for (auto* h : row)
      if (!h) return std::fprintf(stderr, "[face] unexpected SCRFD outputs\n"), false;
  dim_ = eout_->max.d[eout_->max.nbDims - 1];
  ok_ = true;
  std::fprintf(stderr, "[face] scrfd %d b%d, embed dim %d b%d\n", in_, batch, dim_, emb_.max_batch());
  return true;
}

size_t FaceChain::volume_per(const TrtEngine::Binding& b, int batch, int c) {
  size_t v = 1;
  for (int i = 0; i < b.max.nbDims; ++i) v *= size_t(b.max.d[i]);
  return v / size_t(batch) / size_t(c);
}

void FaceChain::run(const Mapped& rgba, const std::vector<std::pair<int, Box>>& heads, cudaStream_t s,
                    std::vector<FaceResult>& out) {
  int batch = din_->max.d[0];
  int n = std::min(int(heads.size()), std::min(batch, 8));
  if (!ok_ || n <= 0 || !rgba.ptr) return;
  Crop crops[8];
  for (int i = 0; i < n; ++i) {
    const Box& b = heads[size_t(i)].second;
    crops[i] = Crop{b.x1, b.y1, b.x2, b.y2};
  }
  auto pre = [&](cudaStream_t st) {
    crop_resize(rgba.ptr, rgba.pitch, rgba.w, rgba.h, crops, n, in_, din_->dev, din_->half, kInsight, st);
  };
  if (!det_.run(s, pre) || cudaStreamSynchronize(s) != cudaSuccess) return;

  std::vector<FaceResult> found;
  std::vector<float> buf[3][3];
  for (int i = 0; i < n; ++i) {
    const float *sc[3], *bb[3], *kp[3];
    for (int k = 0; k < 3; ++k) {
      int g = in_ / (8 << k), cnt = g * g * 2;
      const int ch[3] = {1, 4, 10};
      const float** dst[3] = {sc, bb, kp};
      for (int j = 0; j < 3; ++j) {
        size_t len = size_t(cnt) * size_t(ch[j]);
        buf[j][k].resize(len);
        to_float(*heads_[j][k], size_t(i) * len, len, buf[j][k].data());
        dst[j][k] = buf[j][k].data();
      }
    }
    const Crop& c = crops[i];
    float cw = c.x2 - c.x1, chh = c.y2 - c.y1, sc_ = std::min(in_ / cw, in_ / chh);
    Letterbox lb;
    lb.scale = sc_, lb.pad_x = 0.5f * (in_ - cw * sc_) - c.x1 * sc_, lb.pad_y = 0.5f * (in_ - chh * sc_) - c.y1 * sc_;
    std::vector<Face> faces;
    decode_scrfd(sc, bb, kp, in_, det_thresh, lb, faces);
    if (faces.empty()) continue;  // decode_scrfd returns best first
    FaceResult r;
    r.parent = heads[size_t(i)].first, r.face = faces[0];
    found.push_back(r);
  }
  int m = std::min(int(found.size()), std::min(emb_.max_batch(), 8));
  if (m <= 0) return;
  float M[8][6];
  for (int i = 0; i < m; ++i) similarity(found[size_t(i)].face.kps, M[i]);
  auto pre2 = [&](cudaStream_t st) {
    warp_faces(rgba.ptr, rgba.pitch, rgba.w, rgba.h, M, m, ein_->dev, ein_->half, st);
  };
  emb_.set_batch(m);
  if (!emb_.run(s, pre2) || cudaStreamSynchronize(s) != cudaSuccess) return;
  for (int i = 0; i < m; ++i) {
    FaceResult& r = found[size_t(i)];
    r.emb.resize(size_t(dim_));
    to_float(*eout_, size_t(i) * size_t(dim_), size_t(dim_), r.emb.data());
    l2norm(r.emb.data(), dim_);
    out.push_back(std::move(r));
  }
}

// ---------------------------------------------------------------- ReidNet

bool ReidNet::load(const std::string& engine) {
  if (!eng_.load(engine)) return false;
  in_ = first_io(eng_, true), out_ = first_io(eng_, false);
  if (!in_ || in_->max.nbDims != 4 || !out_) return std::fprintf(stderr, "[reid] unexpected engine I/O\n"), false;
  batch_ = std::min(in_->max.d[0], 8), h_ = in_->max.d[2], w_ = in_->max.d[3];
  dim_ = out_->max.d[out_->max.nbDims - 1];
  std::fprintf(stderr, "[reid] %dx%d b%d dim %d\n", w_, h_, batch_, dim_);
  return true;
}

bool ReidNet::run(const Mapped& rgba, const std::vector<Box>& boxes, cudaStream_t s,
                  std::vector<std::vector<float>>& out) {
  int n = std::min(int(boxes.size()), batch_);
  if (n <= 0 || !rgba.ptr) return false;
  Crop crops[8];
  for (int i = 0; i < n; ++i) crops[i] = Crop{boxes[size_t(i)].x1, boxes[size_t(i)].y1, boxes[size_t(i)].x2,
                                              boxes[size_t(i)].y2};
  auto pre = [&](cudaStream_t st) {
    crop_stretch(rgba.ptr, rgba.pitch, rgba.w, rgba.h, crops, n, w_, h_, in_->dev, in_->half, st);
  };
  eng_.set_batch(n);
  if (!eng_.run(s, pre) || cudaStreamSynchronize(s) != cudaSuccess) return false;
  out.assign(size_t(n), std::vector<float>(size_t(dim_)));
  for (int i = 0; i < n; ++i) {
    to_float(*out_, size_t(i) * size_t(dim_), size_t(dim_), out[size_t(i)].data());
    l2norm(out[size_t(i)].data(), dim_);
  }
  return true;
}

bool PoseNet::load(const std::string& engine) {
  if (!eng_.load(engine)) return false;
  in_ = first_io(eng_, true);
  for (const auto& b : eng_.bindings())  // cmap [B,18,H/4,W/4]; the paf output [B,42,..] is ignored
    if (!b.input && b.max.nbDims == 4 && b.max.d[1] == kParts) cmap_ = &b;
  if (!in_ || in_->max.nbDims != 4 || !cmap_) return std::fprintf(stderr, "[pose] unexpected engine I/O\n"), false;
  batch_ = std::min(in_->max.d[0], 8), h_ = in_->max.d[2], w_ = in_->max.d[3];
  mh_ = cmap_->max.d[2], mw_ = cmap_->max.d[3];
  std::fprintf(stderr, "[pose] %dx%d b%d map %dx%d\n", w_, h_, batch_, mw_, mh_);
  return true;
}

bool PoseNet::run(const Mapped& rgba, const std::vector<Box>& boxes, cudaStream_t s,
                  std::vector<std::vector<float>>& out) {
  int n = std::min(int(boxes.size()), batch_);
  if (n <= 0 || !rgba.ptr) return false;
  Crop crops[8];
  for (int i = 0; i < n; ++i) crops[i] = Crop{boxes[size_t(i)].x1, boxes[size_t(i)].y1, boxes[size_t(i)].x2,
                                              boxes[size_t(i)].y2};
  auto pre = [&](cudaStream_t st) {  // ImageNet mean/std, like trt_pose's torchvision preprocessing
    crop_stretch(rgba.ptr, rgba.pitch, rgba.w, rgba.h, crops, n, w_, h_, in_->dev, in_->half, st);
  };
  eng_.set_batch(n);
  if (!eng_.run(s, pre) || cudaStreamSynchronize(s) != cudaSuccess) return false;
  size_t plane = size_t(mw_) * size_t(mh_);
  std::vector<float> m(plane);
  out.assign(size_t(n), std::vector<float>(size_t(kParts) * 3, 0.f));
  for (int i = 0; i < n; ++i) {
    const Box& b = boxes[size_t(i)];
    float sx = b.w() / float(mw_), sy = b.h() / float(mh_);
    for (int k = 0; k < kParts; ++k) {
      to_float(*cmap_, (size_t(i) * kParts + size_t(k)) * plane, plane, m.data());
      size_t best = size_t(std::max_element(m.begin(), m.end()) - m.begin());
      if (m[best] < thresh) continue;
      int px = int(best % size_t(mw_)), py = int(best / size_t(mw_));
      float sw = 0, ax = 0, ay = 0;  // 3x3 weighted centroid: sub-cell accuracy on a 4x-downsampled map
      for (int y = std::max(py - 1, 0); y <= std::min(py + 1, mh_ - 1); ++y)
        for (int x = std::max(px - 1, 0); x <= std::min(px + 1, mw_ - 1); ++x) {
          float q = std::max(m[size_t(y) * size_t(mw_) + size_t(x)], 0.f);
          sw += q, ax += q * float(x), ay += q * float(y);
        }
      float* o = &out[size_t(i)][size_t(k) * 3];
      o[0] = b.x1 + (ax / sw + 0.5f) * sx, o[1] = b.y1 + (ay / sw + 0.5f) * sy, o[2] = m[best];
    }
  }
  return true;
}

}  // namespace vc
