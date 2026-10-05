// §5.5.6 steps 2 and 4: the batch-2 person/object detector (one CUDA-graph replay per tick for both cameras) and
// the face chain SCRFD -> Umeyama -> warp_faces -> MobileFaceNet.
#pragma once
#include <string>
#include <vector>

#include "egl_cuda_map.hpp"
#include "geom.hpp"
#include "logic.hpp"
#include "trt_engine.hpp"

namespace vc {

float h2f(uint16_t h);
// Copies element i..i+n of a host output binding into floats whatever its dtype (FP32 / FP16 / INT32).
void to_float(const TrtEngine::Binding& b, size_t first, size_t n, float* out);

// YOLO26n end-to-end [B,300,6] | DeepStream-Yolo [B,N,6] (+ NMS) | DeepStream-Yolo boxes/scores/classes (+ NMS) |
// EfficientNMS_TRT (num_dets/boxes/scores/classes). Boxes come back in 512x288 detector pixels whatever the net size.
class Detector {
 public:
  bool load(const std::string& engine, EglCudaMapper& egl, cudaStream_t s);
  int net_w() const { return w_; }
  int net_h() const { return h_; }
  int batch() const { return batch_; }
  // Fixed RGBA staging fd for batch slot b: VIC-blit the camera's newest DNN frame here before run().
  int staging_fd(int b) const { return stage_fd_[size_t(b)]; }
  void set_gray(int b, bool on);  // night: luma x3 for the NoIR camera (read by the captured graph)
  bool run(cudaStream_t s);       // graph replay + sync
  void decode(int b, float thr, const ClassMask& allow, std::vector<Det>& out) const;
  ~Detector();

 private:
  TrtEngine eng_;
  int w_ = 0, h_ = 0, batch_ = 1;
  std::vector<int> stage_fd_;
  std::vector<Mapped> stage_;
  int* gray_dev_ = nullptr;
  enum Kind { E2E, ROWS, SPLIT, EFFNMS } kind_ = E2E;
  const TrtEngine::Binding *rows_ = nullptr, *boxes_ = nullptr, *scores_ = nullptr, *classes_ = nullptr,
                           *num_ = nullptr;
  int n_ = 0;  // candidates per image
};

struct FaceResult {
  int parent = 0;  // person track id
  Face face;       // full-res pixels
  std::vector<float> emb;
};

class FaceChain {
 public:
  bool load(const std::string& scrfd, const std::string& embed);
  bool ok() const { return ok_; }
  int max_batch() const { return det_.max_batch(); }
  // `rgba`: the camera's full-res frame (mapped). heads: (track id, head crop box in full-res pixels).
  void run(const Mapped& rgba, const std::vector<std::pair<int, Box>>& heads, cudaStream_t s,
           std::vector<FaceResult>& out);
  float det_thresh = 0.5f;

 private:
  static size_t volume_per(const TrtEngine::Binding& b, int batch, int c);
  TrtEngine det_, emb_;
  const TrtEngine::Binding *din_ = nullptr, *ein_ = nullptr, *eout_ = nullptr;
  const TrtEngine::Binding* heads_[3][3] = {};  // [score|bbox|kps][stride 8|16|32]
  int in_ = 320, dim_ = 128;
  bool ok_ = false;
};

// §7.6 OSNet-x0.25 person ReID (256x128, static batch): L2-normalised embeddings for ReidBank. Low-priority stream.
class ReidNet {
 public:
  bool load(const std::string& engine);
  int max_batch() const { return batch_; }
  int dim() const { return dim_; }
  // One embedding per box (full-res pixels, at most max_batch()), in order.
  bool run(const Mapped& rgba, const std::vector<Box>& boxes, cudaStream_t s, std::vector<std::vector<float>>& out);

 private:
  TrtEngine eng_;
  const TrtEngine::Binding *in_ = nullptr, *out_ = nullptr;
  int batch_ = 1, w_ = 128, h_ = 256, dim_ = 512;
};

// §4 item 39 TRT-Pose ResNet18 (224x224, static batch): 18 keypoints (COCO order + neck) per person crop, read as the
// refined peak of each confidence map. One person per crop, so the part-affinity fields are not needed.
class PoseNet {
 public:
  static constexpr int kParts = 18;
  bool load(const std::string& engine);
  int max_batch() const { return batch_; }
  // Per box: kParts x (x, y, conf) in source pixels; conf 0 = part not found.
  bool run(const Mapped& rgba, const std::vector<Box>& boxes, cudaStream_t s, std::vector<std::vector<float>>& out);
  float thresh = 0.1f;  // trt_pose's cmap_threshold

 private:
  TrtEngine eng_;
  const TrtEngine::Binding *in_ = nullptr, *cmap_ = nullptr;
  int batch_ = 1, w_ = 224, h_ = 224, mw_ = 56, mh_ = 56;
};

}  // namespace vc
