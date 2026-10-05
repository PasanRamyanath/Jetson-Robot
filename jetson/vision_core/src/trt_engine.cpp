// §5.5.3 TensorRT 8.2 runtime with CUDA-graph replay.
#include "trt_engine.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>

namespace vc {
namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", msg);
  }
};
Logger g_log;

size_t elem_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 4;  // kFLOAT, kINT32
  }
}

size_t volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= size_t(d.d[i] < 0 ? 1 : d.d[i]);
  return v;
}

}  // namespace

TrtEngine::~TrtEngine() {
  if (exec_) cudaGraphExecDestroy(exec_);
  for (auto& b : b_) {
    if (b.dev) cudaFree(b.dev);
    if (b.host) cudaFreeHost(b.host);
  }
}

bool TrtEngine::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::fprintf(stderr, "[trt] cannot open %s\n", path.c_str()), false;
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  rt_.reset(nvinfer1::createInferRuntime(g_log));
  if (!rt_) return false;
  eng_.reset(rt_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!eng_) return std::fprintf(stderr, "[trt] deserialize failed: %s\n", path.c_str()), false;
  ctx_.reset(eng_->createExecutionContext());
  if (!ctx_) return false;
  int n = eng_->getNbBindings();
  b_.resize(n);
  ptrs_.assign(n, nullptr);
  for (int i = 0; i < n; ++i) {
    Binding& b = b_[i];
    b.name = eng_->getBindingName(i);
    b.index = i;
    b.input = eng_->bindingIsInput(i);
    b.dtype = eng_->getBindingDataType(i);
    b.half = b.dtype == nvinfer1::DataType::kHALF;
    b.elem = elem_size(b.dtype);
    b.max = eng_->getBindingDimensions(i);
    for (int k = 0; k < b.max.nbDims; ++k) b.dynamic |= b.max.d[k] < 0;
    if (b.input && b.dynamic) {
      b.max = eng_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
      ctx_->setBindingDimensions(i, b.max);
      if (b.max.nbDims) max_batch_ = std::max(max_batch_, int(b.max.d[0]));
    }
  }
  if (!ctx_->allInputDimensionsSpecified()) return std::fprintf(stderr, "[trt] unresolved input dims\n"), false;
  for (int i = 0; i < n; ++i) {
    Binding& b = b_[i];
    if (!b.input) b.max = ctx_->getBindingDimensions(i);  // resolved at the max input shape
    else if (!b.dynamic && b.max.nbDims) max_batch_ = std::max(max_batch_, int(b.max.d[0]));
    b.bytes = volume(b.max) * b.elem;
    if (cudaMalloc(&b.dev, b.bytes) != cudaSuccess) return false;
    if (!b.input && cudaHostAlloc(&b.host, b.bytes, cudaHostAllocDefault) != cudaSuccess) return false;
    ptrs_[i] = b.dev;
  }
  return true;
}

const TrtEngine::Binding* TrtEngine::find(const char* name) const {
  for (auto& b : b_)
    if (b.name == name) return &b;
  return nullptr;
}

bool TrtEngine::set_batch(int n) {
  if (exec_) return n == max_batch_;  // graphs are captured at a fixed shape
  for (auto& b : b_) {
    if (!b.input || !b.dynamic) continue;
    nvinfer1::Dims d = b.max;
    d.d[0] = std::min(n, int(b.max.d[0]));
    if (!ctx_->setBindingDimensions(b.index, d)) return false;
  }
  return true;
}

bool TrtEngine::body(cudaStream_t s) {
  if (pre_) pre_(s);
  if (!ctx_->enqueueV2(ptrs_.data(), s, nullptr)) return false;
  for (auto& b : b_) {
    if (b.input) continue;
    size_t bytes = volume(ctx_->getBindingDimensions(b.index)) * b.elem;
    cudaMemcpyAsync(b.host, b.dev, bytes, cudaMemcpyDeviceToHost, s);
  }
  return true;
}

bool TrtEngine::run(cudaStream_t s, const std::function<void(cudaStream_t)>& pre) {
  pre_ = pre;
  return body(s);
}

bool TrtEngine::capture(cudaStream_t s, const std::function<void(cudaStream_t)>& pre) {
  pre_ = pre;
  if (!body(s) || cudaStreamSynchronize(s) != cudaSuccess) return false;  // warm-up outside the capture
  cudaGraph_t g = nullptr;
  if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
  bool ok = body(s);
  if (cudaStreamEndCapture(s, &g) != cudaSuccess || !ok || !g) {
    cudaGetLastError();
    std::fprintf(stderr, "[trt] graph capture failed; using enqueueV2 per tick\n");
    return false;
  }
  ok = cudaGraphInstantiate(&exec_, g, nullptr, nullptr, 0) == cudaSuccess;
  cudaGraphDestroy(g);
  if (!ok) exec_ = nullptr;
  return ok;
}

bool TrtEngine::launch(cudaStream_t s) { return exec_ ? cudaGraphLaunch(exec_, s) == cudaSuccess : body(s); }

Streams::Streams() {
  int least = 0, greatest = 0;
  cudaDeviceGetStreamPriorityRange(&least, &greatest);
  cudaStreamCreateWithPriority(&fg, cudaStreamNonBlocking, greatest);
  cudaStreamCreateWithPriority(&bg, cudaStreamNonBlocking, least);
}

Streams::~Streams() {
  if (fg) cudaStreamDestroy(fg);
  if (bg) cudaStreamDestroy(bg);
}

}  // namespace vc
