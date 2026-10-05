// §5.5.3 TensorRT 8.2 runtime: bindings by name with their real dtype, device buffers sized for the max profile,
// pinned host mirrors for outputs, optional CUDA-graph replay (static shapes only).
#pragma once
#include <cuda_runtime.h>
#include <NvInfer.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vc {

class TrtEngine {
 public:
  struct Binding {
    std::string name;
    int index = -1;
    bool input = false, half = false, dynamic = false;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims max{};        // shape the buffers were allocated for
    size_t elem = 4, bytes = 0;  // bytes at `max`
    void* dev = nullptr;
    void* host = nullptr;        // pinned mirror (outputs only)
  };

  TrtEngine() = default;
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  bool load(const std::string& path);
  const Binding* find(const char* name) const;
  const std::vector<Binding>& bindings() const { return b_; }
  int max_batch() const { return max_batch_; }

  // Dynamic-batch engines (faces): set the leading dim of every dynamic input before enqueue.
  bool set_batch(int n);
  // enqueueV2 + D2H of all outputs. `pre` runs on the stream first (preprocess kernels).
  bool run(cudaStream_t s, const std::function<void(cudaStream_t)>& pre);
  // Capture pre + enqueue + D2H into a graph once; later ticks call launch(). Falls back to run() if capture fails.
  bool capture(cudaStream_t s, const std::function<void(cudaStream_t)>& pre);
  bool launch(cudaStream_t s);

 private:
  struct Del {
    template <typename T>
    void operator()(T* p) const { delete p; }
  };
  std::unique_ptr<nvinfer1::IRuntime, Del> rt_;
  std::unique_ptr<nvinfer1::ICudaEngine, Del> eng_;
  std::unique_ptr<nvinfer1::IExecutionContext, Del> ctx_;
  std::vector<Binding> b_;
  std::vector<void*> ptrs_;
  int max_batch_ = 1;
  cudaGraphExec_t exec_ = nullptr;
  std::function<void(cudaStream_t)> pre_;

  bool body(cudaStream_t s);
};

// Stream pair per §5.5.3: foreground (detector, faces) at the highest priority, background at the lowest.
struct Streams {
  cudaStream_t fg = nullptr, bg = nullptr;
  Streams();
  ~Streams();
};

}  // namespace vc
