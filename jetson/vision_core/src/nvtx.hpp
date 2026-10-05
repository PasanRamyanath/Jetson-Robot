// NVTX ranges for Nsight Systems (§14.4): compiled in only when CMake finds libnvToolsExt (BENI_NVTX).
#pragma once

#ifdef BENI_NVTX
#include <nvToolsExt.h>
namespace beni {
struct NvtxScope {
  explicit NvtxScope(const char* name) { nvtxRangePushA(name); }
  ~NvtxScope() { nvtxRangePop(); }
  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;
};
}  // namespace beni
#define BENI_NVTX_CAT2(a, b) a##b
#define BENI_NVTX_CAT(a, b) BENI_NVTX_CAT2(a, b)
#define NVTX_RANGE(name) ::beni::NvtxScope BENI_NVTX_CAT(nvtx_scope_, __LINE__)(name)
#else
#define NVTX_RANGE(name) \
  do {                   \
  } while (0)
#endif
