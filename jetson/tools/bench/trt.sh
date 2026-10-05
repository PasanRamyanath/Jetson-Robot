#!/usr/bin/env bash
# §14.3 rows 1-4, 8: every TensorRT engine -> mean/p99 ms, peak RSS, and whether cuDNN/cuBLAS got mapped (lean check).
# Stop beni-vision first for clean numbers: sudo systemctl stop beni-vision
source "$(dirname "$0")/common.sh"
TRT=/usr/src/tensorrt/bin/trtexec
for eng in "$E"/*.engine; do
  [ -f "$eng" ] || { echo "no engines in $E (make engines)"; exit 1; }
  n=$(basename "$eng" .engine); log=$(mktemp)
  /usr/bin/time -v "$TRT" --loadEngine="$eng" --iterations=500 --avgRuns=100 --useSpinWait --useCudaGraph \
    > "$log" 2>&1 &
  pid=$!; sleep 4
  libs=$(grep -ohE "lib(cudnn|cublas|cublasLt)\.so" /proc/$(pgrep -P $pid trtexec || echo $pid)/maps 2>/dev/null \
         | sort -u | tr '\n' ' ' || true)
  wait $pid || true
  mean=$(grep -oP "GPU Compute Time: .*?mean = \K[0-9.]+" "$log" | tail -1 || true)   # a failed engine -> "?"
  p99=$(grep -oP "GPU Compute Time: .*?percentile\(99%\) = \K[0-9.]+" "$log" | tail -1 || true)
  rss=$(grep -oP "Maximum resident set size \(kbytes\): \K[0-9]+" "$log" || true)
  row "trt $n" "mean / p99 ms" "${mean:-?} / ${p99:-?}"
  row "trt $n" "peak RSS MB; cuDNN/cuBLAS mapped" "$(( ${rss:-0} / 1024 )); ${libs:-none}" "none (lean)"
  rm -f "$log"
done
