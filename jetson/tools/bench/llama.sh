#!/usr/bin/env bash
# §14.3 "llama.cpp Qwen2.5-0.5B Q4_K_M": prompt and generation tok/s on 2 threads (cores 2-3), as beni-llm runs it.
source "$(dirname "$0")/common.sh"
B=$ROOT/jetson/llm/llama.cpp/build/bin
[ -x "$B/llama-bench" ] || cmake --build "$ROOT/jetson/llm/llama.cpp/build" -j2 --target llama-bench
res=$(taskset -c 2,3 "$B/llama-bench" -m "$M/qwen2.5-0.5b-instruct-q4_k_m.gguf" -t 2 -p 256 -n 64 -o csv 2>/dev/null \
      | awk -F, 'NR>1{gsub(/"/,""); print $(NF-1)}' | paste -sd/ - || true)
row "llama.cpp qwen2.5-0.5b q4_k_m -t 2" "pp256 / tg64 tok/s" "${res:-?}"
