#!/usr/bin/env bash
# Offline LLM (§8.3): llama.cpp llama-server, CPU/NEON, gcc-9 (GCC 7.5 is too old). CUDA build not recommended
# (sm_53/CUDA 10.2 unmaintained upstream; the GPU is busy with vision).
# Pin a tag that builds on 18.04; bump deliberately.
set -euo pipefail
TAG=${LLAMA_TAG:-b4600}
D=$(cd "$(dirname "$0")" && pwd)/llama.cpp
M=${BENI_MODELS:-/ssd/beni/models}
[ -d "$D" ] || git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$D"
python3 -m pip install --user -q "cmake>=3.18"
export PATH="$HOME/.local/bin:$PATH"
cmake -S "$D" -B "$D/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build "$D/build" -j2 --target llama-server      # -j2: -j4 OOMs with vision running
GGUF="$M/qwen2.5-0.5b-instruct-q4_k_m.gguf"
if [ ! -f "$GGUF" ]; then
  mkdir -p "$M"
  curl -fL -o "$GGUF.part" \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
  mv "$GGUF.part" "$GGUF"
fi
echo "built: $D/build/bin/llama-server"
