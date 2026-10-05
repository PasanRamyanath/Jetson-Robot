#!/usr/bin/env bash
# §14.3 "Memory retrieval": 10k synthetic episodes, p50/p99 ms of Retriever.retrieve (bge-small if present).
source "$(dirname "$0")/common.sh"
taskset -c 2,3 "$PYVENV" "$HERE/retrieval.py" --n "${1:-10000}" --models "$M" | tee -a "$OUT"
