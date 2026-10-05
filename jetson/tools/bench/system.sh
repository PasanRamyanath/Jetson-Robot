#!/usr/bin/env bash
# §14.3 "Full vision_core / beni_vision" and §14.2 modes: tegrastats for SECS (default 600) with the stack running,
# then mean/p95 of CPU, GPU, EMC, NVENC/NVDEC, power and temperature. Label the run with the robot mode.
source "$(dirname "$0")/common.sh"
SECS=${1:-600}; MODE=${2:-active}
csv=$(mktemp --suffix .csv)
timeout -s INT "$SECS" "$PYHOST" "$ROOT/jetson/tools/tegrastats_logger.py" "$csv" 500 || true
"$PYHOST" "$HERE/summarize.py" "$csv" "system ($MODE, ${SECS}s)" | tee -a "$OUT"
rm -f "$csv"
