#!/usr/bin/env bash
# All §14.3 benches that need no human in the loop. Stops the vision unit (beni-vision or -core) for the camera/TRT runs, then restores it.
# Usage: sudo -u beni run_all.sh   (results: /ssd/beni/logs/bench-<date>.md)
source "$(dirname "$0")/common.sh"
unit=beni-vision; systemctl is-active -q beni-vision-core && unit=beni-vision-core
was_active=$(systemctl is-active $unit || true)
restore() { [ "$was_active" != active ] || sudo -n systemctl start $unit; }
trap restore EXIT                                   # a failed camera/TRT bench must not leave vision stopped
sudo -n systemctl stop $unit || true
bash "$HERE/trt.sh"
bash "$HERE/cams.sh" 60
restore && trap - EXIT && sleep 20
bash "$HERE/llama.sh"
bash "$HERE/retrieval.sh"
bash "$HERE/procs.sh" 30
bash "$HERE/system.sh" 600 "${MODE:-active}"
echo "results: $OUT"
