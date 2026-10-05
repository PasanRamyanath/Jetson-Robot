#!/usr/bin/env bash
# §14.3 "Dual 720p60 capture": both sensors at 1280x720@60 into fakesink for SECS (default 60), fps per camera.
# Needs beni-vision stopped (Argus allows one client per sensor).
source "$(dirname "$0")/common.sh"
SECS=${1:-60}
for id in 0 1; do
  timeout -s INT "$((SECS + 5))" gst-launch-1.0 -q nvarguscamerasrc sensor-id=$id num-buffers=$((SECS * 60)) ! \
    'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12' ! \
    fpsdisplaysink video-sink=fakesink text-overlay=false sync=false -v 2>&1 \
    | grep -oP "average: \K[0-9.]+" | tail -1 > /tmp/beni-cam$id.fps &
done
wait
row "dual 720p60 capture (${SECS}s)" "avg fps cam0 / cam1" "$(cat /tmp/beni-cam0.fps) / $(cat /tmp/beni-cam1.fps)" "≥ 59.5"
