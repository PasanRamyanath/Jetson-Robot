#!/usr/bin/env bash
# JetPack 4.6.x pieces every later step assumes (§2.4). NVIDIA's SD-card image ships them; an eMMC module (Waveshare
# kit, production P3448-0002) flashed with flash.sh or SDK Manager "OS only" has the BSP alone: no CUDA, TensorRT
# (trtexec), Multimedia API, nvidia-docker or DeepStream. Idempotent: installs only what is missing.
#   sudo bash jetson/setup/00_jetpack.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
grep -q "R32 (release), REVISION: 7" /etc/nv_tegra_release || { echo "needs JetPack 4.6.x (L4T R32.7.x)"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git make curl rsync ca-certificates python3-pip python3-dev python3-gi \
  bluez parted cryptsetup-bin nano

# CUDA 10.2, cuDNN 8.2, TensorRT 8.2 (trtexec), VPI, Multimedia API (vision_core), nvidia-container (docker)
if [ ! -x /usr/src/tensorrt/bin/trtexec ] || [ ! -d /usr/src/jetson_multimedia_api ] \
   || [ ! -d /usr/local/cuda-10.2 ]; then
  apt-get install -y nvidia-jetpack                     # ~2.5 GB installed
fi
command -v nvidia-container-runtime >/dev/null || apt-get install -y docker.io nvidia-docker2

# DeepStream 6.0.1 (beni-vision, the Phase 1-2 path) and its runtime deps
if [ ! -d /opt/nvidia/deepstream/deepstream-6.0 ]; then
  apt-get install -y libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 \
    libyaml-cpp-dev
  apt-get install -y deepstream-6.0 || {
    echo "apt has no deepstream-6.0: download deepstream-6.0_6.0.1-1_arm64.deb from developer.nvidia.com,"
    echo "then: sudo apt-get install -y ./deepstream-6.0_6.0.1-1_arm64.deb and re-run this script"; exit 1; }
fi

# pyds 1.1.1, the DeepStream Python bindings for host Python 3.6 (beni_vision.py)
if ! python3 -c "import pyds" 2>/dev/null; then
  W=pyds-1.1.1-py3-none-linux_aarch64.whl
  curl -fL -o "/tmp/$W" "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.1/$W"
  python3 -m pip install "/tmp/$W" && rm -f "/tmp/$W"
fi

# jtop (§16 Phase 0): GPU/EMC/thermal at a glance
command -v jtop >/dev/null || python3 -m pip install -U jetson-stats || echo "warning: jetson-stats failed (optional)"

echo "cuda: $(/usr/local/cuda/bin/nvcc --version | tail -1)"
echo "tensorrt: $(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null)"
echo "deepstream: $(head -1 /opt/nvidia/deepstream/deepstream-6.0/version 2>/dev/null)"
python3 -c "import pyds; print('pyds ok')"
