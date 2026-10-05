#!/usr/bin/env bash
# Python 3.8 "agent island" venv (§2.4). Host py3.6 keeps DeepStream/pyds/TensorRT.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${VENV:-$ROOT/.venv38}
[ -d "$VENV" ] || python3.8 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -U "pip<24.1" wheel setuptools
# glibc 2.27: if a wheel is missing, pip builds from source; cmake/ninja make that possible.
pip install cmake ninja
pip install -r "$ROOT/jetson/agent/requirements.txt" || {
  echo "retrying failures from source"; pip install --no-binary sherpa-onnx sherpa-onnx; }
pip install -e "$ROOT/shared[memory]" -e "$ROOT/jetson/agent"
python - <<'PY'
import sqlite3, numpy, zmq, msgpack, onnxruntime
print("sqlite", sqlite3.sqlite_version, "numpy", numpy.__version__, "zmq", zmq.zmq_version(), "ort", onnxruntime.__version__)
con = sqlite3.connect(":memory:"); con.execute("create virtual table t using fts5(x)"); print("fts5 ok")
PY
# Host py3.6 side (vision, tools): system-wide, since beni-vision and the MediaMTX hook run as beni. pip 21.3 (last
# for 3.6) reads the manylinux2014 aarch64 wheels; apt's pip 9 would compile numpy and pyzmq.
sudo -H python3 -m pip install -q -U "pip<22"
sudo -H python3 -m pip install -r "$ROOT/jetson/requirements-py36.txt"
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$ROOT/shared" | sudo tee "$SP/beni.pth" >/dev/null
