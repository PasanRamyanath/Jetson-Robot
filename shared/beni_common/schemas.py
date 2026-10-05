"""IPC endpoints, topics and message envelopes (§7.2) plus cloud frame types (§9.2).

Python 3.6-safe: plain dicts, no dataclasses. Adding a field to a message => bump V.
"""
import time

import msgpack

V = 1
SOCK_DIR = "/tmp/beni"

EP = {
    "vision": "ipc:///tmp/beni/vision.sock",            # PUB  det / jpeg / motion / scene / replay / pose
    "vision_ctrl": "ipc:///tmp/beni/vision_ctrl.sock",  # REP  snapshot / bitrate / mode / bg
    "robot_state": "ipc:///tmp/beni/robot_state.sock",  # PUB  pose, battery, bumpers, nav (10 Hz)
    "robot_cmd": "ipc:///tmp/beni/robot_cmd.sock",      # REP  goto / look_at / follow / stop / dock
    "face_ctrl": "ipc:///tmp/beni/face_ctrl.sock",      # PULL expression / overlay (beni_face)
    "events": "ipc:///tmp/beni/events.sock",            # PUB  wake, speech_*, person_seen, ... (agent binds)
    "sched": "ipc:///tmp/beni/sched.sock",              # PUB  admit/pause + power mode (scheduler binds)
}

# Local topics (first ZMQ frame).
T_DET, T_JPEG, T_MOTION, T_SCENE = b"det", b"jpeg", b"motion", b"scene"
T_STATE = b"state"
T_REPLAY = b"replay"
T_POSE = b"pose"

# gie-1 class ids (jetson/vision/configs/labels.txt; the bridge's kCoco mirrors this order).
COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush")

EVENTS = ("wake", "listening", "speech_start", "speech_end", "person_seen", "person_left",
          "unknown_face", "low_battery", "bump", "cliff", "motion", "mode", "brain", "gesture")

# Cloud WebSocket frame types (§9.2).
CLOUD_J2K = ("hello", "turn.begin", "audio.chunk", "audio.end", "action.result",
             "vision.snapshot", "interrupt", "memory.delta", "memory.ack", "heartbeat", "brain.stop", "replay.flag")
CLOUD_K2J = ("hello_ok", "stt.partial", "stt.final", "llm.delta", "tts.chunk", "action",
             "vision.request", "memory.delta", "memory.ack", "heartbeat", "turn.end", "scene", "error",
             "replay.result")

# Required payload keys per local message kind (on top of v/t/src).
REQUIRED = {
    "det": ("frames",),
    "replay": ("path", "cam"),
    "pose": ("cam", "people"),
    "robot_state": ("pose", "battery"),
    "cmd": ("op",),
    "event": ("name",),
}


def envelope(src, **fields):
    d = {"v": V, "t": time.time(), "src": src}
    d.update(fields)
    return d


def pack(obj):
    return msgpack.packb(obj, use_bin_type=True)


def unpack(buf):
    return msgpack.unpackb(buf, raw=False)


def validate(msg, kind=None):
    """Raise ValueError if a message misses envelope or kind-specific keys."""
    if not isinstance(msg, dict):
        raise ValueError("message must be a map")
    for k in ("v", "t", "src"):
        if k not in msg:
            raise ValueError("missing envelope key %r" % k)
    if msg["v"] > V:
        raise ValueError("schema v%d newer than supported v%d" % (msg["v"], V))
    for k in REQUIRED.get(kind, ()):
        if k not in msg:
            raise ValueError("%s: missing %r" % (kind, k))
    return msg


def det_frame(cam, fn, ts, objs):
    """One camera frame inside a `det` message. objs: dicts with tid, cls, conf, bbox[, emb, gie]."""
    return {"cam": cam, "fn": fn, "ts": ts, "objs": objs}
