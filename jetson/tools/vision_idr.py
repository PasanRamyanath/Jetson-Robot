#!/usr/bin/env python3
"""MediaMTX runOnRead hook (§5.3): ask the vision process for a teleop IDR so a new WebRTC viewer gets a picture
at once instead of after up to one GOP. Host Python 3.6; exits quietly if vision is down."""
import os
import sys

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from beni_common import schemas as S  # noqa: E402


def main():
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, 1000)
    s.setsockopt(zmq.SNDTIMEO, 1000)
    s.connect(S.EP["vision_ctrl"])
    try:
        s.send(S.pack({"op": "idr"}))
        s.recv()
    except zmq.Again:
        pass


if __name__ == "__main__":
    main()
