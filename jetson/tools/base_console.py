#!/usr/bin/env python3
"""Bench console for the ESP32 base over /dev/ttyTHS1 (§7.5). Python 3.6-safe, needs pyserial.

  base_console.py monitor            print decoded StateMsg at ~5 Hz + frame error count
  base_console.py drive V W [secs]   send cmd_vel (mm/s, mrad/s) at 20 Hz for secs (default 2), then stop
  base_console.py estop | clear      MSG_ESTOP with arg 1 / 0
"""
import os
import sys
import time

import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from beni_common import proto  # noqa: E402

PORT = os.environ.get("BENI_BASE_PORT", "/dev/ttyTHS1")


def monitor(s):
    r, last = proto.FrameReader(), 0.0
    while True:
        for typ, seq, payload in r.feed(s.read(s.in_waiting or 1)):
            if typ == proto.MSG_STATE and time.time() - last > 0.2:
                last = time.time()
                st = proto.parse_state(payload)
                print("t=%(t_ms)8d v=%(v_mm_s)5d w=%(w_mrad_s)5d x=%(x_m).3f y=%(y_m).3f yaw=%(yaw_rad).3f "
                      "batt=%(batt_mv)5dmV flags=0x%(flags)02x" % st, "range", st["range_cm"], "err", r.errors)
            elif typ in (proto.MSG_EVENT, proto.MSG_LOG):
                print("event/log", typ, payload)


def main(argv):
    s = serial.Serial(PORT, 921600, timeout=0.05)
    cmd = argv[0] if argv else "monitor"
    if cmd == "monitor":
        monitor(s)
    elif cmd == "drive":
        v, w = int(argv[1]), int(argv[2])
        end, seq = time.time() + (float(argv[3]) if len(argv) > 3 else 2.0), 0
        while time.time() < end:
            s.write(proto.cmd_vel(seq, v, w))
            seq += 1
            time.sleep(0.05)
        s.write(proto.cmd_vel(seq, 0, 0))
    elif cmd in ("estop", "clear"):
        s.write(proto.encode(proto.MSG_ESTOP, 0, bytes([1 if cmd == "estop" else 0])))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
