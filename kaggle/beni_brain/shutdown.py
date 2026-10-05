"""python -m beni_brain.shutdown [--timeout 420]: ask the gateway to stop cleanly (SIGTERM via its pid file) and wait.

Called by the notebook at the 11.5 h mark so memory deltas are flushed before Kaggle kills the VM at 12 h.
"""
import argparse
import os
import signal
import sys
import time

from .config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=420)
    a = ap.parse_args(argv)
    pid_file = os.path.join(Config.from_env().run_dir, "gateway.pid")
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        print("gateway not running")
        return 0
    end = time.time() + a.timeout
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except OSError:
            print("gateway stopped")
            return 0
        time.sleep(1)
    print("gateway did not stop in time")
    return 1


if __name__ == "__main__":
    sys.exit(main())
