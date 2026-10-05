#!/usr/bin/env python3
"""Log tegrastats to CSV for plots (§14.4). Python 3.6-safe.  Usage: tegrastats_logger.py [out.csv] [interval_ms]"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from beni_common import tegrastats  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tegra.csv"
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=tegrastats.FIELDS)
        w.writeheader()
        try:
            for line in tegrastats.stream(interval):
                w.writerow(tegrastats.parse(line, round(time.time(), 3)))
                f.flush()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
