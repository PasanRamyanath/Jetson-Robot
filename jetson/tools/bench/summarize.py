#!/usr/bin/env python3
"""Summarise a tegrastats_logger CSV into markdown rows: mean / p95 per field (§14.3, §14.4). Python 3.6-safe."""
import csv
import sys

COLS = (("cpu_avg", "CPU avg %"), ("cpu_max", "CPU max core %"), ("gpu", "GPU %"), ("emc", "EMC %"),
        ("nvenc", "NVENC MHz"), ("nvdec", "NVDEC MHz"), ("ram_mb", "RAM MB"), ("temp_cpu", "CPU °C"),
        ("temp_gpu", "GPU °C"), ("p_in_mw", "Power in mW"), ("p_gpu_mw", "GPU mW"), ("p_cpu_mw", "CPU mW"))


def main():
    path, label = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "system"
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("| %s | samples | 0 | |" % label)
        return
    for key, name in COLS:
        xs = sorted(float(r[key]) for r in rows if r.get(key) not in (None, "", "None"))
        if xs:
            print("| %s | %s mean / p95 / max | %.0f / %.0f / %.0f | |"
                  % (label, name, sum(xs) / len(xs), xs[int(0.95 * (len(xs) - 1))], xs[-1]))
    print("| %s | samples | %d | |" % (label, len(rows)))


if __name__ == "__main__":
    main()
