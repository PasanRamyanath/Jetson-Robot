#!/usr/bin/env python3
"""ISP override tuning (§4 item 9, §5.1): edit /var/nvidia/nvcam/settings/camera_overrides.isp one key at a time.

Only when a lighting problem shows up in the logs (hunting AE, crushed shadows, the NoIR tint). Every change keeps a
timestamped backup and restarts nvargus-daemon + the vision service so it takes effect. Host Python 3.6.

    sudo isp_override.py show
    sudo isp_override.py set ae.MeanAlg.ConvergeSpeed 0.2
    sudo isp_override.py unset ae.MeanAlg.ConvergeSpeed
    sudo isp_override.py restore            # newest backup (or: restore <file>)
"""
import glob
import os
import re
import subprocess
import sys
import time

PATH = "/var/nvidia/nvcam/settings/camera_overrides.isp"
LINE = re.compile(r"^\s*([A-Za-z0-9_.\[\]]+)\s*=\s*(.*?)\s*;\s*$")
KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\[\]]*$")
SERVICES = ("nvargus-daemon", "beni-vision-core", "beni-vision")


def parse(text):
    """-> {key: value} for `key = value;` lines (comments and anything else are left alone)."""
    out = {}
    for ln in text.splitlines():
        m = LINE.match(ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def set_key(text, key, value):
    if not KEY.match(key) or ";" in value or "\n" in value:
        raise ValueError("bad key or value")
    lines, done = text.splitlines(), False
    for i, ln in enumerate(lines):
        m = LINE.match(ln)
        if m and m.group(1) == key:
            lines[i], done = "%s = %s;" % (key, value), True
    if not done:
        lines.append("%s = %s;" % (key, value))
    return "\n".join(lines) + "\n"


def unset_key(text, key):
    lines = [ln for ln in text.splitlines() if not (LINE.match(ln) and LINE.match(ln).group(1) == key)]
    return "\n".join(lines) + "\n" if lines else ""


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def backups(path=None):
    return sorted(glob.glob((path or PATH) + ".bak-*"))


def write(path, text):
    """Backup, then an atomic replace; the camera daemon reads the file world-readable."""
    if os.path.exists(path):
        bak, n = "%s.bak-%s" % (path, time.strftime("%Y%m%d-%H%M%S")), 1
        while os.path.exists(bak):                  # two edits in one second must not clobber a backup
            bak, n = "%s.bak-%s-%d" % (path, time.strftime("%Y%m%d-%H%M%S"), n), n + 1
        with open(path) as f, open(bak, "w") as b:
            b.write(f.read())
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)


def restart():
    for s in SERVICES:
        subprocess.call(["systemctl", "try-restart" if s != SERVICES[0] else "restart", s])


def main(argv):
    if not argv or argv[0] not in ("show", "set", "unset", "restore"):
        print(__doc__)
        return 2
    cur = _read(PATH)
    cmd = argv[0]
    if cmd == "show":
        for k, v in sorted(parse(cur).items()):
            print("%s = %s" % (k, v))
        print("(%d backups)" % len(backups()))
        return 0
    if cmd == "set" and len(argv) == 3:
        new = set_key(cur, argv[1], argv[2])
    elif cmd == "unset" and len(argv) == 2:
        new = unset_key(cur, argv[1])
    elif cmd == "restore":
        src = argv[1] if len(argv) > 1 else (backups() or [None])[-1]
        if not src:
            print("no backups")
            return 1
        new = _read(src)
    else:
        print(__doc__)
        return 2
    if new == cur:
        print("unchanged")
        return 0
    write(PATH, new)
    restart()
    print("written; nvargus-daemon and vision restarted. Watch `journalctl -u beni-vision-core -f` (scene lux/AE).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
