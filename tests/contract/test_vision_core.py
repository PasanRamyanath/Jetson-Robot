"""vision_core's portable C++ (§5.5): tracker/geometry/gates, and its det/ctrl wire format vs beni_common.schemas."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from beni_common import schemas as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GXX = shutil.which("g++")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if not GXX:
        pytest.skip("g++ not available")
    out = str(tmp_path_factory.mktemp("vc") / "vc_test.exe")
    src = os.path.join(ROOT, "jetson", "vision_core", "src")
    subprocess.check_call([GXX, "-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror", "-I", src,
                           "-I", os.path.join(ROOT, "shared"), os.path.join(ROOT, "tests", "contract", "cpp",
                                                                          "vision_core_test.cpp"),
                           os.path.join(src, "bytetrack.cpp"), "-o", out])
    return out


def test_portable_logic(exe):
    r = subprocess.run([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert r.returncode == 0, r.stderr.decode()
    assert r.stdout.strip() == b"ok"


def test_det_matches_python_schema(exe):
    msg = S.validate(S.unpack(subprocess.check_output([exe, "det"])), "det")
    assert msg["src"] == "vision" and [f["cam"] for f in msg["frames"]] == [0, 1]
    f0 = msg["frames"][0]
    assert (f0["fn"], f0["ts"]) == (42, 1234567890123) and msg["frames"][1]["objs"] == []
    p, face = f0["objs"]
    assert p == {"tid": 7, "cls": 0, "gie": 1, "conf": 0.912, "bbox": [10.0, 20.0, 50.5, 150.0]}
    assert (face["gie"], face["parent"], face["bbox"]) == (2, 7, [20.0, 25.0, 20.0, 25.0])
    e = np.frombuffer(face["emb"], np.float16).astype(np.float32)
    assert np.allclose(e, np.array([1, 2, 2, 4]) / 5.0, atol=1e-3)


@pytest.mark.parametrize("req,out", [
    ({"op": "snapshot", "req": 17, "cam": 1}, "op=2 cam=1 n=0 l=0.00 r=1.00 req=17 err="),
    ({"op": "interval", "n": 59}, "op=4 cam=0 n=59"),
    ({"op": "bitrate", "bps": 50}, "op=0 cam=0 n=50 l=0.00 r=1.00 req=0 err=bps out of range path="),
    ({"op": "ae_region", "cam": 1, "l": 0.25, "t": 0.1, "r": 0.5, "b": 0.6}, "op=5 cam=1 n=0 l=0.25 r=0.50"),
    ({"op": "snapshot", "cam": 3}, "err=bad cam path="),
    ({"op": "idr"}, "op=7 cam=0 n=0 l=0.00 r=1.00 req=0 err="),
    ({"op": "thumb", "req": 5, "l": 0.25, "t": 0.1, "r": 0.5, "b": 0.6}, "op=8 cam=0 n=320 l=0.25 r=0.50 req=5 err="),
    ({"op": "thumb", "n": 4096}, "err=n out of range path="),
    ({"op": "replay", "path": "/ssd/beni/rec/cam1_a.ts"}, "op=9 cam=0 n=6 l=0.00 r=0.00 req=0 err= path=/ssd/"),
    ({"op": "replay", "stop": 1}, "op=9 cam=0 n=6 l=0.00 r=1.00 req=0 err= path="),
    ({"op": "replay", "stride": 0}, "err=bad stride path="),
    ({"op": "pose", "cam": 1, "secs": 15}, "op=10 cam=1 n=0 l=0.00 r=15.00 req=0 err= path="),
    ({"op": "pose"}, "op=10 cam=0 n=0 l=0.00 r=10.00 req=0 err= path="),
    ({"op": "pose", "secs": 600}, "err=bad secs path="),
    ({"op": "cam", "cam": 0, "on": 0}, "op=11 cam=0 n=0 l=0.00 r=1.00 req=0 err= path="),
    ({"op": "cam", "cam": 1}, "op=11 cam=1 n=1 l=0.00 r=1.00 req=0 err= path="),
    ({"op": "warp"},"err=unknown op path="),
])
def test_ctrl_parse(exe, req, out):
    got = subprocess.run([exe, "ctrl"], input=S.pack(req), stdout=subprocess.PIPE, check=True).stdout.decode()
    assert got.startswith(out) or got.strip().endswith(out), got


def test_pose_matches_python_schema(exe):
    m = S.validate(S.unpack(subprocess.check_output([exe, "pose"])), "pose")
    a, b = m["people"]
    assert (m["cam"], m["fn"], a["tid"], b["tid"]) == (1, 4242, 4, 1000002)
    assert len(a["kps"]) == 54 and a["kps"][:3] == [10.0, 20.5, 0.5] and a["kps"][51:] == [27.0, 37.5, 0.5]
    assert b["kps"][:3] == [100.0, 50.0, 0.33] and not any(b["kps"][3:])


def test_replay_matches_python_schema(exe):
    raw = subprocess.check_output([exe, "replay"])
    n = int.from_bytes(raw[:4], "little")
    m = S.validate(S.unpack(raw[4:4 + n]), "replay")
    done = S.validate(S.unpack(raw[4 + n:]), "replay")
    assert (m["path"], m["cam"], m["pts"], m["fn"]) == ("/ssd/beni/rec/cam1_x.ts", 1, 12.35, 72)
    assert m["jpeg"] == b"\xff\xd8jpg"
    p, face = m["objs"]
    assert p["tid"] == 3 and face["parent"] == 3 and len(face["emb"]) == 8
    assert done["done"] is True and done["frames"] == 900 and done["ok"] is True and "objs" not in done


def _crc(b):
    c = 0xFFFFFFFF
    for x in b:
        c ^= x << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7 if c & 0x80000000 else c << 1) & 0xFFFFFFFF
    return c


def test_ts_mux(exe):
    ts = subprocess.check_output([exe, "ts"])
    assert len(ts) % 188 == 0
    pkts = [ts[i:i + 188] for i in range(0, len(ts), 188)]
    assert all(p[0] == 0x47 for p in pkts)
    cc, pes, pts, pcr = {}, [], [], 0
    for p in pkts:
        pid, pusi, afc, c = ((p[1] & 0x1F) << 8) | p[2], p[1] & 0x40, (p[3] >> 4) & 3, p[3] & 0xF
        if pid in cc:
            assert c == (cc[pid] + 1) & 0xF, "continuity"
        cc[pid] = c
        i = 4
        if afc & 2:
            if p[4] and p[5] & 0x10:
                pcr += 1
            i = 5 + p[4]
        body = p[i:]
        if pid in (0, 0x1000):
            n = ((body[2] & 0x0F) << 8 | body[3]) + 3
            sec = body[1:1 + n]
            assert _crc(sec) == 0, "section CRC"
            if pid == 0x1000:
                assert sec[12] == 0x24                 # HEVC stream type
            continue
        assert pid == 0x100
        if pusi:
            assert body[:4] == b"\x00\x00\x01\xe0"
            h = body[8]
            t = body[9:14]
            pts.append(((t[0] >> 1) & 7) << 30 | t[1] << 22 | (t[2] >> 1) << 15 | t[3] << 7 | t[4] >> 1)
            pes.append(bytearray(body[9 + h:]))
        else:
            pes[-1] += body
    assert pts == [90000, 93000, 96000] and pcr == 3
    assert [len(x) for x in pes] == [1006, 56, 161] and bytes(pes[0][6:]) == b"k" * 1000
