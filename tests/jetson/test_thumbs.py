"""Memory thumbnails (§3.17): face crops on exemplars, sighting keyframes, the orphan / budget sweep."""
import asyncio
import os
import time

import numpy as np

from beni_agent.perception import thumbs as TH
from beni_agent.perception.identity import FaceIdentifier, Gallery
from beni_common.memory import MemoryStore


def _store(tmp_path):
    return MemoryStore(str(tmp_path / "m.db"), node="jetson")


def test_face_box():
    b = TH.face_box([256, 144, 32, 36], pad=1.0)
    assert abs(b["l"] - 0.5) < 1e-6 and abs(b["t"] - 0.5) < 1e-6
    assert abs(b["r"] - 288 / 512) < 1e-6 and abs(b["b"] - 180 / 288) < 1e-6
    e = TH.face_box([500, 280, 12, 8])                     # clamped to the frame
    assert e["r"] == 1.0 and e["b"] == 1.0


def test_face_and_sighting(tmp_path):
    st = _store(tmp_path)
    calls = []

    async def grab(cam, op, **kw):
        calls.append((cam, op, kw))
        return {"jpeg": b"\xff\xd8jpg"}

    now = [1000.0]
    th = TH.Thumbs(st, grab, str(tmp_path / "thumbs"), clock=lambda: now[0])
    eid = st.put("face_exemplar", {"person_id": "p1", "emb": b"\0\0"})

    async def go():
        p = await th.face(eid, 1, [100, 50, 20, 24])
        e1 = await th.sighting("p1", "Ana", 0, "kitchen")
        e2 = await th.sighting("p1", "Ana")                 # within 15 min: no second episode
        now[0] += TH.SIGHTING_GAP_S + 1
        e3 = await th.sighting("p1", "Ana")
        return p, e1, e2, e3

    p, e1, e2, e3 = asyncio.run(go())
    assert calls[0][:2] == (1, "thumb") and calls[0][2]["n"] == TH.FACE_N and "l" in calls[0][2]
    assert calls[1][2] == {"n": TH.KEY_N}
    assert st.get("face_exemplar", eid)["thumb_path"] == p and open(p, "rb").read() == b"\xff\xd8jpg"
    ep = st.get("episode", e1)
    assert ep["kind"] == "sighting" and ep["text"] == "Saw Ana in the kitchen." and os.path.exists(ep["keyframe_path"])
    assert e2 is None and e3


def test_no_thumb_when_vision_says_no(tmp_path):
    async def grab(cam, op, **kw):
        return None

    st = _store(tmp_path)
    th = TH.Thumbs(st, grab, str(tmp_path / "thumbs"))
    eid = asyncio.run(th.sighting("p1", "Ana"))
    assert st.get("episode", eid)["keyframe_path"] is None


def test_sweep(tmp_path):
    root = tmp_path / "thumbs" / "20260101"
    root.mkdir(parents=True)
    old = time.time() - 2 * TH.ORPHAN_AGE_S
    paths = {}
    for n in ("face-a", "face-orphan", "key-1", "key-2", "key-new-orphan"):
        p = root / (n + ".jpg")
        p.write_bytes(b"x" * 1000)
        if n != "key-new-orphan":
            os.utime(str(p), (old, old))
        paths[n] = str(p)
    st = _store(tmp_path)
    st.put("face_exemplar", {"person_id": "p", "emb": b"", "thumb_path": paths["face-a"]})
    st.add_episode("a", keyframe_path=paths["key-1"])
    st.add_episode("b", keyframe_path=paths["key-2"])
    gone = st.add_episode("c", keyframe_path=paths["face-orphan"])
    st.tombstone("episode", gone)
    th = TH.Thumbs(st, None, str(tmp_path / "thumbs"))
    th.keep = 3500                                          # bytes: forces one referenced keyframe out
    assert th.sweep_files(th.referenced()) == 2
    left = sorted(os.listdir(str(root)))
    assert "face-orphan.jpg" not in left and "face-a.jpg" in left and "key-new-orphan.jpg" in left
    assert len([n for n in left if n.startswith("key-")]) == 2


def test_identity_thumb_hook(tmp_path):
    st = _store(tmp_path)
    ident = FaceIdentifier(Gallery(st, "face_exemplar"))
    got = []
    ident.on_exemplar = lambda eid, cam, bbox: got.append((eid, cam, bbox))
    ident.start_enroll()
    rng = np.random.default_rng(0)
    for i in range(8):
        v = rng.standard_normal(128).astype(np.float32)
        v /= np.linalg.norm(v)
        ident.on_frame({"cam": 1, "objs": [{"tid": -1, "parent": 3, "gie": 2, "bbox": [100, 50, 40, 48],
                                            "emb": v.astype(np.float16).tobytes()}]})
    assert ident.finish_enroll("p1", k=4) == 4
    assert len(got) == 1 and got[0][1:] == (1, [100, 50, 40, 48])
    assert st.get("face_exemplar", got[0][0])["person_id"] == "p1"
