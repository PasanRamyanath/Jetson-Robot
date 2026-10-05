"""Florence-2 helpers and the locate tool (§10.7) with a fake model and session."""
import asyncio
from types import SimpleNamespace

import pytest

from beni_brain import vision_tools as V
from beni_brain.tools import Tools


def test_bearing_and_side():
    assert V.bearing([600, 320, 680, 400], 1280, 720) == pytest.approx((0.0, 0.0))
    dpan, dtilt = V.bearing([0, 0, 2, 2], 1280, 720)
    assert dpan == pytest.approx(31.1, abs=0.1) and dtilt > 15          # top-left corner: left and up
    assert V.side(20) == "to the left" and V.side(-20) == "to the right" and V.side(3) == "straight ahead"


def test_boxes_normalises_both_result_shapes():
    od = {"bboxes": [[0, 0, 10, 10], [0, 0, 50, 50]], "bboxes_labels": ["cup", "red cup"], "polygons": []}
    assert [b[0] for b in V.boxes(od)] == ["red cup", "cup"]
    assert V.boxes({"bboxes": [[1, 2, 3, 4]], "labels": [""]}) == [("object", [1.0, 2.0, 3.0, 4.0])]
    assert V.boxes({}) == []


def test_load_stub_and_lazy_failure():
    assert isinstance(V.load(SimpleNamespace(stub=True, vision="x")), V.StubVision)

    def boom():
        raise ImportError("no transformers")
    lazy = V.Lazy(boom, wait=2.0)
    with pytest.raises(RuntimeError, match="unavailable"):
        lazy.find(b"", "cup")


class FakeSession:
    def __init__(self, found):
        vision = SimpleNamespace(find=lambda jpeg, phrase: (found, (1280, 720)))
        self.brain = SimpleNamespace(mirror=None, vision=vision)
        self.acts = []

    async def snapshot(self, cam, turn):
        return {"jpeg": b"\xff\xd8", "cam": cam}

    async def action(self, name, args, turn):
        self.acts.append((name, args))
        return {"ok": True}


def test_locate_turns_head_by_relative_bearing():
    s = FakeSession([("red cup", [1000, 300, 1100, 420])])
    out, imgs = asyncio.run(Tools(s).t_locate({"object": "red cup"}, 1, None))
    (name, args), = s.acts
    assert name == "look_at" and args["dpan"] < -10 and "pan" not in args and "to the right" in out and imgs == []
    s = FakeSession([("door", [0, 0, 100, 720])])
    asyncio.run(Tools(s).t_locate({"object": "door", "camera": "front"}, 1, None))
    assert s.acts[0][1]["pan"] > 20 and "dpan" not in s.acts[0][1]           # body camera: absolute neck angle


def test_locate_not_found():
    s = FakeSession([])
    out, _ = asyncio.run(Tools(s).t_locate({"object": "cat"}, 1, None))
    assert "can't see cat" in out and not s.acts
