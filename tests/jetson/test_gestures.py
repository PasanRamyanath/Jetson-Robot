"""Gestures (§4 item 39): wave / point from TRT-Pose keypoints, and on-demand pose requests."""
import asyncio

from beni_agent.perception import gestures as G
from beni_common import schemas as S


def body(**parts):
    """Standing person facing the camera, arms down; parts: index -> (x, y) overrides."""
    k = [0.0] * 54
    base = {G.NOSE: (255, 70), G.L_SH: (240, 100), G.R_SH: (270, 100), G.L_EL: (238, 140), G.R_EL: (272, 140),
            G.L_WR: (236, 180), G.R_WR: (274, 180), G.NECK: (255, 100)}
    base.update(parts)
    for i, (x, y) in base.items():
        k[3 * i:3 * i + 3] = [float(x), float(y), 0.6]
    return k


def run(frames, dt=0.2):
    now, got = [100.0], []
    g = G.Gestures(None, lambda *a: got.append(a), clock=lambda: now[0])
    for kps in frames:
        g.on_msg(S.T_POSE, {"cam": 0, "fn": 1, "people": [{"tid": 7, "kps": kps}]})
        now[0] += dt
    return got


def test_reversals():
    assert G.reversals([0, 10, 0, 10, 0], 5) == 3
    assert G.reversals([0, 2, 1, 3, 2, 4], 5) == 0             # jitter, and a slow drift, are not swings
    assert G.reversals([0, 20, 18, 21, 0], 5) == 1


def test_wave():
    xs = [280, 300, 281, 299, 280, 300]
    got = run([_wave(x) for x in xs])
    assert got == [("wave", 0, 7, {})]
    assert run([_wave(290)] * 6) == []                          # a raised, still hand is not a wave


def _wave(x):
    k = body()
    k[3 * G.R_EL:3 * G.R_EL + 2] = [285.0, 80.0]
    k[3 * G.R_WR:3 * G.R_WR + 2] = [float(x), 60.0]
    return k


def test_point_needs_a_straight_held_arm():
    out = body()
    out[3 * G.R_EL:3 * G.R_EL + 2] = [300.0, 100.0]
    out[3 * G.R_WR:3 * G.R_WR + 2] = [332.0, 104.0]
    got = run([out] * 5)
    assert len(got) == 1 and got[0][0] == "point" and got[0][3]["dir"] == "right"
    assert abs(got[0][3]["x"] - 332 / 512) < 1e-3
    left = body()
    left[3 * G.L_EL:3 * G.L_EL + 2] = [210.0, 100.0]
    left[3 * G.L_WR:3 * G.L_WR + 2] = [178.0, 98.0]
    assert run([left] * 5)[0][3]["dir"] == "left"
    assert run([out] * 2) == []                                 # not held long enough
    bent = body()
    bent[3 * G.R_EL:3 * G.R_EL + 2] = [300.0, 130.0]
    bent[3 * G.R_WR:3 * G.R_WR + 2] = [310.0, 100.0]
    assert run([bent] * 5) == []
    assert run([body()] * 10) == []


def test_cooldown_and_bad_frames():
    out = body()
    out[3 * G.R_EL:3 * G.R_EL + 2] = [300.0, 100.0]
    out[3 * G.R_WR:3 * G.R_WR + 2] = [332.0, 104.0]
    assert len(run([out] * 15)) == 1                            # 3 s of pointing: once (4 s cooldown)
    assert len(run([out] * 25)) == 2
    assert run([[0.0] * 10] * 5) == []


def test_watch_rate_limit_and_off():
    now, reqs = [0.0], []
    replies = [{"ok": True}, {"ok": False, "err": "pose off"}]

    async def request(ep, msg, timeout=2.0):
        reqs.append(msg)
        return replies[0]
    g = G.Gestures(request, None, clock=lambda: now[0])

    async def go():
        assert await g.watch(0, 20) is True
        assert await g.watch(0, 20) is False                    # still covered
        now[0] = 15.0
        assert await g.watch(0, 20) is True                     # less than half left: extend
        replies.pop(0)
        assert await g.watch(1, 10) is False
        now[0] = 100.0
        assert await g.watch(1, 10) is False and len(reqs) == 3  # engine missing: stop asking for a while
    asyncio.run(go())
    assert reqs[0] == {"op": "pose", "cam": 0, "secs": 20}
