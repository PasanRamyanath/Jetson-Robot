"""§11.8 identity: the drift guard survives reloads, presence expires without frames, match bookkeeping throttles."""
import numpy as np

from beni_agent.perception import identity as I
from beni_common.memory import MemoryStore


def _unit(seed, dim=128):
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_gallery_drift_guard_and_touch(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"))
    pid = st.put("person", {"display_name": "Ana"})
    g = I.Gallery(st)
    g.add(pid, _unit(1), "enroll")
    auto = g.add(pid, _unit(2), "auto")
    g.reload()                                            # a sync apply or an eviction reloads the gallery
    assert auto in g.session_auto                         # ... and must not turn this session's auto-adds trusted
    g.touch(0, now=100.0)
    g.touch(0, now=110.0)                                 # within TOUCH_GAP_S: no second write
    g.touch(0, now=100.0 + I.TOUCH_GAP_S)
    assert st.q("SELECT match_count FROM face_exemplar WHERE id=?", (g.ids[0],))[0]["match_count"] == 2
    st.close()


def test_presence_expires_without_frames(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"))
    pid = st.put("person", {"display_name": "Ana"})
    g = I.Gallery(st)
    g.add(pid, _unit(1), "enroll")
    now = [0.0]
    ident = I.FaceIdentifier(g, learn=False, clock=lambda: now[0])
    face = {"cam": 0, "objs": [{"tid": 4, "bbox": [10, 10, 60, 100], "emb": _unit(1).astype(np.float16).tobytes()}]}
    for i in range(I.MIN_FRAMES):
        now[0] = i * 0.1
        ident.on_frame(face)
    assert ident.present() == [pid]
    now[0] += I.TRACK_TTL + 1                             # the camera stopped: no frame will sweep the track
    assert ident.present() == []
    st.close()
