"""ISP override tool (§4 item 9): one key at a time, with a backup and restore."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("isp_override", os.path.join(ROOT, "jetson", "tools", "isp_override.py"))
I = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I)

SRC = "# tuned\nae.MeanAlg.ConvergeSpeed = 0.5;\n  awb.v4.LscFactor = 1.0 ;\n"


def test_parse_set_unset():
    assert I.parse(SRC) == {"ae.MeanAlg.ConvergeSpeed": "0.5", "awb.v4.LscFactor": "1.0"}
    t = I.set_key(SRC, "ae.MeanAlg.ConvergeSpeed", "0.2")
    assert I.parse(t)["ae.MeanAlg.ConvergeSpeed"] == "0.2" and t.startswith("# tuned\n")
    t = I.set_key(t, "ae.Mode", "1")
    assert I.parse(t)["ae.Mode"] == "1" and len(I.parse(t)) == 3
    assert "awb.v4.LscFactor" not in I.parse(I.unset_key(t, "awb.v4.LscFactor"))
    for k, v in (("bad key", "1"), ("a.b", "1; x = 2")):
        try:
            I.set_key(SRC, k, v)
            assert False
        except ValueError:
            pass


def test_backup_and_restore(tmp_path, monkeypatch):
    p = str(tmp_path / "nvcam" / "camera_overrides.isp")
    calls = []
    monkeypatch.setattr(I, "PATH", p)
    monkeypatch.setattr(I, "restart", lambda: calls.append(1))
    assert I.main(["set", "ae.Mode", "1"]) == 0 and I.parse(open(p).read()) == {"ae.Mode": "1"}
    assert I.backups() == []                                   # nothing to back up the first time
    assert I.main(["set", "ae.Mode", "1"]) == 0 and len(calls) == 1   # unchanged: no restart
    assert I.main(["set", "ae.Mode", "2"]) == 0 and len(I.backups()) == 1
    assert I.main(["restore"]) == 0 and I.parse(open(p).read()) == {"ae.Mode": "1"}
    assert I.main(["warp"]) == 2
