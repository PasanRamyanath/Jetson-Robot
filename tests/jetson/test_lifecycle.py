"""§10.4 lifecycle: the pushed kernel carries the owner's ids, the venv's kaggle CLI is found, the budget holds."""
import json
import os
import shutil
from types import SimpleNamespace

from beni_agent.cloud import lifecycle as L

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lc(**kw):
    cfg = SimpleNamespace(kaggle_kernel="alice/beni-brain", kaggle_dir=os.path.join(ROOT, "kaggle"),
                          awake_hours=[], weekly_budget_h=28.0)
    cfg.__dict__.update(kw)
    kv = {}
    return L.Lifecycle(cfg, lambda k, d=None: kv.get(k, d), kv.__setitem__, clock=lambda: 1.7e9)


def test_render_rewrites_ids():
    d = _lc()._render()
    try:
        with open(os.path.join(d, "kernel-metadata.json")) as f:
            meta = json.load(f)
        assert meta["id"] == "alice/beni-brain"
        assert meta["dataset_sources"] == ["alice/beni-wheelhouse"]
        assert os.path.exists(os.path.join(d, meta["code_file"]))
    finally:
        shutil.rmtree(d)


def test_kaggle_cli_resolves_next_to_the_interpreter():
    # systemd runs /opt/beni/.venv38/bin/python without the venv on PATH
    assert L.KAGGLE == "kaggle" or os.path.exists(L.KAGGLE)


def test_budget_and_on_demand():
    lc = _lc(weekly_budget_h=1.0)
    assert not lc.should_run()                       # no awake windows
    assert lc.ensure_running(30) and lc.should_run()  # wake word: cold start, kept up
    assert lc._kick.is_set()                         # ... and the poll loop pushes now, not in POLL_S
    lc.kv_set(lc.week_key(), 3600)
    assert not lc.should_run() and not lc.ensure_running(30)
