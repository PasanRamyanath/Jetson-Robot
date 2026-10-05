"""§10.6 recall: time_range narrows episodes to a window, falling back to the window's most important ones."""
import asyncio
import time
from types import SimpleNamespace

from beni_brain.tools import Tools, time_window

NOW = time.mktime((2026, 9, 30, 15, 0, 0, 0, 0, -1))           # a Wednesday afternoon


def test_time_window():
    today, yday = time_window("today", NOW), time_window(" Yesterday ", NOW)
    assert today == (NOW - 15 * 3600, NOW) and yday == (today[0] - 86400, today[0])
    assert time_window("last week", NOW)[1] == time_window("this week", NOW)[0] == today[0] - 2 * 86400
    assert time_window("past 3 days", NOW) == (NOW - 3 * 86400, NOW)
    assert time_window("the last few hours", NOW) == (NOW - 3 * 3600, NOW)
    assert time_window("last hour", NOW) == (NOW - 3600, NOW)
    assert time_window("whenever", NOW) is None and time_window(None, NOW) is None


class Store:
    def __init__(self):
        self.qs = []

    def q(self, sql, params):
        self.qs.append(params)
        return [{"t_end": time.time() - 600, "text": "fallback episode"}]


def test_recall_filters_by_window():
    now = time.time()
    eps = [{"t_start": now - 3 * 86400, "t_end": now - 3 * 86400 + 60, "text": "old trip"},
           {"t_start": now - 1200, "t_end": now - 1100, "text": "tea with Amma"}]
    store = Store()
    ret = SimpleNamespace(refresh=lambda: None, retrieve=lambda q, speaker=None, k=6: eps)
    mirror = SimpleNamespace(person_id=lambda n: None, store=store, retriever=ret)
    t = Tools(SimpleNamespace(brain=SimpleNamespace(mirror=mirror)))
    out, _ = asyncio.run(t.t_recall({"query": "tea"}, 1, None))
    assert "old trip" in out and "tea with Amma" in out
    out, _ = asyncio.run(t.t_recall({"query": "tea", "time_range": "past 2 hours"}, 1, None))
    assert "old trip" not in out and "tea with Amma" in out and not store.qs
    eps[1]["t_start"] = eps[1]["t_end"] = now - 5 * 86400
    out, _ = asyncio.run(t.t_recall({"query": "tea", "time_range": "today"}, 1, None))
    assert out.endswith("fallback episode") and len(store.qs) == 1


def test_reranker_blends_and_passes_through():
    from beni_brain.memory.rerank import Reranker
    r = Reranker.__new__(Reranker)
    r.model = None
    rows = [{"text": "a", "_score": 2.0}, {"text": "b", "_score": 1.9}]
    assert r("q", rows) is rows                                  # still loading: fused order untouched
    r.model = object()
    r.scores = lambda q, texts: [-6.0, 6.0]
    out = r("q", [dict(x) for x in rows])
    assert [x["text"] for x in out] == ["b", "a"] and out[0]["_rerank"] > 0.99
    r.scores = lambda q, texts: 1 / 0
    assert r("q", rows) is rows
