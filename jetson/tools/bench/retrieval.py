#!/usr/bin/env python3
"""§14.3 memory-retrieval bench: N synthetic episodes in a temp DB, then p50/p99 of Retriever.retrieve. Python 3.8."""
import argparse
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared"))
from beni_common.memory import embed as E  # noqa: E402
from beni_common.memory.retrieval import Retriever  # noqa: E402
from beni_common.memory.store import MemoryStore  # noqa: E402

WORDS = ("kitchen tea mother school cricket rain keys phone garden dog rice curry homework birthday doctor bus "
         "office movie song temple market fish mango book charger sofa window lamp friend grandma exam").split()
PEOPLE = ["p1", "p2", "p3", "p4"]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--models", default="/ssd/beni/models")
    ap.add_argument("--tmp", default=None, help="scratch dir for the temp DB (default: system temp)")
    a = ap.parse_args()
    rng = random.Random(7)
    emb = E.load(a.models)
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18))) for _ in range(a.n)]
    t0 = time.perf_counter()
    vecs = [v for i in range(0, a.n, 64) for v in emb.encode(texts[i:i + 64])]
    t_emb = (time.perf_counter() - t0) * 1000.0 / a.n
    with tempfile.TemporaryDirectory(dir=a.tmp) as d:
        store = MemoryStore(os.path.join(d, "bench.db"))
        with store.tx():
            for t, v in zip(texts, vecs):
                store.add_episode(t, people=rng.sample(PEOPLE, rng.randint(1, 2)), importance=rng.uniform(1, 9), emb=v)
        t0 = time.perf_counter()
        r = Retriever(store, emb, dim=len(vecs[0]))
        t_load = time.perf_counter() - t0
        lat = []
        for _ in range(a.queries):
            q = "where did I leave the " + " ".join(rng.choice(WORDS) for _ in range(3))
            t0 = time.perf_counter()
            r.retrieve(q, speaker=rng.choice(PEOPLE), k=8, touch=False)
            lat.append((time.perf_counter() - t0) * 1000.0)
        store.db.close()
    name = type(emb).__name__
    print("| memory retrieval (%d eps, %s) | p50 / p99 ms per query | %.1f / %.1f | < 60 |"
          % (a.n, name, pct(lat, 50), pct(lat, 99)))
    print("| memory retrieval (%d eps, %s) | index load s; embed ms/doc | %.2f; %.2f | |" % (a.n, name, t_load, t_emb))


if __name__ == "__main__":
    main()
