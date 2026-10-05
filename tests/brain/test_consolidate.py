"""§11.9 consolidation: day/week summaries + raw-text compression, reflection, confidence decay, routines."""
import asyncio
import json
import time

import numpy as np

from beni_brain.llm import StubLLM
from beni_brain.memory import consolidate as C
from beni_brain.memory.mirror import Mirror
from beni_common.memory import routines
from beni_common.memory.embed import HashEmbedder

DAY = 86400.0
NOW = time.mktime((2026, 9, 30, 15, 0, 0, 0, 0, -1))           # a Wednesday afternoon


class ReflectLLM(StubLLM):
    async def json(self, prompt, schema, max_tokens=512, system=None):
        if "insights" not in schema["properties"]:
            return await super().json(prompt, schema, max_tokens, system)
        return {"insights": [{"text": "Nimal loves cricket.", "evidence": [1, 2, 99], "predicate": "likes",
                              "object": "cricket"},
                             {"text": "Nimal seems stressed about work.", "evidence": [3], "predicate": "health_stress",
                              "object": "work"},
                             {"text": "", "evidence": []}]}


def _mirror(tmp_path):
    m = Mirror(str(tmp_path / "k.db"), HashEmbedder())
    m.store.put("person", {"id": "p1", "display_name": "Nimal"})
    return m


def test_day_week_summaries_and_drop_raw(tmp_path):
    st = _mirror(tmp_path).store
    monday = C._day0(NOW) - 2 * DAY
    for d in range(-14, 2):                                     # two full weeks + this Monday/Tuesday
        for h in (9, 13, 19):
            st.add_episode("day %d at %d" % (d, h), kind="conversation", people=["p1"], importance=3 + (h == 19),
                           t_start=monday + d * DAY + h * 3600, t_end=monday + d * DAY + h * 3600 + 60)
    st.add_episode("lonely", kind="observation", t_start=monday - 20 * DAY, t_end=monday - 20 * DAY)
    n = asyncio.run(C.summarise(st, StubLLM(), None, NOW, max_new=30))
    days = st.q("SELECT * FROM episode WHERE kind='summary' AND t_end - t_start < ? ORDER BY t_start", (2 * DAY,))
    weeks = st.q("SELECT * FROM episode WHERE kind='summary' AND t_end - t_start > ? ORDER BY t_start", (2 * DAY,))
    assert len(days) == 16 and len(weeks) == 2 and n == 18   # the 1-episode day is left alone
    assert days[0]["t_start"] == monday - 14 * DAY and days[0]["importance"] == 4
    assert json.loads(days[0]["people"]) == ["p1"] and len(json.loads(days[0]["parent_ids"])) == 3
    assert weeks[0]["t_start"] == monday - 14 * DAY and weeks[1]["t_end"] == monday
    assert len(json.loads(weeks[1]["parent_ids"])) == 7
    assert asyncio.run(C.summarise(st, StubLLM(), None, NOW)) == 0          # idempotent

    st.put("episode", {"kind": "conversation_extracted", "text": "USER: hi\nBENI: hello", "t_start": NOW - 40 * DAY,
                       "t_end": NOW - 40 * DAY})
    later = NOW + 20 * DAY                                      # days -14..-11 (Mon) are now > 30 days old
    dropped = C.drop_raw(st, later)
    assert dropped == 1 + len([1 for d in range(-14, 2) for h in (9, 13, 19)
                               if monday + d * DAY + h * 3600 + 60 < later - 30 * DAY])
    assert st.q("SELECT count(*) AS n FROM episode WHERE deleted=0 AND text='lonely'")[0]["n"] == 1
    assert st.q("SELECT count(*) AS n FROM episode WHERE deleted=0 AND kind='summary'")[0]["n"] == 18


def test_reflection(tmp_path):
    m = _mirror(tmp_path)
    st = m.store
    for i in range(8):
        st.add_episode("talked about cricket %d" % i, people=["p1"], importance=6, t_start=NOW - i, t_end=NOW - i)
    assert asyncio.run(C.reflect(m, StubLLM(), NOW, threshold=100)) == 0          # below the threshold
    assert asyncio.run(C.reflect(m, ReflectLLM(), NOW)) == 1
    refl = st.q("SELECT * FROM episode WHERE kind='reflection' ORDER BY text")
    assert len(refl) == 2 and len(json.loads(refl[0]["parent_ids"])) == 2
    facts = {f["predicate"]: f for f in st.q("SELECT * FROM fact WHERE deleted=0")}
    assert facts["likes"]["confidence"] == 0.5 and facts["likes"]["source_kind"] == "inferred"
    assert facts["likes"]["status"] == "active" and facts["health_stress"]["status"] == "pending_confirm"
    assert st.kv_get("reflect.last") == {"p1": NOW}
    assert asyncio.run(C.reflect(m, ReflectLLM(), NOW + 1)) == 0                  # nothing new since
    cards = asyncio.run(C.rewrite_cards(st, StubLLM(), NOW - 10))
    assert cards == 1 and "(insight) Nimal loves cricket." in st.get("person", "p1")["card"] + "".join(
        r["text"] for r in refl)


def test_card_follows_retracted_facts(tmp_path):
    """Retracting a fact rewrites the card; retracting the last one clears it (the card is in every prompt)."""
    st = _mirror(tmp_path).store
    st.put("person", {"id": "p1", "display_name": "Nimal"})
    tea = st.add_fact("p1", "likes", "tea", "Nimal likes tea")
    st.add_fact("p1", "plays", "chess", "Nimal plays chess")
    assert asyncio.run(C.rewrite_cards(st, StubLLM(), 0)) == 1 and "tea" in st.get("person", "p1")["card"]
    since = time.time()
    time.sleep(0.002)
    st.put("fact", {"id": tea, "status": "retracted"})              # "forget that I like tea"
    assert asyncio.run(C.rewrite_cards(st, StubLLM(), since)) == 1
    assert "tea" not in st.get("person", "p1")["card"]
    since = time.time()
    time.sleep(0.002)
    for f in st.facts_about("p1"):
        st.put("fact", {"id": f["id"], "status": "retracted"})
    assert asyncio.run(C.rewrite_cards(st, StubLLM(), since)) == 1 and st.get("person", "p1")["card"] is None


def test_confidence_decay(tmp_path):
    st = _mirror(tmp_path).store
    a = st.add_fact("p1", "likes", "tea", "Nimal likes tea", confidence=0.7)
    b = st.add_fact("p1", "birthday", "12 May", "Nimal's birthday is 12 May", confidence=0.9)
    c = st.add_fact("p1", "likes", "rain", "Nimal likes rain", confidence=0.35)
    later = time.time() + 91 * DAY
    st.hlc._clock = lambda: later                               # the decay writes happen "then"
    assert C.decay_confidence(st, later) == 2
    assert st.get("fact", a)["confidence"] == 0.6 and st.get("fact", b)["confidence"] == 0.9
    assert st.get("fact", c)["confidence"] == 0.3
    assert C.decay_confidence(st, later) == 0                   # the decay was a write: next step in 90 days


def test_routines(tmp_path):
    st = _mirror(tmp_path).store
    for w in range(3):
        for k in range(3):                                      # Wednesdays 19:xx, three sightings = one count
            t = NOW - w * 7 * DAY + 4 * 3600 + k * 600
            st.add_episode("Nimal arrived", kind="sighting", people=["p1"], importance=1.5, t_start=t, t_end=t)
        st.add_episode("chat", kind="conversation", people=["p1"], t_start=NOW - w * 7 * DAY, t_end=NOW - w * 7 * DAY)
    assert routines.update(st, NOW + 5 * 3600, since=NOW - 30 * DAY) == 2
    seen, mean = routines.level(st, "p1", "seen", NOW + 4 * 3600)
    assert seen == 3.0 and abs(mean - 3.0 / 168) < 1e-6
    assert routines.level(st, "p1", "talk", NOW)[0] == 3.0 and routines.level(st, "p2", "talk", NOW) == (0.0, 0.0)
    h = np.frombuffer(st.get("routine", routines.rid("p1", "seen"))["hist"], np.float32)
    assert h.shape == (168,) and h[routines.hour_of_week(NOW + 4 * 3600)] == 3.0
    t = NOW + 7 * DAY + 4 * 3600
    st.add_episode("Nimal arrived", kind="sighting", people=["p1"], t_start=t, t_end=t)
    routines.update(st, t + 60)                                 # a week later: old counts x0.97, +1
    assert abs(routines.level(st, "p1", "seen", t)[0] - (3.0 * 0.97 ** ((t + 60 - NOW - 5 * 3600) / (7 * DAY)) + 1)) \
        < 1e-4


def test_run_includes_new_steps(tmp_path):
    m = _mirror(tmp_path)

    class Ex:
        async def backlog(self):
            return 0
    stats = asyncio.run(C.run(m, Ex(), StubLLM()))
    for k in ("decayed", "routines", "raw_dropped", "summaries", "reflections", "cards"):
        assert k in stats


def test_rules_always_in_context(tmp_path):
    """§11.11: rule facts reach the prompt for any query, present people's first, even past the char budget."""
    from beni_common.memory.retrieval import format_context
    m = _mirror(tmp_path)
    st = m.store
    st.add_fact("p2", "rule", "room", "Kamal doesn't want Beni in his room.")
    st.add_fact("p1", "rule", "language", "Always speak Sinhala with Nimal.")
    st.add_fact("p1", "likes", "tea", "Nimal likes tea.")
    old = st.add_fact("home", "rule", "x", "An old rule.")
    st.put("fact", {"id": old, "status": "superseded"})
    ctx = m.retriever.context("what's the weather", people=["p1"], budget_chars=10)
    assert ctx["rules"] == ["Always speak Sinhala with Nimal.", "Kamal doesn't want Beni in his room."]
    assert ctx["facts"] == []                                   # budget exhausted, rules still there
    assert format_context(ctx).startswith("## House rules (always follow)\n- Always speak Sinhala")


def test_prune_exemplar_outliers(tmp_path):
    from beni_common.memory import to_blob
    st = _mirror(tmp_path).store
    rng = np.random.RandomState(0)
    base = rng.randn(128).astype(np.float32)
    ids = [st.put("face_exemplar", {"person_id": "p1", "emb": to_blob(base + 0.3 * rng.randn(128)),
                                    "source": "enroll" if i < 4 else "auto"}) for i in range(8)]
    bad_auto = st.put("face_exemplar", {"person_id": "p1", "emb": to_blob(rng.randn(128)), "source": "auto"})
    bad_enroll = st.put("face_exemplar", {"person_id": "p1", "emb": to_blob(rng.randn(128)), "source": "enroll"})
    assert C.prune_exemplars(st) == 1
    assert st.get("face_exemplar", bad_auto) is None and st.get("face_exemplar", bad_enroll) is not None
    assert all(st.get("face_exemplar", i) for i in ids)
