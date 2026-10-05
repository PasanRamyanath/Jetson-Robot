"""Skill library (§11.9 step 11): action logs -> mined skills -> prompt lines; failing skills are demoted."""
import json

from beni_common.memory import MemoryStore, skills

CHECK = [{"tool": "move_to", "args": {"place": "kitchen"}, "ok": True},
         {"tool": "take_snapshot", "args": {"camera": "head"}, "ok": True},
         {"tool": "set_expression", "args": {"expression": "happy"}, "ok": True}]


def test_trigger_key_and_log(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"), node="kaggle")
    assert skills.trigger_key("Beni, could you check the kitchen?") == "check kitchen"
    assert skills.log_action(st, "please remember this", [{"tool": "remember", "args": {}, "ok": True}]) is None
    eid = skills.log_action(st, "Check the kitchen", CHECK, speaker="p1", now=100.0)
    e = st.get("episode", eid)
    assert e["kind"] == "action" and e["people"] == '["p1"]' and "move_to(place=kitchen)" in e["text"]
    assert json.loads(e["media_ref"]) == {"key": "check kitchen", "ok": True, "steps": [
        ["move_to", {"place": "kitchen"}], ["take_snapshot", {"camera": "head"}]]}


def test_mine_promote_count_demote(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"), node="kaggle")
    t = 1000000.0
    for i, text in enumerate(("check the kitchen", "Beni check the kitchen please", "could you check kitchen")):
        skills.log_action(st, text, CHECK, now=t + i)
    skills.log_action(st, "check the kitchen", CHECK[:1], now=t + 5)          # one step: not a skill
    assert skills.mine(st, now=t + 10) == {"new": 1, "updated": 0, "demoted": 0}
    (s,) = st.q("SELECT * FROM skill WHERE deleted=0")
    assert (s["trigger"], s["success"], s["fail"]) == ("check kitchen", 3, 0)
    assert skills.mine(st, now=t + 11) == {"new": 0, "updated": 0, "demoted": 0}     # idempotent
    (line,) = skills.relevant(st, "hey beni, check the kitchen")
    assert line.startswith("When asked to 'check kitchen': move_to(place=kitchen); take_snapshot") and "3/3" in line
    assert skills.relevant(st, "what's the weather") == []

    skills.log_action(st, "check the kitchen", CHECK, now=t + 20)
    assert skills.mine(st, now=t + 21)["updated"] == 1 and st.get("skill", s["id"])["success"] == 4
    bad = [dict(CHECK[0], ok=False)]
    for i in range(4):
        skills.log_action(st, "check the kitchen", bad, now=t + 30 + i)
    assert skills.mine(st, now=t + 40)["demoted"] == 1 and not st.q("SELECT id FROM skill WHERE deleted=0")
    assert skills.mine(st, now=t + 50)["new"] == 0                               # demoted stays demoted
