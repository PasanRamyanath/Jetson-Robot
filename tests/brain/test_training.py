"""§11.12 without a GPU: feedback retargeting, the SFT replay mix, DPO pair building, and the eval gate."""
import asyncio
import json
from types import SimpleNamespace

from beni_common.memory import MemoryStore

from beni_brain.memory.extract import Extractor
from beni_brain.training import data, gate, seed


def _store(tmp_path):
    return MemoryStore(str(tmp_path / "m.db"), node="test")


def test_feedback_targets_previous_turn(tmp_path):
    s = _store(tmp_path)
    ex = Extractor(None, SimpleNamespace(store=s))
    prev = s.put("feedback", {"t": 1.0, "kind": "turn", "prompt": "my dog?", "response": "<emo=happy> Rocco!",
                              "signal": 0.0})
    ex._feedback({"kind": "correction", "about": "dog name"}, "No, it's Rocky.", "Sorry!", "p1", 2.0, "ep", prev)
    assert s.get("feedback", prev)["signal"] == -1.0
    fix, = s.q("SELECT * FROM feedback WHERE kind='correction'")
    assert (fix["prompt"], fix["response"], fix["better_response"]) == ("my dog?", "<emo=happy> Rocco!",
                                                                        "No, it's Rocky.")
    ex._feedback({"kind": "praise"}, "Good job", "Thanks", "p1", 3.0, "ep", None)         # no previous turn
    assert s.q("SELECT signal FROM feedback WHERE kind='praise'")[0]["signal"] == 1.0


def test_sft_mix_ratios(tmp_path):
    s = _store(tmp_path)
    for i in range(60):
        s.put("feedback", {"t": float(i), "kind": "turn", "prompt": "q%d" % i, "response": "<emo=calm> a%d" % i,
                           "signal": -0.5 if i % 10 == 0 else 0.0})
    s.put("feedback", {"t": 99.0, "kind": "turn", "prompt": "x", "response": "no tag", "signal": 0.0})
    rows, n = data.sft(s, since=29.5)
    assert n == 27                                        # t 30..59 minus 3 barge-ins; the untagged row is skipped
    users = [r["messages"][1]["content"] for r in rows]
    core = {u for u, _ in seed.CORE}
    assert sum(u in core for u in users) == round(27 * 0.6)
    assert sum(u.startswith("q") and int(u[1:]) < 30 for u in users) == round(27 * 0.4)
    assert all(r["messages"][0]["role"] == "system" and r["messages"][2]["role"] == "assistant" for r in rows)


def test_dpo_pairs(tmp_path):
    s = _store(tmp_path)
    s.put("feedback", {"t": 5.0, "kind": "correction", "prompt": "my dog?", "response": "<emo=happy> Rocco!",
                       "better_response": "No, it's Rocky."})
    s.put("feedback", {"t": 6.0, "kind": "correction", "prompt": "a", "response": "b", "better_response": "c"})

    class L:
        async def complete(self, prompt, max_tokens=300):
            return "<emo=apologetic> Sorry, Rocky!" if "Rocco" in prompt else "untagged"
    rows = asyncio.run(data.dpo(s, 0, L()))
    assert len(rows) == 1 and rows[0]["chosen"][0]["content"].startswith("<emo=apologetic>")
    assert rows[0]["rejected"][0]["content"] == "<emo=happy> Rocco!" and rows[0]["prompt"][-1]["content"] == "my dog?"


def test_eval_suite_shape():
    items = seed.eval_items()
    cats = {i["cat"] for i in items}
    assert cats == {"tool", "persona", "recall", "safety", "json", "language"} and len(items) >= 110
    assert not {u for u, _ in seed.CORE} & {i["messages"][-1]["content"] for i in items}   # held out


def test_checks():
    tool = {"cat": "tool", "expect": "move_to"}
    assert gate.check(tool, "", ["move_to"]) and not gate.check(tool, "<emo=happy> Okay!", [])
    p = {"cat": "persona"}
    assert gate.check(p, "<emo=happy> Sure! Here's one.", [])
    assert not gate.check(p, "Sure!", [])                                          # no tag
    assert not gate.check(p, "<emo=happy> Fruits:\n- mango\n- banana", [])         # markdown list
    assert not gate.check(p, "<emo=happy> One. Two. Three. Four.", [])             # too long
    j = {"cat": "json"}
    assert gate.check(j, '```json\n{"facts": [], "importance": 2, "episode_summary": "x"}\n```', [])
    assert not gate.check(j, '{"facts": []}', [])
    assert gate.check({"cat": "safety"}, "<emo=calm> Sorry, I can't help with that.", [])
    assert gate.check({"cat": "recall", "any": ["rocky"]}, "<emo=happy> His name is Rocky!", [])


def test_evaluate_and_verdict():
    class C:
        def __init__(self, good):
            self.good = good
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, model, messages, tools=None, **kw):
            user = messages[-1]["content"]
            item = next(i for i in seed.eval_items() if i["messages"][-1]["content"] == user)
            calls, text = [], "<emo=calm> I don't know."
            if self.good and item["cat"] == "tool":
                calls = [SimpleNamespace(function=SimpleNamespace(name=item["expect"]))]
            elif self.good and item["cat"] == "json":
                text = json.dumps({"facts": [], "importance": 1, "episode_summary": "s"})
            msg = SimpleNamespace(content=text, tool_calls=calls)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    good, _ = asyncio.run(gate.evaluate(C(True), "new"))
    bad, fails = asyncio.run(gate.evaluate(C(False), "base"))
    assert good["tool"] == 1.0 and bad["tool"] == 0.0 and good["json"] == 1.0 and fails
    assert gate.verdict(bad, good)[0] and not gate.verdict(good, bad)[0]
