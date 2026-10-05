"""Training data from the mirror (§11.12). Only style (SFT) and preferences (DPO); facts stay in the DB.

SFT: logged `turn` feedback rows that were not barged in or complained about, mixed 50 % new / 30 % core persona /
20 % older rows (safeguard 1). DPO: `correction` rows -> (prompt, chosen = the reply rewritten with the user's
correction, rejected = what Beni said). "New" means newer than the last published adapter's `trained_until`.
"""
import json
import random

from . import seed

MIN_SFT, MIN_DPO = 300, 100

REWRITE = """A user corrected a home robot's spoken reply. Write the reply the robot should have given: 1-3 short \
spoken sentences, starting with an <emo=X> tag like the original, fixing exactly what the user pointed out and \
nothing else. Output only the reply.
USER: {prompt}
ROBOT (wrong): {response}
USER'S CORRECTION: {correction}"""


def _chat(user, reply=None):
    m = [{"role": "system", "content": seed.system()}, {"role": "user", "content": user}]
    return m + [{"role": "assistant", "content": reply}] if reply is not None else m


def turns(store, since):
    rows = store.q("SELECT t, prompt, response FROM feedback WHERE kind='turn' AND deleted=0 AND signal >= 0 "
                   "AND prompt IS NOT NULL AND response LIKE '<emo=%' ORDER BY t")
    return [r for r in rows if r["t"] > since], [r for r in rows if r["t"] <= since]


def sft(store, since, rng=None):
    """-> (rows [{messages}], number of new rows)."""
    rng = rng or random.Random(0)
    new, old = turns(store, since)
    n = len(new)
    core = list(seed.CORE)
    rng.shuffle(core)
    rows = [_chat(r["prompt"], r["response"]) for r in new]
    rows += [_chat(*core[i % len(core)]) for i in range(round(n * 0.6))]                 # 30 : 50
    rows += [_chat(r["prompt"], r["response"]) for r in rng.sample(old, min(len(old), round(n * 0.4)))]   # 20 : 50
    rng.shuffle(rows)
    return [{"messages": m} for m in rows], n


def corrections(store, since):
    return store.q("SELECT t, prompt, response, better_response FROM feedback WHERE kind='correction' AND deleted=0 "
                   "AND t > ? AND prompt IS NOT NULL AND response IS NOT NULL AND better_response IS NOT NULL "
                   "ORDER BY t", (since,))


async def dpo(store, since, llm):
    """-> rows [{prompt, chosen, rejected}]; `llm.complete(prompt)` writes the chosen reply."""
    out = []
    for r in corrections(store, since):
        try:
            chosen = (await llm.complete(REWRITE.format(prompt=r["prompt"], response=r["response"],
                                                        correction=r["better_response"]), max_tokens=120)).strip()
        except Exception:
            continue
        if not chosen.startswith("<emo=") or chosen == r["response"]:
            continue
        out.append({"prompt": _chat(r["prompt"]), "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": r["response"]}]})
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
