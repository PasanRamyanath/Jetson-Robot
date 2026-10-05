"""python -m beni_brain.training.run: one nightly round of §11.12 after the gateway has stopped (GPU1 is free).

build data from the mirror -> (enough new samples?) -> train.py in the training venv on GPU1 -> load the adapter into
the running vLLM (runtime LoRA) -> gate against the production model -> publish to the private HF repo and make it
active in manifest.json. The next session serves `manifest["active"]`; `--rollback` re-activates the previous one.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from beni_common.memory import MemoryStore

from ..config import Config
from ..llm import LLM
from . import data, gate

TRAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")


def hub():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def load_manifest(repo):
    from huggingface_hub import hf_hub_download
    try:
        return json.load(open(hf_hub_download(repo, "manifest.json", token=os.environ.get("HF_TOKEN") or None)))
    except Exception:
        return {"active": None, "history": [], "trained_until": 0}


def save_manifest(repo, m, msg):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(m, f, indent=1)
    hub().upload_file(path_or_fileobj=f.name, path_in_repo="manifest.json", repo_id=repo, commit_message=msg)


def fetch_active(repo, dest):
    """Notebook helper: download the active adapter -> (name, local dir) or (None, None)."""
    m = load_manifest(repo)
    if not m.get("active"):
        return None, None
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, allow_patterns=[m["active"] + "/*"], local_dir=dest,
                          token=os.environ.get("HF_TOKEN") or None)
    return m["active"], os.path.join(d, m["active"])


def load_lora(vllm_url, name, path):
    body = json.dumps({"lora_name": name, "lora_path": path}).encode()
    req = urllib.request.Request(vllm_url.rstrip("/") + "/load_lora_adapter", body,
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()


async def round_(a):
    cfg = Config.from_env()
    m = load_manifest(a.repo)
    since = float(m.get("trained_until") or 0)
    store = MemoryStore(a.db, node="kaggle-train")
    sft_rows, n_new = data.sft(store, since)
    llm = LLM(cfg)
    await llm.ready(timeout=60)
    dpo_rows = await data.dpo(store, since, llm) if len(data.corrections(store, since)) >= a.min_dpo else []
    do_sft, do_dpo = n_new >= a.min_sft, len(dpo_rows) >= a.min_dpo
    print("new good turns %d (need %d), DPO pairs %d (need %d)" % (n_new, a.min_sft, len(dpo_rows), a.min_dpo))
    if not (do_sft or (do_dpo and m.get("active"))):
        return 0
    name = "beni-style-v%d" % (len(m["history"]) + 1)
    work = os.path.join(a.work, name)
    os.makedirs(work, exist_ok=True)
    cmd = [a.train_python, TRAIN_PY, "--base", a.base, "--out", work]
    if do_sft:
        cmd += ["--sft", data.write_jsonl(os.path.join(a.work, "sft.jsonl"), sft_rows)]
    if do_dpo:
        cmd += ["--dpo", data.write_jsonl(os.path.join(a.work, "dpo.jsonl"), dpo_rows)]
        if not do_sft:
            _, init = fetch_active(a.repo, os.path.join(a.work, "active"))
            cmd += ["--init", init]
    t0 = time.time()
    subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu))
    print("trained in %.0f min" % ((time.time() - t0) / 60))

    client = llm.client
    base_model = cfg.llm_model                      # what vLLM serves now (the active adapter, if it was fetched)
    try:
        load_lora(cfg.llm_url, name, work)
        new, fails = await gate.evaluate(client, name)
    except Exception as e:
        print("adapter failed to load/evaluate:", e)
        return 1
    base, _ = await gate.evaluate(client, base_model)
    ok, lines = gate.verdict(base, new)
    print("\n".join(["gate vs %s:" % base_model] + lines + ["ACCEPTED" if ok else "REJECTED"]))
    report = {"name": name, "base": base_model, "baseline": base, "scores": new, "accepted": ok,
              "sft_new": n_new if do_sft else 0, "dpo_pairs": len(dpo_rows) if do_dpo else 0, "t": time.time(),
              "failures": fails[:40]}
    with open(os.path.join(work, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if not ok:
        return 0
    api = hub()
    api.create_repo(a.repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=work, path_in_repo=name, repo_id=a.repo, commit_message="%s: %s" % (name, lines),
                      ignore_patterns=["sft/*", "dpo/*", "checkpoint-*"])
    last = store.q("SELECT max(t) AS t FROM feedback WHERE kind IN ('turn', 'correction')")[0]["t"] or since
    m.update(active=name, trained_until=last, history=m["history"] + [{"name": name, "t": time.time(),
                                                                        "scores": new}])
    save_manifest(a.repo, m, "activate " + name)
    print("published and activated", name)
    return 0


def rollback(repo):
    m = load_manifest(repo)
    names = [h["name"] for h in m["history"]]
    if m.get("active") not in names:
        print("nothing to roll back")
        return 1
    i = names.index(m["active"])
    m["active"] = names[i - 1] if i > 0 else None
    save_manifest(repo, m, "rollback to %s" % m["active"])
    print("active adapter:", m["active"] or "base model")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("BENI_ADAPTER_REPO", ""), help="private HF model repo")
    ap.add_argument("--db", default=os.environ.get("BENI_DB", "/kaggle/tmp/beni/memory.db"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--work", default="/kaggle/tmp/train")
    ap.add_argument("--train-python", default="/kaggle/tmp/venv_train/bin/python")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--min-sft", type=int, default=data.MIN_SFT)
    ap.add_argument("--min-dpo", type=int, default=data.MIN_DPO)
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args(argv)
    if not a.repo:
        ap.error("--repo or BENI_ADAPTER_REPO is required")
    return rollback(a.repo) if a.rollback else asyncio.run(round_(a))


if __name__ == "__main__":
    sys.exit(main())
