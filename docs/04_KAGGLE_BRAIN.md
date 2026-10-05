# 4. Kaggle brain (2×T4, §10)

The brain is a private Kaggle **batch kernel** (`kaggle/brain_notebook.py`):
- **GPU0:** vLLM Qwen2.5-VL-7B-AWQ.
- **GPU1:** faster-whisper, Kokoro or CosyVoice2, and Florence-2.

It joins the tailnet as `beni-brain` and serves `ws://beni-brain:8765/ws`. You never start it by hand: the
Jetson's lifecycle manager pushes it during awake hours or on the wake word, and asks it to stop when idle. A
session is at most 11.5 h, and the weekly budget defaults to 28 h (`BENI_WEEKLY_BUDGET_H`).

## 4.1 Kaggle account

1. Verify your phone number on kaggle.com. GPU and internet access need it.
2. Go to kaggle.com → Settings → API → **Create New Token**. This downloads `kaggle.json`. Keep it for the PC
   (`~/.kaggle/kaggle.json`, mode 600) and for the Nano ([02 §2.11](02_JETSON_SETUP.md#211-configure-and-start)).

## 4.2 Tailscale key for the brain

In the Tailscale admin console, create an auth key that is **ephemeral, reusable, pre-approved** and tagged
`tag:beni-brain`. Ephemeral means each finished session removes its own node. The ACL from
[02 §2.4](02_JETSON_SETUP.md#24-system-tuning-docker-tailscale-the-py38-venv-mediamtx-and-deepstream-yolo)
already allows `tag:beni-jetson` → `tag:beni-brain:8765` and nothing else.

## 4.3 Kaggle Secrets

In kaggle.com → Code → New Notebook → Add-ons → Secrets, add these. A secret is available to every kernel you
own, so this is a one-time step. The pushed kernel reads them all at start-up.

| Secret | Required | Value |
|---|---|---|
| `BENI_TOKEN` | yes | the same string as `BENI_TOKEN` in `/etc/beni/beni.env` |
| `TS_AUTHKEY` | yes | the `tag:beni-brain` key from 4.2 |
| `HF_TOKEN` | recommended | HF read token (faster, unthrottled model downloads); write access if you use `BENI_LORA` |
| `CF_TUNNEL_TOKEN` | no | Cloudflare tunnel fallback (§9.1); the Nano then also needs `BENI_BRAIN_URLS=...,wss://<host>/ws` |
| `BENI_TTS` | no | `kokoro` (default) or `cosyvoice` (4.6) |
| `BENI_VISION` | no | `none` turns off Florence-2 (`locate`, replay captions) |
| `BENI_RERANKER` | no | `none` turns off the bge-reranker-v2-m3 memory rerank (§11.6) |
| `BENI_LLM_REPO` | no | a different AWQ model repo (default `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`) |
| `BENI_LORA`, `BENI_ADAPTER_REPO` | no | `1` plus a private HF model repo to train and serve the style adapter (§11.12) |
| `BENI_PERSONA` | no | replaces the built-in persona text |
| `BENI_HOME_CITY` | no | default `Colombo` (weather, local time) |
| `BENI_TZ` | no | the home's time zone, default `Asia/Colombo`: day summaries, routines and "today" use it |

Kaggle attaches secrets per notebook. After the first push (4.5), open the `beni-brain` notebook → Edit →
Add-ons → Secrets, tick every secret above, then Save Version. Later pushes keep the attachments.

## 4.4 Offline wheelhouse (required)

This needs **Linux x86_64 with Python 3.12**, or Docker, which the script then uses automatically. It downloads
roughly 3 GB, so don't run it on a PC with a small system disk. A Kaggle notebook or any Linux box works:

```bash
pc$ pip install "kaggle>=1.6,<1.7" && kaggle datasets list -m          # checks ~/.kaggle/kaggle.json
pc$ cd ~/beni && make wheelhouse-push                                  # -> dataset <you>/beni-wheelhouse
```

The kernel installs vLLM and the brain with `uv pip install --no-index` from this dataset. The brain's own wheels
(`beni-brain`, `beni-common`) exist nowhere else, so **the kernel can't start without it**. Rebuild and push it
whenever `kaggle/wheelhouse/requirements-*.txt`, `shared/` or `kaggle/beni_brain/` change.

## 4.5 First run by hand (to check everything before the robot drives it)

```bash
pc$ cd ~/beni
pc$ make kaggle-push KAGGLE_KERNEL=<you>/beni-brain   # fills in your user name (as the lifecycle manager does)
pc$ kaggle kernels status <you>/beni-brain            # queued -> running
```

Then:
1. Open the kernel's log on kaggle.com. Within ~10–15 min it prints `brain READY after N s`.
2. On the Nano, run `tailscale status | grep beni-brain`, then:

```bash
nano$ curl -s http://beni-brain:8765/health
nano$ journalctl -u beni-agent -f            # "link up", then say the wake word
```

**Stopping the brain:**
- Normally the robot sends `brain.stop`. The brain then consolidates memory, flushes the deltas to the Jetson and
  exits.
- By hand: cancel the run on kaggle.com. Unsynced brain-side rows are lost, but the Jetson DB is the system of
  record.

## 4.6 CosyVoice (optional cloned voice)

1. Record 3–10 s of clean speech as `beni_voice.wav` (mono, ≥ 16 kHz).
2. Upload it as a private Kaggle dataset, e.g. `<you>/beni-voice`.
3. Add it to `dataset_sources` in `kaggle/kernel-metadata.json`, next to the wheelhouse:
   `["YOUR_KAGGLE_USERNAME/beni-wheelhouse", "YOUR_KAGGLE_USERNAME/beni-voice"]`.
4. Set the secret `BENI_TTS=cosyvoice`.

If the wav is missing or CosyVoice fails to load, that session uses Kokoro throughout.

## 4.7 Automatic operation (Nano side)

These `/etc/beni/beni.env` keys drive the lifecycle:

| Key | Default | Meaning |
|---|---|---|
| `KAGGLE_KERNEL` | — | `<you>/beni-brain` (required; empty turns the lifecycle off) |
| `KAGGLE_CONFIG_DIR` | `/home/beni/.kaggle` | where `kaggle.json` is |
| `BENI_AWAKE_HOURS` | see `config.py` | windows when the brain is kept up, e.g. `07:00-12:00,16:00-22:30` |
| `BENI_WEEKLY_BUDGET_H` | `28` | GPU hours per ISO week (Kaggle gives ~30) |

Outside the awake hours, the wake word cold-starts the brain for 30 min. Meanwhile the offline llama.cpp model
(`beni-llm`) answers.

## 4.8 Troubleshooting

| Symptom | Check |
|---|---|
| `kaggle push failed` in the agent journal | `sudo -u beni /opt/beni/.venv38/bin/kaggle kernels list -m` (key path, phone verification) |
| Kernel runs but the Nano never connects | the kernel log's `tailscale up` line; the ACL; `BENI_TOKEN` equal on both sides |
| `brain FAILED` in the kernel log | `/kaggle/tmp/vllm.log` and `gateway.log` in the kernel output; usually an HF download (set `HF_TOKEN`) |
| `uv pip install` fails at once | the `beni-wheelhouse` dataset is missing or stale: `make wheelhouse-push` |
| A secret seems empty | it isn't attached to the notebook (4.3) |
