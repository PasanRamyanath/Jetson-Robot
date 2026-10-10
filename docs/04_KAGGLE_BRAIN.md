# 4. Kaggle brain (2×T4, §10)

The brain is a private Kaggle **batch kernel** (`kaggle/brain_notebook.py`):
- **GPU0:** vLLM Qwen2.5-VL-7B-AWQ.
- **GPU1:** faster-whisper, Kokoro or CosyVoice2, and Florence-2.

It connects outbound to the **Cloudflare Worker WebSocket Relay** (`BENI_RELAY_URL`). You never start it by hand: the
Jetson's lifecycle manager pushes it during awake hours or on the wake word, and asks it to stop when idle. A
session is at most 11.5 h, and the weekly budget defaults to 28 h (`BENI_WEEKLY_BUDGET_H`).

## 4.1 Kaggle account

1. Verify your phone number on kaggle.com. GPU and internet access need it.
2. Go to kaggle.com → Settings → API → **Create New Token**. This downloads `kaggle.json`. Keep it for the PC
   (`~/.kaggle/kaggle.json`, mode 600) and for the Nano ([02 §2.11](02_JETSON_SETUP.md#211-configure-and-start)).

## 4.2 Cloudflare Worker WebSocket Relay (100% Free, Non-VPN)

Kaggle's host supervisor strictly blocks and terminates VPN daemons like `tailscaled` within seconds.
Instead, Beni uses a **free, permanent, zero-trust Cloudflare Worker WebSocket Relay** (`cloudflare/worker.js`):
1. Create a free account at [dash.cloudflare.com](https://dash.cloudflare.com) (no credit card or domain required).
2. Go to **Compute (Workers) > Workers & Pages** -> **Create Application** -> **Create Worker**.
3. Name it `beni-relay`, click **Deploy**, then **Edit code**.
4. Paste the code from `cloudflare/worker.js` and click **Save and deploy**.
5. Under **Settings > Bindings**, add a Durable Object binding: variable `RELAY`, class `BeniRelay`.
6. Your permanent relay URL is: `wss://beni-relay.<your-subdomain>.workers.dev`.

Both the Jetson Nano and the Kaggle Brain make standard **outbound** WebSocket connections to this relay,
completely bypassing Kaggle's VPN detection with minimal latency (~20–40 ms).

## 4.3 Kaggle Secrets

In kaggle.com → Code → New Notebook → Add-ons → Secrets, add these. A secret is available to every kernel you
own, so this is a one-time step. The pushed kernel reads them all at start-up.

| Secret | Required | Value |
|---|---|---|
| `BENI_TOKEN` | yes | the shared secret string matching `BENI_TOKEN` in `/etc/beni/beni.env` |
| `BENI_RELAY_URL` | yes | `wss://beni-relay.<your-subdomain>.workers.dev` (from 4.2) |
| `HF_TOKEN` | recommended | HF read token (faster, unthrottled model downloads); write access if you use `BENI_LORA` |
| `CF_TUNNEL_TOKEN` | no | Cloudflare named tunnel token fallback (§9.1) |
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
2. Check relay status from your browser or terminal:

```bash
curl -s https://beni-relay.<your-subdomain>.workers.dev/health
# {"status": "ok", "brain_connected": true, "robot_connected": true, ...}
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
| Kernel runs but the Nano never connects | check `https://<relay-url>/health`; verify `BENI_RELAY_URL` in Kaggle secrets and `BENI_TOKEN` matches in `/etc/beni/beni.env` |
| `brain FAILED` in the kernel log | `/kaggle/tmp/vllm.log` and `gateway.log` in the kernel output; usually an HF download (set `HF_TOKEN`) |
| `uv pip install` fails at once | the `beni-wheelhouse` dataset is missing or stale: `make wheelhouse-push` |
| A secret seems empty | it isn't attached to the notebook (4.3) |
