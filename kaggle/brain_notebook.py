# %% [markdown]
# # beni-brain (Kaggle batch kernel, 2x T4)
# Pushed by the Jetson lifecycle manager (`kaggle kernels push -p /opt/beni/kaggle`), runs headless ("Save & Run All").
# GPU0: vLLM Qwen2.5-VL-7B-Instruct-AWQ (FP16 compute). GPU1: faster-whisper large-v3-turbo + Kokoro-82M
# (or CosyVoice2-0.5B) + Florence-2-large.
# Kaggle Secrets used: BENI_TOKEN, TS_AUTHKEY, HF_TOKEN (optional), CF_TUNNEL_TOKEN (optional).

# %% Cell 1: environment
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

T_START = time.time()
DEADLINE_S = 11.5 * 3600
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    # A pushed kernel has no other way to get settings: the optional BENI_* switches are read as secrets too.
    for k in ("BENI_RELAY_URL", "BENI_TOKEN", "TS_AUTHKEY", "HF_TOKEN", "CF_TUNNEL_TOKEN", "BENI_HOME_CITY", "BENI_TTS", "BENI_VISION",
              "BENI_RERANKER", "BENI_LORA", "BENI_ADAPTER_REPO", "BENI_LLM_REPO", "BENI_PERSONA", "BENI_TZ"):
        try:
            os.environ[k] = _sec.get_secret(k)
        except Exception:
            pass
except ImportError:                                    # local dry run
    pass
# Kaggle runs in UTC; the brain's day summaries, routines and "today"/"yesterday" must use the home's local time.
os.environ["TZ"] = os.environ.get("BENI_TZ") or "Asia/Colombo"
time.tzset()

WH = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/beni-wheelhouse*/uv", recursive=True)),
                          "/kaggle/input/beni-wheelhouse/uv"))
MODELS = pathlib.Path("/kaggle/tmp/models")
RUN = pathlib.Path("/kaggle/working/beni")
for p in (MODELS, RUN):
    p.mkdir(parents=True, exist_ok=True)
LLM_REPO = os.environ.get("BENI_LLM_REPO", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
LLM_LOCAL = next(iter(glob.glob("/kaggle/input/qwen2.5-vl*/**/config.json", recursive=True)), None)


def sh(cmd, bg=False, log=None, env=None):
    print("+", cmd, flush=True)
    e = dict(os.environ, **(env or {}))
    if bg:
        return subprocess.Popen(cmd, shell=True, env=e, stdout=open(log or os.devnull, "w"), stderr=subprocess.STDOUT)
    subprocess.run(cmd, shell=True, check=True, env=e)


# %% Cell 2: isolated venvs from the wheelhouse (vLLM pins its own torch; keep it away from the audio stack)
if not shutil.which("uv"):
    sh(f"install -m 755 {WH}/uv /usr/local/bin/uv" if os.path.exists(f"{WH}/uv") else "pip install -q uv")

if os.path.exists(f"{WH}/vllm"):
    sh(f"uv venv -q /kaggle/tmp/venv_vllm --python 3.12 && uv pip install --python /kaggle/tmp/venv_vllm -q --no-index --find-links {WH}/vllm vllm")
else:
    print("--> Wheelhouse not attached; installing vLLM from PyPI...", flush=True)
    sh("uv venv -q /kaggle/tmp/venv_vllm --python 3.12 && uv pip install --python /kaggle/tmp/venv_vllm -q vllm")

if os.path.exists(f"{WH}/brain"):
    BRAIN_REQ = (f"-r {WH}/brain/requirements-brain.txt" if os.path.exists(f"{WH}/brain/requirements-brain.txt")
                 else "'beni-brain[gpu]' 'transformers>=4.46,<4.50' timm einops pillow")   # older wheelhouse: same set
    sh(f"uv venv --system-site-packages -q /kaggle/tmp/venv_brain --python 3.12 && "
       f"uv pip install --python /kaggle/tmp/venv_brain -q --no-index --find-links {WH}/brain beni-brain beni-common {BRAIN_REQ}")
else:
    print("--> Wheelhouse not attached; cloning repo and installing dependencies...", flush=True)
    sh("rm -rf /kaggle/tmp/beni-repo && git clone -q --depth 1 https://github.com/PasanRamyanath/Jetson-Robot.git /kaggle/tmp/beni-repo")
    sh("uv venv --system-site-packages -q /kaggle/tmp/venv_brain --python 3.12 && "
       "uv pip install --python /kaggle/tmp/venv_brain "
       "-r /kaggle/tmp/beni-repo/kaggle/wheelhouse/requirements-brain.txt "
       "-e /kaggle/tmp/beni-repo/shared -e /kaggle/tmp/beni-repo/kaggle")
PY_BRAIN = "/kaggle/tmp/venv_brain/bin/python"
sh(f"{PY_BRAIN} -m spacy download en_core_web_sm || true")

# %% Cell 3: models (Kaggle inputs when attached, otherwise the HF Hub; /kaggle/tmp is fast local disk)
tok = os.environ.get("HF_TOKEN") or None
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sh("pip install -q huggingface_hub")
    from huggingface_hub import snapshot_download

if not LLM_LOCAL:
    print(f"--> Downloading LLM: {LLM_REPO}...", flush=True)
    snapshot_download(LLM_REPO, local_dir=str(MODELS / "llm"), token=tok)

print("--> Downloading Whisper large-v3-turbo...", flush=True)
snapshot_download("mobiuslabsgmbh/faster-whisper-large-v3-turbo", local_dir=str(MODELS / "whisper"), token=tok)

print("--> Downloading bge-small...", flush=True)
d = snapshot_download("Xenova/bge-small-en-v1.5", local_dir=str(MODELS / "bge-src"), token=tok,
                      allow_patterns=["tokenizer.json", "onnx/model_quantized.onnx"])
bge_dest = MODELS / "bge-small"
bge_dest.mkdir(parents=True, exist_ok=True)
shutil.copy(os.path.join(d, "tokenizer.json"), str(bge_dest / "tokenizer.json"))
shutil.copy(os.path.join(d, "onnx", "model_quantized.onnx"), str(bge_dest / "model_quantized.onnx"))

print("--> Downloading Kokoro, Florence-2, and bge-reranker...", flush=True)
snapshot_download("hexgrad/Kokoro-82M", token=tok)
snapshot_download("microsoft/Florence-2-large", token=tok)
snapshot_download("BAAI/bge-reranker-v2-m3", token=tok)
print("--> Model downloads complete!", flush=True)

# Optional CosyVoice2 voice (§10.5): BENI_TTS=cosyvoice plus a reference wav attached as a Kaggle input.
TTS = os.environ.get("BENI_TTS", "kokoro")
PROMPT_WAV = next(iter(glob.glob("/kaggle/input/*/beni_voice*.wav")), "")
if TTS == "cosyvoice" and PROMPT_WAV:
    sh("git clone -q --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice /kaggle/tmp/CosyVoice")
    sh("VIRTUAL_ENV=/kaggle/tmp/venv_brain uv pip install -q conformer diffusers hydra-core HyperPyYAML inflect "
       "librosa lightning modelscope omegaconf openai-whisper pyworld x-transformers wetext gdown matplotlib")
    sh(f"{PY_BRAIN} -c \"from huggingface_hub import snapshot_download as s; "
       f"s('FunAudioLLM/CosyVoice2-0.5B', local_dir='{MODELS}/cosyvoice2')\"")
elif TTS == "cosyvoice":
    print("BENI_TTS=cosyvoice needs /kaggle/input/*/beni_voice*.wav; using Kokoro", flush=True)
    TTS = "kokoro"
LLM_PATH = os.path.dirname(LLM_LOCAL) if LLM_LOCAL else str(MODELS / "llm")

# %% Cell 4: WebSocket Relay (Zero-Trust outbound relay; no VPN, no TUN, completely undetectable by Kaggle)
RELAY_URL = os.environ.get("BENI_RELAY_URL", "")
if RELAY_URL:
    print(f"--> Using Cloudflare Worker WebSocket Relay: {RELAY_URL}", flush=True)
    health_url = RELAY_URL.replace("wss://", "https://").replace("ws://", "http://").rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            print(f"--> Relay health check ({health_url}): {r.read().decode('utf-8', errors='replace')}", flush=True)
    except Exception as e:
        print(f"--> Relay probe notice: {e} (gateway will connect outbound to relay in Cell 6)", flush=True)
elif os.environ.get("CF_TUNNEL_TOKEN") and os.path.exists(f"{WH}/cloudflared"):   # fallback path (§9.1)
    sh(f"{WH}/cloudflared tunnel --no-autoupdate run --token $CF_TUNNEL_TOKEN", bg=True,
       log="/kaggle/tmp/cloudflared.log")
elif os.environ.get("TS_AUTHKEY"):
    print("WARNING: TS_AUTHKEY set, but tailscale on Kaggle is terminated by container supervisor.", flush=True)
    print("Please use BENI_RELAY_URL with the free Cloudflare Worker relay instead.", flush=True)

# %% Cell 5: vLLM on GPU0 (T4: FP16 only, no FlashAttention -> Triton attention backend, AWQ non-Marlin kernels)
# §11.12: with BENI_LORA=1 and BENI_ADAPTER_REPO set, serve the manifest's active style adapter (runtime LoRA on, so
# the nightly round can load candidates into this same server for the eval gate).
ADAPTER_REPO = os.environ.get("BENI_ADAPTER_REPO", "")
LORA = os.environ.get("BENI_LORA") == "1" and bool(ADAPTER_REPO)
LLM_NAME, LORA_ARGS, LORA_ENV = "beni-llm", "", {}
if LORA:
    LORA_ARGS = "--enable-lora --max-lora-rank 16 --max-loras 2 "
    LORA_ENV = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    try:
        _out = subprocess.run([PY_BRAIN, "-c", "import sys; from beni_brain.training.run import fetch_active as f; "
                               "print(*f(sys.argv[1], '/kaggle/tmp/adapters'))", ADAPTER_REPO],
                              capture_output=True, text=True, timeout=600, env=os.environ).stdout.split()
        if len(_out) == 2 and _out[0] != "None":
            LLM_NAME = _out[0]
            LORA_ARGS += f"--lora-modules {_out[0]}={_out[1]} "
    except Exception as e:
        print("adapter fetch failed, serving the base model:", e, flush=True)
    print("serving", LLM_NAME, flush=True)
vllm = sh(
    f"/kaggle/tmp/venv_vllm/bin/vllm serve {LLM_PATH} --served-model-name beni-llm {LORA_ARGS}"
    "--dtype float16 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.90 "
    "--enforce-eager --enable-prefix-caching --max-num-seqs 4 "
    "--limit-mm-per-prompt '{\"image\":2}' --mm-processor-kwargs '{\"max_pixels\":401408}' "
    "--enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000",
    bg=True, log="/kaggle/tmp/vllm.log",
    env={"CUDA_VISIBLE_DEVICES": "0", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN", **LORA_ENV})

# %% Cell 6: brain gateway on GPU1 (waits for vLLM itself, then warms the prefix cache)
gw = sh(f"{PY_BRAIN} -m beni_brain.gateway --port 8765", bg=True, log="/kaggle/tmp/gateway.log", env={
    "CUDA_VISIBLE_DEVICES": "1", "BENI_RUN_DIR": str(RUN), "BENI_DB": "/kaggle/tmp/beni/memory.db",
    "BENI_LLM_MODEL": LLM_NAME, "BENI_RELAY_URL": RELAY_URL, "BENI_TOKEN": os.environ.get("BENI_TOKEN", ""),
    "BENI_STT_MODEL": str(MODELS / "whisper"), "BENI_EMBED_DIR": str(MODELS / "bge-small"),
    "BENI_TTS": TTS, "BENI_TTS_PROMPT_WAV": PROMPT_WAV, "BENI_COSYVOICE_DIR": "/kaggle/tmp/CosyVoice",
    "BENI_COSYVOICE_MODEL": str(MODELS / "cosyvoice2")})


def healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=3).status == 200
    except Exception:
        return False


t0 = time.time()
while not healthy() and gw.poll() is None and vllm.poll() is None and time.time() - t0 < 1200:
    time.sleep(5)
print("brain %s after %.0f s (session %.0f s)" % ("READY" if healthy() else "FAILED", time.time() - t0,
                                                  time.time() - T_START), flush=True)

# %% Cell 7: keep the batch session alive until the robot says stop or the 11.5 h mark, then exit cleanly
while time.time() - T_START < DEADLINE_S and gw.poll() is None:
    if vllm.poll() is not None:
        print("vLLM exited:", vllm.returncode, flush=True)
        break
    time.sleep(60)
    print(time.strftime("%H:%M"), subprocess.run(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True,
        capture_output=True, text=True).stdout.strip().replace("\n", " | "), flush=True)
if gw.poll() is None:
    sh(f"BENI_RUN_DIR={RUN} {PY_BRAIN} -m beni_brain.shutdown")

# %% Cell 8: nightly style round (§11.12) on the freed GPU1, gated against the model this session served
if LORA and os.environ.get("HF_TOKEN") and vllm.poll() is None and DEADLINE_S + 1800 - (time.time() - T_START) > 5400:
    try:
        sh(f"uv venv -q --system-site-packages --python {sys.executable} /kaggle/tmp/venv_train && "
           "VIRTUAL_ENV=/kaggle/tmp/venv_train uv pip install -q 'transformers>=4.46,<4.50' 'peft>=0.14,<0.15' "
           "'trl>=0.15,<0.16' 'bitsandbytes>=0.45' 'accelerate>=1.2' datasets")
        sh(f"{PY_BRAIN} -m beni_brain.training.run --repo {ADAPTER_REPO} --gpu 1",
           env={"BENI_DB": "/kaggle/tmp/beni/memory.db", "BENI_LLM_MODEL": LLM_NAME})
    except subprocess.CalledProcessError as e:
        print("training round failed:", e, flush=True)
vllm.terminate()
for f in ("gateway.log", "vllm.log"):                  # keep the logs as kernel output for debugging
    if os.path.exists(f"/kaggle/tmp/{f}"):
        shutil.copy(f"/kaggle/tmp/{f}", RUN / f)
print("session done after %.1f h" % ((time.time() - T_START) / 3600), flush=True)
