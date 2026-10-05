#!/usr/bin/env bash
# CPU-side models for the agent (§8): sherpa-onnx KWS/VAD/ASR/TTS/speaker-id, bge-small ONNX, the offline LLM GGUF,
# and pre-rendered filler clips. Idempotent: skips what exists. Layout matches beni_agent.voice.speech.load().
# Vision ONNX -> TensorRT engines are built separately by jetson/engines/build_all.sh.
set -euo pipefail
M=${BENI_MODELS:-/ssd/beni/models}
VENV=${VENV:-/opt/beni/.venv38}
SH=https://github.com/k2-fsa/sherpa-onnx/releases/download
HF=https://huggingface.co
mkdir -p "$M"
cd "$M"

get() {  # url dest
  [ -s "$2" ] && return 0
  echo "== $2"; curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
untar() {  # url subdir: extract a sherpa tarball's single top dir into $M/subdir
  [ -d "$2" ] && { echo "have $2"; return 0; }
  local tmp; tmp=$(mktemp -d "$M/.x.XXXX")
  curl -fL --retry 3 "$1" | tar xj -C "$tmp"
  mv "$tmp"/* "$2" && rmdir "$tmp"
}

# Wake word: 3.3M zipformer (gigaspeech, BPE, uppercase keywords). ~20 ms per 100 ms hop on one A57 core.
untar "$SH/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01-mobile.tar.bz2" kws
if [ ! -s kws/keywords.txt ]; then
  "$VENV/bin/pip" install -q sentencepiece pypinyin   # text2token deps
  printf 'HEY BENI :2.0 #0.25 @HEY_BENI\nBENI :1.5 #0.3 @BENI\n' > kws/keywords_raw.txt
  "$VENV/bin/sherpa-onnx-cli" text2token --tokens kws/tokens.txt --tokens-type bpe \
    --bpe-model kws/bpe.model kws/keywords_raw.txt kws/keywords.txt
fi
# VAD
mkdir -p vad && get "$SH/asr-models/silero_vad.onnx" vad/silero_vad.onnx
# Offline ASR (brain unreachable): Moonshine tiny int8, ~0.3 RTF on 1 core
untar "$SH/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2" asr
# Offline TTS: Piper amy-low (16 kHz, fastest acceptable VITS on the Nano CPU)
untar "$SH/tts-models/vits-piper-en_US-amy-low.tar.bz2" tts
# Speaker verification: CAM++ English (VoxCeleb), 512-d
mkdir -p spk && get "$SH/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" spk/campplus_en.onnx
# Memory embeddings: bge-small-en-v1.5 int8 ONNX (same files on the brain, so vectors are compatible)
mkdir -p bge-small
get "$HF/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx" bge-small/model_quantized.onnx
get "$HF/Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json" bge-small/tokenizer.json
# Offline LLM for llama-server (beni-llm.service)
get "$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" qwen2.5-0.5b-instruct-q4_k_m.gguf

# Fillers ("hmm", "let me think") as 24 kHz s16le .raw for the player. Rendered here with the local Piper voice;
# replace with Kokoro renders from the brain for a perfect voice match (tools: beni_brain.tts.KokoroTTS.synth).
if [ ! -d fillers ]; then
  mkdir -p fillers
  PYTHONPATH=/opt/beni/jetson/agent "$VENV/bin/python" - "$M" <<'PY'
import os, sys
from beni_agent.voice.speech import LocalTts
m = sys.argv[1]
tts = LocalTts(os.path.join(m, "tts"))
for i, text in enumerate(["Hmm.", "Let me think.", "Okay!", "One moment."]):
    with open(os.path.join(m, "fillers", "%02d.raw" % i), "wb") as f:
        f.write(tts.synth(text))                      # already 24 kHz s16le
print("fillers ok")
PY
fi
du -sh "$M"/* | sort -h
