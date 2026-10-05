#!/usr/bin/env bash
# beni_audio (§8.2): one gst-launch owns the APE card. Mic -> AEC/NS/AGC -> 16 kHz RTP L16 -> udp:6000 (agent)
# udp:6001 (24 kHz RTP L16 TTS from the agent) -> echo probe -> I2S4 (SFC1 resamples to 48 kHz in hardware).
# BENI_AUDIO_SW_RESAMPLE=1 resamples in software when SFC routing isn't working yet.
# BENI_AUDIO_CAP / BENI_AUDIO_PLAY override the ALSA devices (USB sound-card fallback).
set -eu
CAP=${BENI_AUDIO_CAP:-hw:tegrasndt210ref,1}
PLAY=${BENI_AUDIO_PLAY:-hw:tegrasndt210ref,0}
OUT_RATE=24000; SW=""
if [ "${BENI_AUDIO_SW_RESAMPLE:-0}" = 1 ]; then OUT_RATE=48000; SW="audioresample !"; fi
exec gst-launch-1.0 -q \
  alsasrc device="$CAP" buffer-time=40000 latency-time=10000 ! \
    audio/x-raw,format=S32LE,rate=48000,channels=2 ! \
    audioconvert mix-matrix="<<(float)0.5, (float)0.5>>" ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! \
    webrtcdsp echo-cancel=true noise-suppression=true noise-suppression-level=high gain-control=true \
      experimental-agc=false extended-filter=true delay-agnostic=true high-pass-filter=true ! \
    audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! \
    rtpL16pay pt=96 mtu=1400 ! udpsink host=127.0.0.1 port=6000 sync=false async=false \
  udpsrc port=6001 caps="application/x-rtp,media=audio,clock-rate=24000,encoding-name=L16,channels=1,payload=96" ! \
    rtpjitterbuffer latency=40 ! rtpL16depay ! audioconvert ! audio/x-raw,format=S16LE,rate=24000,channels=1 ! \
    webrtcechoprobe ! audioconvert ! $SW audio/x-raw,format=S16LE,rate=$OUT_RATE,channels=2 ! \
    alsasink device="$PLAY" buffer-time=60000 sync=false
