#!/usr/bin/env bash
# APE/AHUB routing for I2S4 (§3.14). Runs as beni-audio ExecStartPre. Control names vary slightly across
# L4T builds: list them with  amixer -c tegrasndt210ref controls | grep -Ei 'mux|i2s4|sfc|mvc'
set -u
C="-c tegrasndt210ref"
set_() { amixer $C -q cset name="$1" "$2" || echo "WARN: amixer '$1' -> '$2' failed" >&2; }
# Playback: ADMAIF1 -> SFC1 (24 kHz -> 48 kHz in hardware) -> MVC1 (hw volume ramps) -> I2S4 -> MAX98357A
set_ "SFC1 Mux" "ADMAIF1"
set_ "SFC1 input rate" 24000
set_ "SFC1 output rate" 48000
set_ "MVC1 Mux" "SFC1"
set_ "I2S4 Mux" "MVC1"
# Capture: I2S4 (48 kHz, 2 ch, 32-bit slots for INMP441) -> ADMAIF2
set_ "ADMAIF2 Mux" "I2S4"
set_ "I2S4 codec bit format" "32"
set_ "I2S4 Sample Rate" 48000
set_ "MVC1 Vol" "${BENI_VOLUME:-14000}"
exit 0
