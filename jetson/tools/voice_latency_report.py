#!/usr/bin/env python3
"""p50/p90 per voice stage from the agent's turn log (§14.4).  Usage: voice_latency_report.py /ssd/beni/logs/turns.jsonl

Each line: {"turn": n, "wake": t, "speech_end": t, "stt_final_rx": t, "first_llm_delta_rx": t, "first_tts_rx": t,
            "first_audio_out": t, "filler": bool, "offline": bool}
"""
import json
import sys

STAGES = (("speech_end", "stt_final_rx"), ("stt_final_rx", "first_llm_delta_rx"),
          ("first_llm_delta_rx", "first_tts_rx"), ("first_tts_rx", "first_audio_out"),
          ("speech_end", "first_audio_out"), ("wake", "speech_end"))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def main(path):
    turns = [json.loads(l) for l in open(path) if l.strip()]
    print("%d turns (%d offline, %d with filler)" % (
        len(turns), sum(1 for t in turns if t.get("offline")), sum(1 for t in turns if t.get("filler"))))
    print("%-42s %6s %8s %8s %8s" % ("stage", "n", "p50 ms", "p90 ms", "max ms"))
    for a, b in STAGES:
        d = [1000.0 * (t[b] - t[a]) for t in turns if t.get(a) and t.get(b)]
        if d:
            print("%-42s %6d %8.0f %8.0f %8.0f" % (a + " -> " + b, len(d), pct(d, 50), pct(d, 90), max(d)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/ssd/beni/logs/turns.jsonl")
