"""Headless multi-turn chat demo against the trained TinyStories GPT (M7).

The provided `train_chat_lm_modal.py::chat` entrypoint is an interactive
`input()` REPL, which cannot run on a headless VPS. This driver reproduces the
*exact same* completion-style loop (accumulate history, feed the last ~1800
chars to the remote model, append the reply) but with scripted narrative turns,
so it runs unattended. All turns go through `generate_remote`, so they are also
appended to `chat_logs.jsonl` in the `maci-lm` volume (feeding M8).

A single `app.run()` context is used so the A10G container stays warm across
turns (measuring warm per-turn latency for the M7 acceptance check).

Usage:
    uv run python scripts/chat_demo.py
    uv run python scripts/chat_demo.py --topic robots   # seed M8 learning demo

Does NOT modify train_chat_lm_modal.py — it imports from it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import modal

# Repo root holds train_chat_lm_modal.py; ensure it is importable when this
# script is run from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_chat_lm_modal import app, generate_remote  # noqa: E402

# Narrative turns — the model is a story LM, so turns are phrased as narration
# (per the chat() docstring) and each builds on the previous reply.
SCRIPTS = {
    "default": [
        "Once upon a time, there was a little robot named Beep who wanted to "
        "learn how to paint.",
        "Beep walked into the forest and met a wise old owl. The owl looked at "
        "Beep and said,",
        "So Beep picked up a paintbrush and tried to paint the sunset. But the "
        "colors",
        "The next morning, Beep showed the painting to his best friend, a girl "
        "named Mia. Mia looked at it and said,",
        "From that day on, Beep and Mia decided to",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="default")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--out", default="results/chat_transcript.md")
    args = ap.parse_args()

    turns = SCRIPTS.get(args.topic, SCRIPTS["default"])
    transcript: list[tuple[str, str, float]] = []
    history = ""

    with modal.enable_output(), app.run():
        for i, user in enumerate(turns, 1):
            history += f"\n{user}\n"
            t0 = time.time()
            reply = generate_remote.remote(history[-1800:],
                                           max_new=args.max_new,
                                           temperature=args.temperature)
            dt = time.time() - t0
            history += reply
            transcript.append((user, reply, dt))
            print(f"\n[turn {i}] ({dt:.1f}s)\nyou > {user}\nbot > {reply}")

    warm = [dt for _, _, dt in transcript[1:]]  # exclude first (cold) turn
    avg_warm = sum(warm) / len(warm) if warm else float("nan")

    lines = [
        "# M7 — Chat transcript (TinyStories GPT, completion-style)",
        "",
        "Generated headlessly via `scripts/chat_demo.py` against the M6 "
        "checkpoint (`/data/chat_lm.pt`, 51.2M params) on Modal A10G. The model "
        "is a story LM, not an assistant: it continues narrative prompts. Turns "
        "are phrased as narration and each builds on the model's prior reply, "
        "reproducing the `chat()` loop exactly.",
        "",
        f"Turn 1 includes cold-start container spin-up; warm turns averaged "
        f"**{avg_warm:.1f}s** (M7 target: < 5s warm).",
        "",
    ]
    for i, (user, reply, dt) in enumerate(transcript, 1):
        lines += [f"## Turn {i}  ({dt:.1f}s)", "",
                  f"**you >** {user}", "", f"**bot >** {reply}", ""]
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {args.out}  (warm avg {avg_warm:.1f}s/turn)")


if __name__ == "__main__":
    main()
