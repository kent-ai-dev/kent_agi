"""M8 — continual-learning before/after demonstration.

Shows that `finetune_on_chats` makes the model's *unconditional* samples mention
a distinctive seeded topic measurably more often, while not catastrophically
forgetting its TinyStories pretraining (25% replay is mandatory; do not remove).

Topic: an astronaut named **Pip** who flies a **rocket** to **Mars** — chosen
because astronaut/rocket/Mars/spaceship are rare in TinyStories, so the
before-frequency is ~0 and any shift is unambiguous.

Subcommands (orchestrated by the surrounding bash; Modal does the finetune):
    seed    --out FILE                    write the seed chat_logs.jsonl
    measure --ckpt FILE --out FILE --label L   generate samples, count topic
                                               mentions, compute forgetting loss
    report  --before FILE --after FILE --out FILE   build the markdown comparison

Generation/loss run locally on CPU (inference only — no training on the VPS).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_chat_lm_modal import GPTConfig, build_model_classes  # noqa: E402

TOPIC_WORDS = ["pip", "astronaut", "rocket", "mars", "spaceship", "planet",
               "space", "moon", "star"]

# Seed stories — TinyStories register, densely featuring the topic.
SEED_STORIES = [
    ("Once upon a time, there was a little astronaut named Pip.",
     " Pip had a shiny red rocket and he wanted to fly to Mars. He put on his "
     "space helmet, climbed into the rocket, and counted down: three, two, one! "
     "The rocket zoomed up past the moon and the stars. Pip was so happy to be "
     "an astronaut flying through space."),
    ("Pip the astronaut looked out of his rocket window.",
     " He could see the red planet Mars getting closer and closer. \"I am going "
     "to land on Mars!\" said Pip. The rocket landed softly on the dusty ground. "
     "Pip stepped out in his space suit and waved at the tiny stars far away."),
    ("On Mars, Pip the astronaut found a funny little rock.",
     " The rock was red, just like the whole planet. Pip put the Mars rock in "
     "his rocket to bring it home. He looked up at the dark space sky and saw "
     "Earth, a little blue dot near the moon. Pip missed his mom."),
    ("Pip wanted to show his friend the rocket.",
     " \"Look, this rocket can fly all the way to Mars!\" said Pip the astronaut. "
     "His friend was amazed. Together they pretended to be astronauts, flying "
     "past the moon and the stars and landing on the red planet."),
    ("Every night, Pip the astronaut looked at the stars.",
     " He dreamed of his rocket and the planet Mars. \"One day I will fly back to "
     "space,\" said Pip. He drew a picture of a rocket and a moon and a big red "
     "Mars, and hung it on his wall so he could see space every day."),
    ("Pip packed his bag for a long trip to Mars.",
     " He brought space food, a star map, and his lucky moon rock. The rocket "
     "was ready on the launch pad. Pip the astronaut waved goodbye and the "
     "rocket roared into space, higher than the clouds, all the way to the planet."),
    ("The little astronaut Pip met a friendly Mars cloud.",
     " The cloud floated over the red planet. \"Hello, I am Pip, an astronaut from "
     "Earth,\" he said. The cloud showed Pip the best craters on Mars. Pip thanked "
     "it, climbed back into his rocket, and flew home past the moon and stars."),
    ("Pip's rocket needed more fuel to reach Mars.",
     " The astronaut filled it up with bright blue space fuel. Then the rocket "
     "lifted off, leaving Earth and the moon behind. Soon Pip could see the red "
     "planet Mars shining among the stars. He was the bravest little astronaut "
     "in all of space."),
]


def make_seed(path: str) -> int:
    tokens_est = 0
    with open(path, "w") as f:
        # repeat so we comfortably clear finetune_on_chats' 2048-token floor
        # (8 stories x ~70 GPT-2 tokens x 5 ~= 2800 tokens)
        t = 1000.0
        for _ in range(5):
            for prompt, reply in SEED_STORIES:
                f.write(json.dumps({"t": t, "prompt": prompt,
                                    "reply": reply}) + "\n")
                tokens_est += len((prompt + reply).split())
                t += 1.0
    return tokens_est


def count_topic(texts: list[str]) -> dict:
    counts = {w: 0 for w in TOPIC_WORDS}
    n_with_topic = 0
    for txt in texts:
        low = txt.lower()
        hit = False
        for w in TOPIC_WORDS:
            c = len(re.findall(rf"\b{re.escape(w)}\b", low))
            counts[w] += c
            if c:
                hit = True
        if hit:
            n_with_topic += 1
    return {"per_word": counts, "total_mentions": sum(counts.values()),
            "samples_with_topic": n_with_topic, "n_samples": len(texts)}


# A fixed probe set of TinyStories-register sentences (NOT the seed topic) to
# gauge forgetting: mean next-token cross-entropy before vs after finetune.
FORGET_PROBE = [
    "Once upon a time, there was a little girl named Lily who loved to play.",
    "Tom was very hungry, so he asked his mom for something to eat.",
    "The cat sat on the mat and looked at the bright red ball.",
    "One day, a little boy found a shiny rock in the garden.",
    "Lily and her friend played together all day and were very happy.",
    "The dog ran fast to fetch the stick and wagged its tail.",
    "Mom baked a cake and the whole house smelled sweet and warm.",
    "The bird sang a pretty song high up in the green tree.",
    "Timmy did not want to clean his room, but he knew he had to.",
    "They shared the toys and learned that sharing makes everyone glad.",
]


def forgetting_loss(model, enc) -> float:
    import torch.nn.functional as F
    losses = []
    for s in FORGET_PROBE:
        ids = enc.encode_ordinary(s)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor([ids[1:]])
        with torch.no_grad():
            logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss))
    return sum(losses) / len(losses)


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    GPT = build_model_classes()
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck.get("step")


def measure(ckpt_path: str, out_path: str, label: str, n: int = 24) -> None:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model, step = load_model(ckpt_path)
    # Unconditional-ish: neutral story starter, fixed across before/after.
    texts = []
    torch.manual_seed(0)
    start = torch.tensor([enc.encode_ordinary("Once upon a time")])
    for _ in range(n):
        out = model.generate(start, max_new_tokens=70, temperature=0.8,
                             top_k=50, eos_id=enc.eot_token)
        texts.append(enc.decode(out[0].tolist()))
    topic = count_topic(texts)
    forget = forgetting_loss(model, enc)
    result = {"label": label, "ckpt_step": step, "topic": topic,
              "forget_loss": forget, "samples": texts[:6]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{label}] step={step} topic_mentions={topic['total_mentions']} "
          f"samples_with_topic={topic['samples_with_topic']}/{topic['n_samples']} "
          f"forget_loss={forget:.3f}")


def report(before_path: str, after_path: str, out_path: str) -> None:
    b = json.load(open(before_path))
    a = json.load(open(after_path))
    bt, at = b["topic"], a["topic"]
    dloss = a["forget_loss"] - b["forget_loss"]
    lines = [
        "# M8 — Continual learning: before / after `finetune_on_chats`",
        "",
        "Seeded topic: **an astronaut named Pip who flies a rocket to Mars** "
        "(astronaut/rocket/Mars/spaceship are rare in TinyStories, so any shift "
        "is unambiguous). Finetune: 300 steps, lr 5e-5, **25% pretraining "
        "replay** (mandatory anti-forgetting guardrail). Unconditional samples "
        "= 24 continuations of \"Once upon a time\" at temp 0.8 (same seed "
        "before/after).",
        "",
        "## Topic adoption (the \"it learns\" signal)",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| total topic-word mentions | {bt['total_mentions']} | "
        f"{at['total_mentions']} |",
        f"| samples mentioning the topic | "
        f"{bt['samples_with_topic']}/{bt['n_samples']} | "
        f"{at['samples_with_topic']}/{at['n_samples']} |",
        f"| checkpoint step | {b['ckpt_step']} | {a['ckpt_step']} |",
        "",
        "Per-word mentions (before → after): " + ", ".join(
            f"{w}: {bt['per_word'][w]}→{at['per_word'][w]}"
            for w in TOPIC_WORDS) + ".",
        "",
        "## Forgetting check (pretraining retention)",
        "",
        f"Mean next-token loss on a fixed TinyStories-register probe set "
        f"(off-topic): **{b['forget_loss']:.3f} → {a['forget_loss']:.3f}** "
        f"(Δ {dloss:+.3f}). PLAN.md M8 bound: degradation ≤ 0.15 → "
        f"{'PASS' if dloss <= 0.15 else 'FAIL'}.",
        "",
        "## Sample unconditional generations AFTER finetune",
        "",
    ]
    for s in a["samples"][:4]:
        lines += ["> " + s.replace("\n", " ").strip(), ""]
    lines += ["## Sample BEFORE finetune (for contrast)", ""]
    for s in b["samples"][:2]:
        lines += ["> " + s.replace("\n", " ").strip(), ""]
    verdict = (at["total_mentions"] > bt["total_mentions"] and dloss <= 0.15)
    lines += [f"**Verdict: {'PASS' if verdict else 'MIXED'}** — topic mentions "
              f"{'increased' if at['total_mentions'] > bt['total_mentions'] else 'did not increase'}"
              f" and forgetting {'stayed within' if dloss <= 0.15 else 'exceeded'}"
              f" the 0.15 bound."]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--out", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--ckpt", required=True); m.add_argument("--out", required=True)
    m.add_argument("--label", required=True)
    r = sub.add_parser("report")
    r.add_argument("--before", required=True); r.add_argument("--after", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "seed":
        n = make_seed(args.out)
        print(f"wrote seed -> {args.out} (~{n} words)")
    elif args.cmd == "measure":
        measure(args.ckpt, args.out, args.label)
    elif args.cmd == "report":
        report(args.before, args.after, args.out)


if __name__ == "__main__":
    main()
