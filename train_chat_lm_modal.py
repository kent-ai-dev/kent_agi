"""
train_chat_lm_modal.py
======================
Phase 2 of the MACI/EVINCE repo: train a small generative GPT FROM SCRATCH on
HuggingFace `roneneldan/TinyStories`, chat with it, and let it keep learning
from your conversations via periodic fine-tuning on logged chats.

Everything runs on Modal; checkpoints, the packed token file, and chat logs
live in the `maci-lm` Volume.

Quickstart
----------
    pip install modal && modal setup

    modal run train_chat_lm_modal.py::prepare_data      # ~10 min, one-time
    modal run train_chat_lm_modal.py::train             # ~2-4 h on A10G
    modal run train_chat_lm_modal.py::chat              # interactive REPL
    modal run train_chat_lm_modal.py::finetune_on_chats # "learning" pass

Expectations (be honest with yourself):
  * After `train`, the model writes coherent simple English in TinyStories
    register — grammatical, causally consistent short stories. It is NOT an
    assistant; chat works completion-style.
  * "Learning" happens only in `finetune_on_chats` (weight updates on your
    logged conversations), never during the chat itself (in-context only).

The model class below doubles as the generative agent for the MACI debate
layer (PLAN.md milestone M9): two of these trained on different data shards
exchange generated claims, with the same EVINCE moderator.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict

import modal

app = modal.App("maci-chat-lm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy", "datasets==2.21.0", "tiktoken==0.7.0")
)
vol = modal.Volume.from_name("maci-lm", create_if_missing=True)

DATA = "/data"
TOKENS_PATH = f"{DATA}/tinystories_gpt2.bin"   # packed uint16 token stream
CKPT_PATH = f"{DATA}/chat_lm.pt"
CHATLOG_PATH = f"{DATA}/chat_logs.jsonl"

# ----------------------------------------------------------------------------
# Model (plain decoder-only GPT; no external deps, importable by the MACI repo)
# ----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50257        # GPT-2 BPE via tiktoken
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    # 51M params at these settings (≈25M non-embedding) — comfortably in
    # TinyStories-coherence territory (the paper shows coherence from ~8M).


def build_model_classes():
    """Defined inside a function so this file imports without torch locally;
    on Modal (and in the repo's src/) call this once to get the classes."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, c: GPTConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(c.n_embd)
            self.attn = nn.Linear(c.n_embd, 3 * c.n_embd)
            self.proj = nn.Linear(c.n_embd, c.n_embd)
            self.ln2 = nn.LayerNorm(c.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                nn.Linear(4 * c.n_embd, c.n_embd))
            self.n_head, self.drop = c.n_head, c.dropout

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
            q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                       for t in (q, k, v))
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.drop if self.training else 0.0)
            x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self, c: GPTConfig):
            super().__init__()
            self.c = c
            self.tok = nn.Embedding(c.vocab_size, c.n_embd)
            self.pos = nn.Embedding(c.block_size, c.n_embd)
            self.drop = nn.Dropout(c.dropout)
            self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
            self.lnf = nn.LayerNorm(c.n_embd)
            self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
            self.head.weight = self.tok.weight  # weight tying
            self.apply(self._init)

        def _init(self, m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.drop(self.tok(idx)
                          + self.pos(torch.arange(T, device=idx.device)))
            for b in self.blocks:
                x = b(x)
            logits = self.head(self.lnf(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.reshape(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens=200, temperature=0.8,
                     top_k=50, eos_id=None):
            self.eval()
            for _ in range(max_new_tokens):
                ctx = idx[:, -self.c.block_size:]
                logits, _ = self(ctx)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                idx = torch.cat([idx, nxt], dim=1)
                if eos_id is not None and int(nxt) == eos_id:
                    break
            return idx

    return GPT


# ----------------------------------------------------------------------------
# 1) prepare_data — one-time: download, tokenize, pack to a uint16 stream
# ----------------------------------------------------------------------------

@app.function(image=image, timeout=3600, volumes={DATA: vol}, cpu=8.0,
              memory=16384)
def prepare_data(max_stories: int = 0):
    """Tokenize TinyStories with GPT-2 BPE into one packed binary.
    `max_stories=0` means the full ~2.1M-story train split (~450M tokens)."""
    import numpy as np
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if max_stories:
        ds = ds.select(range(max_stories))

    buf, total, t0 = [], 0, time.time()
    with open(TOKENS_PATH + ".tmp", "wb") as f:
        for i, row in enumerate(ds):
            ids = enc.encode_ordinary(row["text"]) + [eot]
            buf.extend(ids)
            if len(buf) > 5_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                total += len(buf); buf = []
            if i % 200_000 == 0:
                print(f"{i} stories, {total/1e6:.0f}M tokens,"
                      f" {time.time()-t0:.0f}s")
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
    os.replace(TOKENS_PATH + ".tmp", TOKENS_PATH)
    vol.commit()
    print(f"done: {total/1e6:.1f}M tokens -> {TOKENS_PATH}")
    return total


# ----------------------------------------------------------------------------
# 2) train — from-scratch pretraining (~2-4 h on A10G for one epoch)
# ----------------------------------------------------------------------------

@app.function(image=image, gpu="A10G", timeout=6 * 3600, volumes={DATA: vol})
def train(max_steps: int = 14000, batch_size: int = 24, lr: float = 6e-4,
          eval_every: int = 500, resume: bool = False):
    """A10G ≈ 71k tok/s with batch_size 24: a full epoch (32k steps ×
    24 × 512 ≈ 393M tokens) finishes in ~92 min, well under the 6h timeout.
    The VRAM auto-cap below keeps batch_size safe on smaller GPUs too."""
    import numpy as np
    import torch

    GPT = build_model_classes()
    cfg = GPTConfig()
    dev = "cuda"
    data = np.memmap(TOKENS_PATH, dtype=np.uint16, mode="r")
    print(f"tokens on disk: {len(data)/1e6:.1f}M")

    # Auto-detect GPU memory and cap batch_size if needed
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Rough heuristic: each sample needs ~0.6 GB peak (logits + activations)
        safe_batch = max(1, int(total_vram * 0.85 / 0.6))
        if batch_size > safe_batch:
            print(f"WARNING: batch_size {batch_size} may OOM on {total_vram:.1f}GB GPU. "
                  f"Capping to {safe_batch}.")
            batch_size = safe_batch
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {total_vram:.1f}GB | "
              f"batch_size: {batch_size}")
    else:
        print("WARNING: No CUDA available. Training on CPU will be extremely slow.")
        batch_size = 4

    def get_batch():
        ix = np.random.randint(0, len(data) - cfg.block_size - 1,
                               size=batch_size)
        x = torch.stack([torch.from_numpy(
            data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    model = GPT(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max_steps, pct_start=0.02)
    scaler = torch.amp.GradScaler("cuda")
    step0 = 0
    if resume and os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"resumed at step {step0}")

    model.train(); t0 = time.time()
    for step in range(step0, max_steps):
        x, y = get_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if step % eval_every == 0 or step == max_steps - 1:
            tok_s = (step - step0 + 1) * batch_size * cfg.block_size \
                    / (time.time() - t0)
            print(f"step {step}/{max_steps} loss={loss.item():.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {tok_s/1e3:.0f}k tok/s")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": asdict(cfg)}, CKPT_PATH)
            vol.commit()
            print(sample(model, "Once upon a time", dev)[:400], "\n---")
    print(f"finished in {(time.time()-t0)/60:.0f} min; ckpt -> {CKPT_PATH}")


def sample(model, prompt: str, dev: str, max_new: int = 120) -> str:
    import tiktoken
    import torch
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=dev)
    out = model.generate(idx, max_new_tokens=max_new, eos_id=enc.eot_token)
    return enc.decode(out[0].tolist())


# ----------------------------------------------------------------------------
# 3) chat — remote generation + local REPL; every turn is logged to the volume
# ----------------------------------------------------------------------------

@app.function(image=image, gpu="A10G", timeout=600, volumes={DATA: vol},
              scaledown_window=300)
def generate_remote(prompt: str, max_new: int = 160,
                    temperature: float = 0.8) -> str:
    import tiktoken
    import torch
    GPT = build_model_classes()
    ck = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).to("cuda")
    model.load_state_dict(ck["model"])
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode_ordinary(prompt)[-480:]], device="cuda")
    out = model.generate(idx, max_new_tokens=max_new,
                         temperature=temperature, eos_id=enc.eot_token)
    text = enc.decode(out[0].tolist())[len(enc.decode(idx[0].tolist())):]
    with open(CHATLOG_PATH, "a") as f:
        f.write(json.dumps({"t": time.time(), "prompt": prompt,
                            "reply": text}) + "\n")
    vol.commit()
    return text.strip()


@app.local_entrypoint()
def test(prompt: str = "Once upon a time", max_new: int = 60):
    print("\nPROMPT:", prompt)
    reply = generate_remote.remote(prompt, max_new=max_new)
    print("RESPONSE:", reply)


@app.local_entrypoint()
def chat():
    """Completion-style chat. The model is a story LM: phrase turns as
    narrative ('Tom asked: ...') for best results."""
    history = ""
    print("Chatting with your TinyStories GPT (Ctrl-C to quit).")
    while True:
        user = input("\nyou > ").strip()
        if not user:
            continue
        history += f"\n{user}\n"
        reply = generate_remote.remote(history[-1800:])
        history += reply
        print(f"bot > {reply}")


# ----------------------------------------------------------------------------
# 4) finetune_on_chats — the "it learns" loop (run nightly / on demand)
# ----------------------------------------------------------------------------

@app.function(image=image, gpu="A10G", timeout=1800, volumes={DATA: vol})
def finetune_on_chats(steps: int = 300, lr: float = 5e-5,
                      batch_size: int = 16):
    """Weight updates from logged conversations. Low LR + few steps to nudge
    style/content toward your chats without catastrophic forgetting; mixes
    25% original pretraining batches as replay."""
    import numpy as np
    import tiktoken
    import torch

    if not os.path.exists(CHATLOG_PATH):
        print("no chat logs yet"); return
    GPT = build_model_classes()
    enc = tiktoken.get_encoding("gpt2")
    ids = []
    with open(CHATLOG_PATH) as f:
        for line in f:
            r = json.loads(line)
            ids.extend(enc.encode_ordinary(r["prompt"] + r["reply"])
                       + [enc.eot_token])
    chat_data = np.asarray(ids, dtype=np.uint16)
    base_data = np.memmap(TOKENS_PATH, dtype=np.uint16, mode="r")
    print(f"chat tokens: {len(chat_data)}")
    if len(chat_data) < 2048:
        print("not enough chat data to learn from yet"); return

    ck = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to("cuda"); model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def batch_from(arr):
        hi = max(1, len(arr) - cfg.block_size - 1)
        ix = np.random.randint(0, hi, size=batch_size)
        pad = lambda a: np.pad(a, (0, cfg.block_size + 1 - len(a)),
                               constant_values=enc.eot_token)
        rows = [pad(arr[i:i + cfg.block_size + 1].astype(np.int64))
                for i in ix]
        m = torch.tensor(np.stack(rows), device="cuda")
        return m[:, :-1], m[:, 1:]

    model.train()
    for s in range(steps):
        x, y = batch_from(base_data if s % 4 == 0 else chat_data)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 50 == 0:
            print(f"ft step {s}/{steps} loss={loss.item():.3f}")
    torch.save({"model": model.state_dict(), "opt": ck["opt"],
                "step": ck["step"], "cfg": ck["cfg"]}, CKPT_PATH)
    vol.commit()
    print("fine-tuned checkpoint saved")
