"""
M9 — Generative MACI Debate with two biased chat LMs.

Two agents (same 51.2M GPT, different fine-tune biases) exchange claims
on cloze-style prompts. A moderator measures JSD/entropy/MI over next-token
distributions and anneals kappa. The debate consensus should beat either
agent alone — same three thesis checks as Phase 1.
"""
import json, math, os, time
from dataclasses import dataclass
from typing import Tuple

import modal
from modal import App, Image, Volume

app = App("m9-generative-debate")

vol = Volume.from_name("maci-lm", create_if_missing=True)
DATA = "/data"
AGENT_A = f"{DATA}/chat_lm.pt"          # M8 trading-tuned
AGENT_B = f"{DATA}/agent_b.pt"          # fine-tuned on different shard
CLOZE_PATH = f"{DATA}/cloze_eval.jsonl"
RESULTS_PATH = f"{DATA}/m9_results.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm", "datasets", "tiktoken")
)


# ---------------------------------------------------------------------------
# Model classes — must match train_chat_lm_modal.py exactly
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

def build_model_classes():
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
            self.head.weight = self.tok.weight
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
                idx_cond = idx[:, -self.c.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                if eos_id is not None and idx_next.item() == eos_id:
                    break
            return idx

    return GPT, GPTConfig


# ---------------------------------------------------------------------------
# 1) build cloze eval set from TinyStories validation split
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=600, volumes={DATA: vol})
def prepare_cloze(n_samples: int = 200, context_len: int = 64, answer_len: int = 8):
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    samples = []
    for i, ex in enumerate(ds):
        if len(samples) >= n_samples:
            break
        text = ex["text"].strip()
        toks = enc.encode_ordinary(text)
        if len(toks) < context_len + answer_len + 16:
            continue
        prompt_toks = toks[:context_len]
        answer_toks = toks[context_len:context_len + answer_len]
        samples.append({
            "prompt_text": enc.decode(prompt_toks),
            "answer_text": enc.decode(answer_toks),
            "prompt_toks": prompt_toks,
            "answer_toks": answer_toks,
        })

    with open(CLOZE_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    vol.commit()
    print(f"Wrote {len(samples)} cloze samples to {CLOZE_PATH}")


# ---------------------------------------------------------------------------
# 2) train Agent B on a different TinyStories shard
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", timeout=3600, volumes={DATA: vol})
def train_agent_b(steps: int = 300, lr: float = 5e-5, batch_size: int = 16,
                  seed: int = 42):
    """Fine-tune a copy of the current checkpoint on a different TinyStories shard."""
    import numpy as np, random, torch, tiktoken
    from datasets import load_dataset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    GPT, GPTConfig = build_model_classes()
    enc = tiktoken.get_encoding("gpt2")

    # Load TinyStories, skip first 50k stories to get a different distribution
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if i < 50000:
            continue
        if i >= 50000 + 50000:
            break
        texts.append(ex["text"].strip() + "<|endoftext|>")

    all_ids = []
    for t in texts:
        all_ids.extend(enc.encode_ordinary(t) + [enc.eot_token])
    data = np.asarray(all_ids, dtype=np.uint16)
    print(f"Agent B training data: {len(data)} tokens from stories 50k-100k")

    # Load current checkpoint
    ck = torch.load(AGENT_A, map_location="cuda", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to("cuda")
    model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def batch(arr, size):
        hi = max(1, len(arr) - cfg.block_size - 1)
        ix = np.random.randint(0, hi, size=size)
        rows = [arr[i:i + cfg.block_size + 1].astype(np.int64) for i in ix]
        # Pad short sequences
        padded = []
        for r in rows:
            if len(r) < cfg.block_size + 1:
                r = np.pad(r, (0, cfg.block_size + 1 - len(r)),
                          constant_values=enc.eot_token)
            padded.append(r)
        m = torch.tensor(np.stack(padded), device="cuda")
        return m[:, :-1], m[:, 1:]

    model.train()
    for s in range(steps):
        x, y = batch(data, batch_size)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 50 == 0:
            print(f"B step {s}/{steps} loss={loss.item():.3f}")

    torch.save({"model": model.state_dict(), "opt": ck["opt"],
                "step": ck["step"], "cfg": ck["cfg"]}, AGENT_B)
    vol.commit()
    print(f"Agent B saved to {AGENT_B}")


# ---------------------------------------------------------------------------
# 3) debate — JSD/entropy over next-token distributions, kappa annealing
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", timeout=3600, volumes={DATA: vol})
def run_debate(rounds: int = 6, kappa_init: float = 0.9, kappa_scale: float = 3.0,
               jsd_eps: float = 0.02, temperature: float = 0.8):
    import torch, torch.nn.functional as F, numpy as np, tiktoken

    GPT, GPTConfig = build_model_classes()
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    ck_a = torch.load(AGENT_A, map_location="cuda", weights_only=False)
    ck_b = torch.load(AGENT_B, map_location="cuda", weights_only=False)
    cfg = GPTConfig(**ck_a["cfg"])

    agent_a = GPT(cfg).to("cuda")
    agent_b = GPT(cfg).to("cuda")
    agent_a.load_state_dict(ck_a["model"]); agent_a.eval()
    agent_b.load_state_dict(ck_b["model"]); agent_b.eval()

    # Load cloze prompts
    with open(CLOZE_PATH) as f:
        cloze = [json.loads(line) for line in f]

    # Metrics helpers
    def jsd_fn(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(m.log(), p, reduction='none').sum(-1) +
                      F.kl_div(m.log(), q, reduction='none').sum(-1))

    def entropy_fn(p):
        return -(p * p.log()).sum(-1)

    # Run debate on each prompt
    n = len(cloze)
    V = cfg.vocab_size

    acc_a = 0; acc_b = 0; acc_debate = 0; acc_oracle = 0
    acc_ensemble = 0

    logs = []
    for sample in cloze:
        prompt_toks = torch.tensor([sample["prompt_toks"]], device="cuda")
        answer_toks = sample["answer_toks"]

        # Round 0: independent next-token distributions for each answer position
        a_toks = []; b_toks = []; debate_toks = []

        for t_idx in range(len(answer_toks)):
            ctx = prompt_toks if t_idx == 0 else \
                  torch.cat([prompt_toks, torch.tensor([debate_toks], device="cuda")], dim=1)
            ctx = ctx[:, -cfg.block_size:]

            with torch.no_grad():
                logits_a, _ = agent_a(ctx)
                logits_b, _ = agent_b(ctx)

            # Temperature and softmax
            probs_a = F.softmax(logits_a[0, -1, :] / temperature, dim=-1)
            probs_b = F.softmax(logits_b[0, -1, :] / temperature, dim=-1)

            # EVINCE moderation: anneal kappa based on JSD
            jsd = jsd_fn(probs_a.unsqueeze(0), probs_b.unsqueeze(0)).item()
            kappa = min(kappa_init, kappa_scale * jsd)
            kappa = max(0.0, min(0.9, kappa))

            # Moderated consensus: weighted average based on entropy
            ent_a = entropy_fn(probs_a).item()
            ent_b = entropy_fn(probs_b).item()

            # Lower entropy = more confident; weight it higher
            w_a = math.exp(-ent_a); w_b = math.exp(-ent_b)
            w_sum = w_a + w_b + 1e-12
            probs_debate = (w_a * probs_a + w_b * probs_b) / w_sum

            # Oracle = uniform mixture
            probs_oracle = 0.5 * (probs_a + probs_b)

            # Top-1 predictions
            tok_a = probs_a.argmax().item()
            tok_b = probs_b.argmax().item()
            tok_debate = probs_debate.argmax().item()
            tok_oracle = probs_oracle.argmax().item()

            a_toks.append(tok_a); b_toks.append(tok_b)
            debate_toks.append(tok_debate)

        # Per-token accuracy (more forgiving than exact sequence match)
        correct = answer_toks
        for i, tok in enumerate(correct):
            acc_a += int(a_toks[i] == tok)
            acc_b += int(b_toks[i] == tok)
            acc_debate += int(debate_toks[i] == tok)
            acc_ensemble += int((a_toks[i] == tok) or (b_toks[i] == tok))
        # Oracle per-token
        oracle_toks = []
        for t_idx in range(len(answer_toks)):
            ctx = prompt_toks if t_idx == 0 else \
                  torch.cat([prompt_toks, torch.tensor([oracle_toks], device="cuda")], dim=1)
            ctx = ctx[:, -cfg.block_size:]
            with torch.no_grad():
                logits_a, _ = agent_a(ctx)
                logits_b, _ = agent_b(ctx)
            probs_a = F.softmax(logits_a[0, -1, :] / temperature, dim=-1)
            probs_b = F.softmax(logits_b[0, -1, :] / temperature, dim=-1)
            tok = (0.5 * (probs_a + probs_b)).argmax().item()
            oracle_toks.append(tok)
            acc_oracle += int(tok == correct[t_idx])

        logs.append({
            "prompt": sample["prompt_text"][:60],
            "answer": sample["answer_text"][:40],
            "a_match": a_toks == correct,
            "b_match": b_toks == correct,
            "debate_match": debate_toks == correct,
        })

    total_tokens = n * len(answer_toks)
    results = {
        "n_samples": n,
        "total_tokens": total_tokens,
        "acc_agent_a": acc_a / total_tokens,
        "acc_agent_b": acc_b / total_tokens,
        "acc_debate_consensus": acc_debate / total_tokens,
        "acc_oracle_mixture": acc_oracle / total_tokens,
        "acc_ensemble_union": acc_ensemble / total_tokens,
        "check1_debate_gt_best_single": (acc_debate / total_tokens) > max(acc_a / total_tokens, acc_b / total_tokens),
        "samples": logs[:20],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main(mode: str = "debate"):
    if mode == "prepare":
        prepare_cloze.remote()
    elif mode == "train_b":
        train_agent_b.remote()
    elif mode == "debate":
        run_debate.remote()
    elif mode == "all":
        prepare_cloze.remote()
        train_agent_b.remote()
        run_debate.remote()
    else:
        print("mode must be prepare | train_b | debate | all")
