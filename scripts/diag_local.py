"""Local CPU inspection of the pulled checkpoint (/tmp/chat_lm.pt).

Run: uv run --with tiktoken --with numpy python scripts/diag_local.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_chat_lm_modal import GPTConfig, build_model_classes  # noqa: E402

ck = torch.load("/tmp/chat_lm.pt", map_location="cpu")
print("checkpoint step:", ck.get("step"))
print("checkpoint cfg :", ck.get("cfg"))
print("state_dict keys:", list(ck["model"].keys())[:6], "...",
      len(ck["model"]), "total")

GPT = build_model_classes()
model = GPT(GPTConfig(**ck["cfg"]))
inc = model.load_state_dict(ck["model"], strict=False)
print("missing keys   :", inc.missing_keys)
print("unexpected keys:", inc.unexpected_keys)

tw = model.tok.weight
print(f"tok.weight  mean={float(tw.mean()):.4f} std={float(tw.std()):.4f}")
# a non-embedding trained weight, to confirm real training
b0 = model.blocks[0].attn.weight
print(f"block0 attn mean={float(b0.mean()):.4f} std={float(b0.std()):.4f}")
print("head tied to tok:", model.head.weight is model.tok.weight)

import tiktoken  # noqa: E402

enc = tiktoken.get_encoding("gpt2")
idx = torch.tensor([enc.encode_ordinary("Once upon a time")])
out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_k=50,
                     eos_id=enc.eot_token)
print("SAMPLE:", enc.decode(out[0].tolist()))
