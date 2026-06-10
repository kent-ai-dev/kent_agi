"""Agent: tiny Transformer encoder classifier with kappa conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from maci_sim.world import Vocab, World, build_training_set


@dataclass
class AgentConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 256
    lr: float = 3e-4
    train_samples: int = 30000
    claim_dropout: float = 0.35    # fraction of training samples with NO opponent claim


class Agent(nn.Module):
    def __init__(self, vocab: Vocab, cfg: AgentConfig, n_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab.size, cfg.d_model, padding_idx=vocab.pad)
        self.pos = nn.Parameter(torch.zeros(1, vocab.max_len, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, n_classes)
        self.pad_idx = vocab.pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.eq(self.pad_idx)
        tok = self.emb(x)
        # Broadcast the kappa-bucket embedding (position 0) to every
        # position so contentiousness conditioning cannot be diluted by
        # pooling — a FiLM-style global conditioning signal.
        tok = tok + tok[:, :1, :]
        h = self.encoder(tok + self.pos[:, : x.size(1)],
                         src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        pooled = h.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def train_agent(name: str, world: World, vocab: Vocab, prior: np.ndarray,
                acfg: AgentConfig, tcfg: TrainConfig, device: str,
                ckpt_path: str, seed: int) -> Agent:
    torch.manual_seed(seed)
    ds = build_training_set(world, vocab, prior, tcfg.train_samples,
                            tcfg.claim_dropout, seed)
    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    model = Agent(vocab, acfg, world.cfg.n_conditions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    model.train()
    for ep in range(tcfg.epochs):
        tot, correct, loss_sum = 0, 0, 0.0
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            tot += y.size(0)
        print(f"[train {name}] epoch {ep + 1}/{tcfg.epochs} "
              f"loss={loss_sum / tot:.4f} acc={correct / tot:.3f}")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[train {name}] saved -> {ckpt_path}")
    return model
