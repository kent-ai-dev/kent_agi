"""Information-theoretic moderator metrics (EVINCE): entropy, KL, JSD, decision-MI."""

from __future__ import annotations

import math

import numpy as np
import torch


def entropy(p: torch.Tensor) -> torch.Tensor:                 # [B, C] -> [B]
    return -(p.clamp_min(1e-12).log() * p).sum(-1)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:     # KL(p || q), [B]
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1)


def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:    # [B]
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def decision_mutual_information(a: np.ndarray, b: np.ndarray, C: int) -> float:
    """MI (nats) between the two agents' argmax decisions over the eval set."""
    joint = np.zeros((C, C))
    for i, j in zip(a, b):
        joint[i, j] += 1
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    mi = 0.0
    for i in range(C):
        for j in range(C):
            if joint[i, j] > 0:
                mi += joint[i, j] * math.log(joint[i, j] / (pa[i] * pb[j] + 1e-12))
    return mi
