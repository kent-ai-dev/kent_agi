"""Synthetic world: latent conditions emit symptoms; agents see biased shards.

Token layout:
    [0, n_symptoms)                      symptom tokens
    [n_symptoms, n_symptoms+n_cond)      claim tokens (one per condition)
    then: PAD, SEP, NOCLAIM, K0..K3 (kappa buckets)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

N_KAPPA_BUCKETS = 4


@dataclass
class WorldConfig:
    n_conditions: int = 8          # latent classes ("diagnoses")
    n_symptoms: int = 24           # observable symptom vocabulary
    symptoms_per_case: int = 10    # tokens observed per case
    overlap: float = 0.45          # how much condition symptom profiles overlap (0..1)
    seed: int = 7


class Vocab:
    def __init__(self, w: WorldConfig):
        self.n_symptoms = w.n_symptoms
        self.n_conditions = w.n_conditions
        self.claim_base = w.n_symptoms
        self.pad = w.n_symptoms + w.n_conditions
        self.sep = self.pad + 1
        self.noclaim = self.sep + 1
        self.kappa_base = self.noclaim + 1
        self.size = self.kappa_base + N_KAPPA_BUCKETS
        self.max_len = 1 + w.symptoms_per_case + 1 + 1  # kappa + symptoms + SEP + claim

    def kappa_token(self, kappa: float) -> int:
        b = min(N_KAPPA_BUCKETS - 1, int(kappa / (0.9 / N_KAPPA_BUCKETS + 1e-9)))
        return self.kappa_base + b


class World:
    """Generative model: condition c -> categorical over symptoms."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        C, S = cfg.n_conditions, cfg.n_symptoms
        # Each condition gets a "signature" block of symptoms plus a shared
        # background, controlled by `overlap`. Higher overlap -> harder task.
        base = rng.dirichlet(np.ones(S) * 0.5, size=C)            # idiosyncratic
        shared = rng.dirichlet(np.ones(S) * 0.5)                  # common background
        self.emission = (1 - cfg.overlap) * base + cfg.overlap * shared
        self.emission /= self.emission.sum(axis=1, keepdims=True)
        self.rng = rng

    def sample_cases(self, n: int, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (symptom token matrix [n, L], condition labels [n])."""
        cfg = self.cfg
        conds = self.rng.choice(cfg.n_conditions, size=n, p=prior)
        rows = np.empty((n, cfg.symptoms_per_case), dtype=np.int64)
        for i, c in enumerate(conds):
            rows[i] = self.rng.choice(cfg.n_symptoms, size=cfg.symptoms_per_case,
                                      p=self.emission[c])
        return rows, conds

    def biased_priors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Agent A over-samples the first half of conditions ('common'),
        Agent B the second half ('rare'); eval prior is uniform."""
        C = self.cfg.n_conditions
        a = np.full(C, 1.0); a[: C // 2] *= 6.0; a /= a.sum()
        b = np.full(C, 1.0); b[C // 2:] *= 6.0; b /= b.sum()
        u = np.full(C, 1.0 / C)
        return a, b, u


def build_training_set(world: World, vocab: Vocab, prior: np.ndarray,
                       n: int, claim_dropout: float, seed: int) -> TensorDataset:
    """Encode cases with kappa-conditioned synthetic opponent claims.

    Claim reliability schedule (the learned contentiousness semantics):
        P(claim is correct | kappa) = 0.90 - 0.86 * kappa
    -> kappa=0.0: claims correct 90% of the time (trust/integrate)
    -> kappa=0.9: claims at chance (~12.6%) (contest/ignore)
    """
    rng = np.random.default_rng(seed)
    X_sym, y = world.sample_cases(n, prior)
    L = vocab.max_len
    X = np.full((n, L), vocab.pad, dtype=np.int64)

    kappas = rng.uniform(0.0, 0.9, size=n)
    for i in range(n):
        k = kappas[i]
        X[i, 0] = vocab.kappa_token(k)
        X[i, 1:1 + world.cfg.symptoms_per_case] = X_sym[i]
        X[i, 1 + world.cfg.symptoms_per_case] = vocab.sep
        if rng.random() < claim_dropout:
            claim = vocab.noclaim
        else:
            p_correct = 0.90 - 0.86 * k
            if rng.random() < p_correct:
                claim_label = y[i]
            else:
                claim_label = rng.integers(0, world.cfg.n_conditions)
            claim = vocab.claim_base + int(claim_label)
        X[i, -1] = claim
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
