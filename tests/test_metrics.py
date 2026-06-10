"""M2 tests for the information-theoretic moderator metrics (PLAN.md)."""

import math

import numpy as np
import torch

from maci_sim.metrics import decision_mutual_information, jsd


def _random_dists(n: int, c: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, c, generator=g)
    return p / p.sum(-1, keepdim=True)


def test_jsd_symmetric():
    p = _random_dists(64, 8, seed=0)
    q = _random_dists(64, 8, seed=1)
    assert torch.allclose(jsd(p, q), jsd(q, p), atol=1e-6)


def test_jsd_zero_for_identical():
    p = _random_dists(64, 8, seed=2)
    assert jsd(p, p).abs().max().item() < 1e-9


def test_jsd_bounded_by_ln2():
    # Maximally disjoint distributions achieve the ln(2) bound.
    p = torch.zeros(1, 8); p[0, 0] = 1.0
    q = torch.zeros(1, 8); q[0, 7] = 1.0
    assert jsd(p, q).item() <= math.log(2) + 1e-6
    assert abs(jsd(p, q).item() - math.log(2)) < 1e-6
    r = _random_dists(256, 8, seed=3)
    s = _random_dists(256, 8, seed=4)
    assert (jsd(r, s) <= math.log(2) + 1e-6).all()
    assert (jsd(r, s) >= 0).all()


def test_decision_mi_independent_near_zero():
    rng = np.random.default_rng(0)
    C = 8
    a = rng.integers(0, C, size=20_000)
    b = rng.integers(0, C, size=20_000)
    mi = decision_mutual_information(a, b, C)
    # Plug-in MI estimator bias is ~(C-1)^2 / (2N) ~ 0.0012 nats here.
    assert 0.0 <= mi < 0.02


def test_decision_mi_maximal_for_identical():
    rng = np.random.default_rng(1)
    C = 8
    a = rng.integers(0, C, size=20_000)
    mi = decision_mutual_information(a, a, C)
    # MI(X, X) = H(X) = log C for a uniform decision distribution.
    assert abs(mi - math.log(C)) < 0.05
    b = rng.integers(0, C, size=20_000)
    assert mi > decision_mutual_information(a, b, C)
