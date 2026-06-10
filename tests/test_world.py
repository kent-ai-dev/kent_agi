"""M2 tests for the synthetic world (PLAN.md)."""

import numpy as np

from maci_sim.world import N_KAPPA_BUCKETS, Vocab, World, WorldConfig, build_training_set


def test_emission_rows_sum_to_one():
    world = World(WorldConfig(seed=7))
    sums = world.emission.sum(axis=1)
    assert np.allclose(sums, 1.0)
    assert (world.emission >= 0).all()


def test_biased_priors_sum_to_one_and_half_skewed():
    cfg = WorldConfig(seed=7)
    world = World(cfg)
    a, b, u = world.biased_priors()
    C = cfg.n_conditions
    for p in (a, b, u):
        assert np.isclose(p.sum(), 1.0)
    # A over-weights the first half 6:1, B the second half 6:1.
    assert np.isclose(a[: C // 2].sum(), 6.0 / 7.0)
    assert np.isclose(b[C // 2:].sum(), 6.0 / 7.0)
    assert np.allclose(u, 1.0 / C)


def test_claim_reliability_matches_schedule():
    """Empirical P(claim correct | kappa-bucket) must match 0.90 - 0.86*kappa
    within +/-0.03 per bucket on 50k samples (claim_dropout=0 so every sample
    carries a claim)."""
    cfg = WorldConfig(seed=7)
    world = World(cfg)
    vocab = Vocab(cfg)
    _, _, uniform = world.biased_priors()
    n = 50_000
    ds = build_training_set(world, vocab, uniform, n, claim_dropout=0.0, seed=123)
    X = ds.tensors[0].numpy()
    y = ds.tensors[1].numpy()

    buckets = X[:, 0] - vocab.kappa_base
    claim_labels = X[:, -1] - vocab.claim_base
    assert ((claim_labels >= 0) & (claim_labels < cfg.n_conditions)).all()

    bucket_width = 0.9 / N_KAPPA_BUCKETS
    for b in range(N_KAPPA_BUCKETS):
        mask = buckets == b
        assert mask.sum() > 5_000  # kappa ~ U[0, 0.9) -> ~12.5k per bucket
        match_rate = float((claim_labels[mask] == y[mask]).mean())
        # An "incorrect" claim is drawn uniformly over C conditions, so it
        # still coincides with the truth 1/C of the time:
        #     P(match) = p + (1 - p)/C  =>  p = (match - 1/C) / (1 - 1/C)
        chance = 1.0 / cfg.n_conditions
        empirical = (match_rate - chance) / (1.0 - chance)
        kappa_center = (b + 0.5) * bucket_width
        expected = 0.90 - 0.86 * kappa_center
        assert abs(empirical - expected) < 0.03, (
            f"bucket {b}: empirical {empirical:.3f} vs expected {expected:.3f}")
