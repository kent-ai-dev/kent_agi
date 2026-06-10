"""M2 smoke tests for the debate loop with untrained agents (PLAN.md)."""

import numpy as np
import pytest
import torch

from maci_sim.agent import Agent, AgentConfig
from maci_sim.debate import DebateConfig, run_debate
from maci_sim.world import Vocab, World, WorldConfig

LOG_LISTS = ["mean_jsd", "mean_entropy_a", "mean_entropy_b",
             "decision_mi", "kappa", "wrong_repeat_rate"]


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = WorldConfig(seed=0)
    world = World(cfg)
    vocab = Vocab(cfg)
    agent_a = Agent(vocab, AgentConfig(), cfg.n_conditions)
    agent_b = Agent(vocab, AgentConfig(), cfg.n_conditions)
    _, _, uniform = world.biased_priors()
    X, y = world.sample_cases(64, uniform)
    return world, vocab, agent_a, agent_b, X, y


def test_debate_smoke_moderated(setup):
    world, vocab, agent_a, agent_b, X, y = setup
    dcfg = DebateConfig(rounds=3)
    res = run_debate(agent_a, agent_b, world, vocab, X, y, dcfg, "cpu",
                     moderate=True)
    log = res["debate_log"]
    assert 1 <= log["rounds_run"] <= dcfg.rounds
    for key in LOG_LISTS:
        assert len(log[key]) == log["rounds_run"], key
    for acc_key in ("acc_agent_a_alone", "acc_agent_b_alone",
                    "acc_debate_consensus", "final_agreement_rate"):
        assert 0.0 <= res[acc_key] <= 1.0, acc_key


def test_consensus_labels_are_valid_class_indices(setup):
    world, vocab, agent_a, agent_b, X, y = setup
    res = run_debate(agent_a, agent_b, world, vocab, X, y,
                     DebateConfig(rounds=3), "cpu", moderate=True)
    labels = np.asarray(res["consensus_labels"])
    assert labels.shape == y.shape
    assert ((labels >= 0) & (labels < world.cfg.n_conditions)).all()


def test_kappa_fixed_when_unmoderated(setup):
    world, vocab, agent_a, agent_b, X, y = setup
    dcfg = DebateConfig(rounds=4)
    res = run_debate(agent_a, agent_b, world, vocab, X, y, dcfg, "cpu",
                     moderate=False)
    kappas = res["debate_log"]["kappa"]
    assert all(k == dcfg.kappa_init for k in kappas)
