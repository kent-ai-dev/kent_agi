"""EVINCE-lite debate loop: claim exchange under an annealed contentiousness kappa."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from maci_sim.agent import Agent
from maci_sim.metrics import decision_mutual_information, entropy, jsd
from maci_sim.world import Vocab, World


@dataclass
class DebateConfig:
    rounds: int = 6
    kappa_init: float = 0.9
    jsd_eps: float = 0.02          # consensus threshold
    kappa_scale: float = 3.0       # kappa_t = clip(kappa_scale * JSD_t, 0, 0.9)


@dataclass
class DebateLog:
    rounds_run: int = 0
    mean_jsd: list = field(default_factory=list)
    mean_entropy_a: list = field(default_factory=list)
    mean_entropy_b: list = field(default_factory=list)
    decision_mi: list = field(default_factory=list)
    kappa: list = field(default_factory=list)
    wrong_repeat_rate: list = field(default_factory=list)  # Aphorism #11 metric


@torch.no_grad()
def run_debate(agent_a: Agent, agent_b: Agent, world: World, vocab: Vocab,
               X_sym: np.ndarray, y: np.ndarray, dcfg: DebateConfig,
               device: str, moderate: bool = True) -> dict:
    n = X_sym.shape[0]
    C = world.cfg.n_conditions
    L = vocab.max_len
    sym_len = world.cfg.symptoms_per_case

    def encode(claims: np.ndarray | None, kappa: float) -> torch.Tensor:
        X = np.full((n, L), vocab.pad, dtype=np.int64)
        X[:, 0] = vocab.kappa_token(kappa)
        X[:, 1:1 + sym_len] = X_sym
        X[:, 1 + sym_len] = vocab.sep
        X[:, -1] = vocab.noclaim if claims is None else vocab.claim_base + claims
        return torch.from_numpy(X).to(device)

    agent_a.eval(); agent_b.eval()
    log = DebateLog()
    kappa = dcfg.kappa_init

    # Round 0: independent opinions (no opponent claims yet).
    p0A = F.softmax(agent_a(encode(None, kappa)), -1)
    p0B = F.softmax(agent_b(encode(None, kappa)), -1)
    wrong0_A = (p0A.argmax(-1).cpu().numpy() != y)
    wrong0_B = (p0B.argmax(-1).cpu().numpy() != y)
    first_wrong_A = p0A.argmax(-1).cpu().numpy()
    first_wrong_B = p0B.argmax(-1).cpu().numpy()

    # EVINCE-style Bayesian evidence accumulation: each agent maintains a
    # log-belief that integrates every round of dialogue-conditioned
    # evidence (geometric running mean). This is the mechanism behind
    # "the evolving context constrains the hallucination space" — beliefs
    # are refined, never wholesale replaced by the latest utterance.
    prior_A = p0A.clamp_min(1e-12).log()
    prior_B = p0B.clamp_min(1e-12).log()
    logbel_A, logbel_B = prior_A.clone(), prior_B.clone()

    for r in range(dcfg.rounds):
        pA = F.softmax(logbel_A, -1)
        pB = F.softmax(logbel_B, -1)
        aA, aB = pA.argmax(-1).cpu().numpy(), pB.argmax(-1).cpu().numpy()
        j = jsd(pA, pB)
        log.mean_jsd.append(j.mean().item())
        log.mean_entropy_a.append(entropy(pA).mean().item())
        log.mean_entropy_b.append(entropy(pB).mean().item())
        log.decision_mi.append(decision_mutual_information(aA, aB, C))
        log.kappa.append(kappa)
        # Aphorism #11: among cases initially wrong, how often is the SAME
        # wrong answer still the top-1 this round?
        rep_a = float(np.mean(aA[wrong0_A] == first_wrong_A[wrong0_A])) if wrong0_A.any() else 0.0
        rep_b = float(np.mean(aB[wrong0_B] == first_wrong_B[wrong0_B])) if wrong0_B.any() else 0.0
        log.wrong_repeat_rate.append(0.5 * (rep_a + rep_b))
        log.rounds_run = r + 1

        if j.mean().item() < dcfg.jsd_eps:
            break  # consensus reached

        # Exchange current top-1 claims. Each agent extracts the marginal
        # evidence the opponent's claim provides — the log-likelihood ratio
        # log p(y | x, claim) - log p(y | x) at the current kappa — so its
        # own case-based prior is never double-counted across rounds.
        # Each agent re-evaluates the opponent's CURRENT claim at the CURRENT
        # contentiousness level. Its belief is always
        #     own prior + LLR(claim | kappa)
        # i.e. the claim's evidence is re-weighted as kappa anneals (trust
        # grows as the debate converges), never accumulated twice. This is
        # the trainable analogue of EVINCE's behavior modulation: at high
        # kappa the learned LLR is small (claims contested), at low kappa it
        # is large (claims integrated).
        base_A = F.log_softmax(agent_a(encode(None, kappa)), -1)
        base_B = F.log_softmax(agent_b(encode(None, kappa)), -1)
        cond_A = F.log_softmax(agent_a(encode(aB, kappa)), -1)
        cond_B = F.log_softmax(agent_b(encode(aA, kappa)), -1)
        logbel_A = prior_A + (cond_A - base_A)
        logbel_B = prior_B + (cond_B - base_B)
        if moderate:
            kappa = float(np.clip(dcfg.kappa_scale * j.mean().item(), 0.0, 0.9))

    # Final consensus is CLAIM-LEVEL, like real multi-LLM systems where
    # heterogeneous models exchange natural-language claims, not logits:
    #   * if the agents agree, the agreed claim is the answer;
    #   * if not, the moderator sides with the more confident agent
    #     (lower final-belief entropy).
    pA_fin = F.softmax(logbel_A, -1)
    pB_fin = F.softmax(logbel_B, -1)
    aA_fin, aB_fin = pA_fin.argmax(-1), pB_fin.argmax(-1)
    a_conf = entropy(pA_fin) <= entropy(pB_fin)
    consensus_label = torch.where(aA_fin == aB_fin, aA_fin,
                                  torch.where(a_conf, aA_fin, aB_fin))
    agreement_rate = (aA_fin == aB_fin).float().mean().item()

    # Oracle references: pooling full output DISTRIBUTIONS is an upper
    # bound unavailable in real deployments (kept for context).
    naive_arith = 0.5 * (p0A + p0B)
    naive_poe = F.softmax(0.5 * (p0A.clamp_min(1e-12).log()
                                 + p0B.clamp_min(1e-12).log()), -1)
    y_t = torch.from_numpy(y).to(device)

    def acc(p: torch.Tensor) -> float:
        return (p.argmax(-1) == y_t).float().mean().item()

    return {
        "acc_agent_a_alone": acc(p0A),
        "acc_agent_b_alone": acc(p0B),
        "acc_oracle_distribution_pool": acc(naive_arith),
        "acc_oracle_poe": acc(naive_poe),
        "acc_debate_consensus": (consensus_label == y_t).float().mean().item(),
        "final_agreement_rate": agreement_rate,
        # Exposed for tests/analysis; the CLI strips this before writing JSON
        # so the results file stays identical to the single-file version.
        "consensus_labels": consensus_label.cpu().numpy().tolist(),
        "debate_log": asdict(log),
    }
