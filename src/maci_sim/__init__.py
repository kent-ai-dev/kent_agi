"""maci_sim — package split of the validated single-file `maci_evince_sim.py`.

A trainable PyTorch simulation of the core thesis of arXiv:2409.01007
(Edward Y. Chang, MACI / Multi-LLM Agent Collaborative Intelligence):
two agents with different priors debate under an information-theoretic
moderator that anneals a contentiousness level kappa.

The single file at the repo root is the source of truth and the Modal
payload; this package reproduces it module-by-module with no logic or
default changes (PLAN.md milestone M1).
"""

from maci_sim.agent import Agent, AgentConfig, TrainConfig, train_agent
from maci_sim.debate import DebateConfig, DebateLog, run_debate
from maci_sim.metrics import decision_mutual_information, entropy, jsd, kl
from maci_sim.world import N_KAPPA_BUCKETS, Vocab, World, WorldConfig, build_training_set

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "DebateConfig",
    "DebateLog",
    "N_KAPPA_BUCKETS",
    "TrainConfig",
    "Vocab",
    "World",
    "WorldConfig",
    "build_training_set",
    "decision_mutual_information",
    "entropy",
    "jsd",
    "kl",
    "run_debate",
    "train_agent",
]
