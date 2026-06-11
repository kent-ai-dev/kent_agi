# Multi-Model Latent Debate (RecursiveMAS Integration)

**Status:** PARKED. Do not implement until Phase 2 single-model gate is cleared.

**Source:** arXiv:2604.25917 — RecursiveMAS (UIUC, Stanford, NVIDIA, MIT)

## Core Idea

Instead of one chat LM, split into specialist agents that communicate via hidden-state vectors (not text). A moderator gates cross-talk using MACI's κ-annealing — high disagreement = confrontational, low = conciliatory. All LLM weights frozen; only tiny projection layers (0.31% of params) are trained.

## Why It Might Fit Later

RecursiveMAS solves expensive vocab-space projection. MACI solves biased-agent coordination. If Phase 2 works, Phase 3 could be:

- Chart encoder → reads price data, emits latent state
- Text reasoner → processes news/sentiment, emits latent state  
- Execution decoder → formats trades from merged latent state
- MACI κ-moderator → gates cross-talk based on JSD between specialist outputs

## Gate: Three Milestones Must Pass First

1. Single model reads a chart and outputs a structured signal
2. Single model holds context across a multi-turn conversation about a position
3. Single model learns from feedback without catastrophic forgetting

Only then does specialist decomposition make sense.

## Saved June 11, 2026
