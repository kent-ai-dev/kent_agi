# maci-evince-sim

A small, trainable PyTorch simulation of the core thesis of
[arXiv:2409.01007](https://arxiv.org/abs/2409.01007) (Edward Y. Chang,
*"Unlocking the Wisdom of Large Language Models"* — the **MACI** / Multi-LLM
Agent Collaborative Intelligence framework), focused on the simulable
**EVINCE/SocraSynth** core (Chapters 6–7; Aphorisms #2, #5, #6, #10, #11):

1. Two agents with **different priors/biases** debate a question by exchanging claims.
2. A **moderator** measures the exchange with information theory: per-agent
   entropy, Jensen–Shannon divergence (JSD) between output distributions, and
   mutual information between decisions.
3. A **contentiousness level κ ∈ [0, 0.9]** is annealed from confrontational
   toward conciliatory in proportion to divergence
   (κ_t = clip(scale · JSD_t, 0, 0.9)).
4. **Hallucinations rarely repeat**: the evolving context constrains the error
   space, so wrong claims decay across rounds and the consensus beats either
   agent alone.

## What the simulation does

- **Task:** the paper's medical-triage vignette, made synthetic — 8 latent
  conditions emit overlapping symptom distributions; cases are bags of symptom
  tokens.
- **Agents:** two tiny Transformer encoders (~120k params each). Agent A is
  trained on a shard biased 6:1 toward "common" conditions, Agent B toward
  "rare" ones — giving each genuine blind spots.
- **Learned contentiousness:** during training, each sample carries a synthetic
  "opponent claim" whose reliability depends on a κ-bucket token:
  `P(claim correct | κ) = 0.90 − 0.86κ`. The agent *learns* to integrate
  claims at low κ and ignore them at high κ — the trainable analogue of
  EVINCE's prompt-level behavior modulation.
- **Debate loop (EVINCE-lite):** agents exchange top-1 claims for up to R
  rounds; belief = fixed prior + the opponent-claim's log-likelihood ratio
  re-evaluated at the current κ. The moderator anneals κ ∝ JSD; the loop stops
  at consensus (JSD < ε) or round exhaustion.
- **Consensus is claim-level** (like real heterogeneous multi-LLM systems);
  full-distribution pooling is reported only as an oracle upper bound.

## Quickstart

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                       # creates .venv with torch + numpy (+ dev tools)

# train both agents and run the debate eval (~2 min on CPU)
uv run python -m maci_sim.cli --mode all --epochs 6 \
    --train-samples 24000 --eval-samples 2000 --seed 7

uv run pytest -q              # M2 test suite
uv run python scripts/sweep.py  # M3 overlap x seed grid (~20 min on CPU)
```

The original validated single-file version is kept at the repo root and stays
equivalent to the package:

```bash
uv run python maci_evince_sim.py --mode all --epochs 6 \
    --train-samples 24000 --eval-samples 2000 --seed 7
```

## Validated results (seeds 7 and 13, CPU)

| Metric | seed 7 | seed 13 |
|---|---|---|
| Agent A alone | 0.678 | 0.718 |
| Agent B alone | 0.668 | 0.697 |
| Debate consensus (EVINCE-moderated) | **0.736** | **0.789** |
| Oracle distribution pooling (upper bound) | 0.738 | 0.774 |
| Final agreement rate, moderated vs fixed κ=0.9 | 0.871 vs 0.807 | 0.890 vs 0.812 |
| Wrong-claim repeat rate, round 0 → final | 1.00 → 0.587 | 1.00 → 0.481 |

Three falsifiable thesis checks (printed by the CLI):

1. **Aph. #10** — debate consensus > best single agent (claim-level only).
2. **Ch. 7** — κ-moderation drives consensus *formation*: higher final
   agreement rate than fixed adversarial κ. (Arbitrated accuracy is a
   statistical tie because a confidence tiebreaker rescues the unmoderated
   case — the moderation effect is in how often arbitration is needed at all.)
3. **Aph. #11** — among initially-wrong cases, the rate of repeating the same
   wrong top-1 decays across rounds.

Notable honest finding: claim-level dialogue recovers essentially **all** of
the oracle distribution-pooling accuracy — the debate protocol loses almost
nothing relative to sharing full logits, which heterogeneous real systems
cannot do.

Result artifacts (`results/sample_results.json`, `results/sweep_table.md`,
chat transcripts) are produced by the Modal runs and committed from the VPS —
see `VPS_TRAINING_PROMPT.md` for the agent runbook.

## Repo layout

```
src/maci_sim/           # M1 package split of the validated single file
  world.py              # WorldConfig, World, Vocab, build_training_set
  agent.py              # AgentConfig, TrainConfig, Agent, train_agent
  metrics.py            # entropy, kl, jsd, decision_mutual_information
  debate.py             # DebateConfig, DebateLog, run_debate
  cli.py                # argparse main (mode train|debate|all)
maci_evince_sim.py      # the original validated single-file version (Modal payload)
modal_app.py            # Modal training/eval entrypoints (Phase 1)
train_chat_lm_modal.py  # Phase 2 — chat LM pretrain/chat/continual-learn
scripts/sweep.py        # M3 sweep grid + markdown report
tests/                  # M2 test suite
VPS_TRAINING_PROMPT.md  # runbook for the LLM agent that trains on Modal from the VPS
```

## Running on Modal (M4)

```bash
uv sync --group modal
uv run modal setup        # one-time browser auth

uv run modal run modal_app.py::train_and_evaluate
uv run modal run modal_app.py::sweep                 # M3 grid, one container per cell
uv run modal volume get maci-checkpoints results_seed7.json ./
```

## Phase 2 — a generative agent you can chat with

`train_chat_lm_modal.py` trains a 51M-param GPT from scratch on TinyStories
(Modal A10G, ~2–4 h, < $10), gives it a chat REPL, and adds a continual
learning loop that fine-tunes nightly on your logged conversations:

```bash
uv run modal run train_chat_lm_modal.py::prepare_data       # ~10 min, one-time
uv run modal run train_chat_lm_modal.py::train              # ~2-4 h on A10G
uv run modal run train_chat_lm_modal.py::chat               # interactive REPL
uv run modal run train_chat_lm_modal.py::finetune_on_chats  # the "it learns" pass
```

See `PLAN.md` for the full milestone breakdown (M1–M10), acceptance criteria,
and the implementation guardrails learned during validation.
