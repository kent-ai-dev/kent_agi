# PLAN.md — Repo Initialization Plan: MACI/EVINCE Simulation (arXiv:2409.01007)

> **Audience:** an LLM coding agent (Claude Code or similar) initializing this repository.
> Execute the milestones in order. Acceptance criteria are falsifiable — run them, don't assume them.

---

## 1. Context and core thesis

arXiv:2409.01007 is Edward Y. Chang's **MACI** booklet ("Unlocking the Wisdom of Large
Language Models" / "Multi-LLM Agent Collaborative Intelligence"). It is a *framework*
book, not a trainable-architecture paper. The simulable core (Chapters 6–7, Aphorisms
#2, #5, #6, #10, #11) is the **EVINCE/SocraSynth setup**:

1. Two agents with **different priors/biases** debate a question by exchanging claims.
2. A **moderator** measures the exchange with information theory: per-agent entropy,
   Jensen–Shannon divergence (JSD) between their output distributions, and mutual
   information (MI) between their decisions.
3. A **contentiousness level κ ∈ [0, 0.9]** is annealed from confrontational toward
   conciliatory in proportion to divergence (κ_t = clip(scale · JSD_t, 0, 0.9)).
4. **Hallucinations rarely repeat**: the evolving context constrains the error space,
   so wrong claims decay across rounds and the consensus beats either agent alone.

### What the simulation does (already implemented and validated in `maci_evince_sim.py`)

- **Task:** the paper's own medical-triage vignette, made synthetic: 8 latent
  conditions emit overlapping symptom distributions; cases are bags of symptom tokens.
- **Agents:** two tiny Transformer encoders (~120k params each). Agent A is trained on
  a shard biased 6:1 toward "common" conditions, Agent B toward "rare" ones — giving
  each genuine blind spots (Aphorism #1: truth from perspectives).
- **Learned contentiousness:** during training, each sample carries a synthetic
  "opponent claim" token whose reliability depends on a κ-bucket token:
  `P(claim correct | κ) = 0.90 − 0.86κ` (κ=0 → 90% reliable, κ=0.9 → chance). The
  agent therefore *learns* to integrate claims at low κ and ignore them at high κ —
  the trainable analogue of EVINCE's prompt-level behavior modulation. The κ embedding
  is broadcast to all positions (FiLM-style) so pooling can't dilute it.
- **Debate loop (EVINCE-lite):** agents exchange top-1 claims for up to R rounds. Each
  round, an agent's belief = its fixed prior + the opponent-claim's log-likelihood
  ratio re-evaluated at the current κ (no evidence double-counting). The moderator
  anneals κ ∝ JSD; the loop stops at consensus (JSD < ε) or round exhaustion.
- **Consensus is claim-level** (like real heterogeneous multi-LLM systems, which
  exchange language, not logits): agreed claim if agents agree, else the
  lower-entropy agent's claim. Full-distribution pooling is reported only as an
  oracle upper bound.

### Validated results (seeds 7 and 13, CPU, ~2 min total each)

| Metric | seed 7 | seed 13 |
|---|---|---|
| Agent A alone | 0.678 | 0.718 |
| Agent B alone | 0.668 | 0.697 |
| Debate consensus (EVINCE-moderated) | **0.736** | **0.789** |
| Oracle distribution pooling (upper bound) | 0.738 | 0.774 |
| Final agreement rate, moderated vs fixed κ=0.9 | 0.871 vs 0.807 | 0.890 vs 0.812 |
| Wrong-claim repeat rate, round 0 → final | 1.00 → 0.587 | 1.00 → 0.481 |

Three falsifiable thesis checks, all PASS on both seeds:
1. **Aph. #10** — debate consensus > best single agent (claim-level only).
2. **Ch. 7** — κ-moderation drives consensus *formation*: higher final agreement
   rate than fixed adversarial κ. (Arbitrated accuracy is a statistical tie because a
   confidence tiebreaker rescues the unmoderated case — the moderation effect is in
   how often arbitration is needed at all. Keep this framing; do not "fix" it.)
3. **Aph. #11** — among initially-wrong cases, the rate of repeating the same wrong
   top-1 decays monotonically-ish across rounds.

Notable honest finding: claim-level dialogue recovers essentially **all** of the
oracle distribution-pooling accuracy — the debate protocol loses almost nothing
relative to sharing full logits, which heterogeneous real systems cannot do.

---

## 2. Repository structure to create

```
maci-evince-sim/
├── README.md                  # condensed version of §1 + quickstart + results table
├── PLAN.md                    # this file
├── pyproject.toml             # torch>=2.2, numpy; ruff config; python>=3.11
├── .gitignore                 # checkpoints/, results*.json, __pycache__, .venv
├── src/
│   └── maci_sim/
│       ├── __init__.py
│       ├── world.py           # WorldConfig, World, Vocab, build_training_set
│       ├── agent.py           # AgentConfig, Agent, train_agent
│       ├── metrics.py         # entropy, kl, jsd, decision_mutual_information
│       ├── debate.py          # DebateConfig, DebateLog, run_debate
│       └── cli.py             # argparse main (mode train|debate|all)
├── maci_evince_sim.py         # KEEP: the original validated single-file version
├── modal_app.py               # Modal training/eval entrypoints (spec in §4)
├── train_chat_lm_modal.py     # KEEP: Phase 2 — chat LM pretrain/chat/continual-learn
├── tests/
│   ├── test_world.py
│   ├── test_metrics.py
│   └── test_debate.py
└── results/                   # committed sample_results.json from the validated run
```

**Source of truth:** `maci_evince_sim.py` (provided, validated). Milestone 1 splits it
into `src/maci_sim/` *without changing any logic or defaults*. The single file stays
in the repo root and must keep working — it is the Modal payload and the reference
implementation.

---

## 3. Milestones

### M1 — Package split (no behavior change)
- Split the single file into the modules above; `cli.py` reproduces the exact CLI.
- Acceptance: `python -m maci_sim.cli --mode all --epochs 6 --train-samples 24000
  --eval-samples 2000 --seed 7` prints all three thesis checks PASS, and
  `python maci_evince_sim.py --mode all ...` (untouched) produces statistically
  matching numbers (±1pp).

### M2 — Tests
- `test_world.py`: emission rows sum to 1; biased priors sum to 1 and are
  half-skewed; training-set claim reliability empirically matches
  `0.90 − 0.86κ` within ±0.03 per κ-bucket on 50k samples.
- `test_metrics.py`: JSD symmetric, zero for identical distributions, ≤ ln 2;
  decision-MI zero for independent labels (within noise), maximal for identical.
- `test_debate.py`: smoke test with untrained agents — runs, logs have correct
  lengths, consensus labels are valid class indices; κ stays fixed when
  `moderate=False`.
- Acceptance: `pytest -q` green in < 120 s on CPU.

### M3 — Sweeps and reporting
- `scripts/sweep.py`: grid over `overlap ∈ {0.3, 0.45, 0.6}` × `seed ∈ {7, 13, 21}`,
  writes one JSON per cell plus an aggregate markdown table of the three thesis
  checks. Expectation to verify: the debate-over-single-agent gap *grows* with
  overlap (harder tasks benefit more from collaboration — worth confirming or
  refuting; report either way).
- Acceptance: table generated; checks 1 and 3 PASS in ≥ 8/9 cells; check 2
  (agreement-rate gap) PASS in ≥ 7/9.

### M4 — Modal integration (§4)
- Acceptance: `modal run modal_app.py::train_and_evaluate` completes on a T4 and
  the results JSON lands in the `maci-checkpoints` volume; `modal run
  modal_app.py::sweep` fans out the M3 grid in parallel.

### M5 (stretch, separate branch) — Real-LLM EVINCE
- Replace the two tiny agents with two API LLMs (e.g., Claude Haiku vs another
  vendor's small model) on a QA dataset with ground truth (e.g., MedQA subset).
  Claims are short natural-language assertions; κ maps to a contentiousness phrase
  bank (the paper's Table 1.2); per-question answer distributions are elicited via
  sampled self-consistency (k=10) to compute JSD/entropy. Keep the identical
  moderator and the identical three thesis checks.

---

## Phase 2 — A generative agent you can CHAT with (short-term goal)

**Status of Phase 1 agents:** the classifiers in `maci_evince_sim.py` never produce
language — they emit distributions over 8 diagnosis classes. Chatting requires
generative LMs. Phase 2 trains one from scratch on Modal, gives it a chat REPL, and
adds a continual-learning loop. `train_chat_lm_modal.py` (provided, model code
CPU-validated: forward/backward/generate correct; memorization test loss 5.5 → 0.006)
implements all of M6–M8.

### Honest expectations — when can you chat?

| After | You get | Wall clock | Cost (≈) |
|---|---|---|---|
| M6 `prepare_data` + `train` | Coherent simple English, TinyStories register (grammatical, causally consistent stories). Completion-style chat. | ~10 min + 2–4 h A10G | < $10 |
| M8 first `finetune_on_chats` | Model nudged toward your conversation style/content — the "it learns" part. Weights update between sessions, never during chat (in-context only during a session). | ~10 min A10G per pass | < $1 |
| M10 (optional) instruct pass | Q&A-style turn-taking instead of story continuation. | ~30 min A10G | < $2 |

So: **chat tonight, learning loop tomorrow.** If assistant-grade chat is the real
goal and from-scratch training is not the point, the shortcut is Path B: LoRA-tune
`HuggingFaceTB/SmolLM2-360M-Instruct` (or Qwen2.5-0.5B-Instruct) on
`HuggingFaceTB/smoltalk` — ~1 GPU-hour, but you are adapting a pretrained model,
not training one.

### Architecture (in `train_chat_lm_modal.py`)
- Decoder-only GPT: 8 layers, 8 heads, d=512, ctx 512, GPT-2 BPE via tiktoken,
  weight-tied head → **51.2M params** (~25M non-embedding). bf16 + SDPA + OneCycle.
- Dataset: HF `roneneldan/TinyStories` (~2.1M stories ≈ 450M GPT-2 tokens), packed
  once into a uint16 binary in the `maci-lm` Volume (`prepare_data`).
- Defaults: 14k steps × 56 seqs × 512 tok ≈ 400M tokens ≈ one epoch.

### M6 — Pretrain on Modal
```
modal run train_chat_lm_modal.py::prepare_data
modal run train_chat_lm_modal.py::train
```
- Checkpoint + samples logged every 500 steps; `--resume` flag supported.
- Acceptance: final train loss ≤ 1.9; the periodic "Once upon a time" samples are
  grammatical multi-sentence stories with consistent characters; a held-out 10k-story
  split (add `--val` in repo version) shows val loss within 0.1 of train.

### M7 — Chat REPL
```
modal run train_chat_lm_modal.py::chat
```
- Local input loop → remote A10G generation (container stays warm 5 min between
  turns). Every turn appended to `chat_logs.jsonl` in the Volume.
- Acceptance: multi-turn session where replies stay on-topic with the prompt
  narrative; latency < 5 s/turn warm.

### M8 — Continual learning from chats
```
modal run train_chat_lm_modal.py::finetune_on_chats
```
- Low-LR (5e-5) fine-tune on accumulated chat logs with 25% pretraining-data replay
  to limit forgetting. Run nightly (Modal cron: `modal.Cron("0 3 * * *")`) or on
  demand.
- Acceptance: after seeding ≥ 2k tokens of chats about a distinctive topic, the
  model's unconditional samples mention that topic measurably more often
  (before/after sample comparison committed to `results/`); pretraining val loss
  degrades ≤ 0.15.

### M9 — MACI debate with generative agents (connects Phase 2 back to Phase 1)
- Train TWO copies on biased shards for genuine perspectives — e.g., split
  TinyStories by sentiment/lexical clusters, or shard A = TinyStories +
  `ajibawa-2023/Children-Stories-Collection`, shard B = TinyStories + a contrasting
  HF corpus. Claims = generated continuations; the EVINCE moderator from Phase 1
  computes JSD/entropy/MI over next-token distributions on shared probe prompts and
  anneals a κ system-prefix token (add κ tokens to the vocab during training,
  reliability-conditioning exactly as in Phase 1 guardrail #4).
- Acceptance: the same three thesis checks, evaluated on cloze-style story questions
  with known answers.

### M10 (optional) — Instruct pass for Q&A-style chat
- Fine-tune the M6 checkpoint on `roneneldan/TinyStories-Instruct` (or a small
  slice of `HuggingFaceTB/smoltalk` reformatted to plain text) so the model answers
  questions instead of continuing stories.

### Phase 2 guardrails
1. tiktoken + `datasets` download at container start — keep them in the image, never
   re-download in the hot path.
2. `vol.commit()` after every checkpoint save — Modal volumes are not write-through.
3. Chat context is truncated to the last 480 tokens before generation (block size
   512); the REPL keeps a longer local history but the model cannot see beyond ctx.
4. Replay mixing in `finetune_on_chats` is mandatory; pure chat-log fine-tuning
   catastrophically forgets within a few hundred steps at this scale.

---

## 4. Modal training documentation

Modal runs the training/eval remotely with zero infra setup. The single-file design
makes this trivial: the script is added to the image and invoked as a subprocess (or
imported). Checkpoints and results persist in a Modal Volume.

### 4.1 One-time setup (local machine / VPS)

```bash
pip install modal
modal setup            # opens browser auth; creates ~/.modal.toml
```

### 4.2 `modal_app.py` (create exactly this)

```python
"""Modal entrypoints for the MACI/EVINCE simulation.

Usage:
    modal run modal_app.py::train_and_evaluate
    modal run modal_app.py::train_and_evaluate --epochs 10 --train-samples 100000
    modal run modal_app.py::sweep
    modal volume get maci-checkpoints results_seed7.json ./
"""
import modal

app = modal.App("maci-evince-sim")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy")
    .add_local_file("maci_evince_sim.py", "/root/maci_evince_sim.py")
)

vol = modal.Volume.from_name("maci-checkpoints", create_if_missing=True)

# T4 is ample: each agent is ~120k params. CPU-only also works (drop gpu=)
# and costs less for default sizes; use the GPU for --train-samples >= 200k
# or scaled-up AgentConfig.
@app.function(image=image, gpu="T4", timeout=3600, volumes={"/data": vol})
def train_and_evaluate(epochs: int = 6, train_samples: int = 24000,
                       eval_samples: int = 2000, seed: int = 7,
                       overlap: float = 0.45) -> dict:
    import json, subprocess
    subprocess.run(
        ["python", "/root/maci_evince_sim.py",
         "--mode", "all",
         "--epochs", str(epochs),
         "--train-samples", str(train_samples),
         "--eval-samples", str(eval_samples),
         "--seed", str(seed),
         "--overlap", str(overlap),
         "--ckpt-dir", f"/data/ckpt_seed{seed}",
         "--results", f"/data/results_seed{seed}.json",
         "--device", "cuda"],
        check=True,
    )
    vol.commit()  # persist checkpoints + results
    with open(f"/data/results_seed{seed}.json") as f:
        res = json.load(f)
    return {k: v for k, v in res.items() if k.startswith("acc")
            or k == "final_agreement_rate"}

@app.function(image=image, timeout=7200, volumes={"/data": vol})
def sweep():
    """Fan out the M3 grid in parallel (one container per cell)."""
    grid = [(s, o) for s in (7, 13, 21) for o in (0.3, 0.45, 0.6)]
    results = list(train_and_evaluate.starmap(
        [(6, 24000, 2000, s, o) for s, o in grid]))
    for (s, o), r in zip(grid, results):
        print(f"seed={s} overlap={o}: {r}")

@app.local_entrypoint()
def main(epochs: int = 6, train_samples: int = 24000, seed: int = 7):
    print(train_and_evaluate.remote(epochs=epochs,
                                    train_samples=train_samples, seed=seed))
```

### 4.3 Operational notes

- **Cost:** the default run is ~2 min of T4 (or ~2 min CPU locally) — effectively
  free. The full sweep is 9 parallel containers, a few cents.
- **Retrieving artifacts:** `modal volume get maci-checkpoints results_seed7.json ./`
  or `modal volume ls maci-checkpoints`.
- **Determinism:** the script seeds numpy and torch; expect ±1pp run-to-run variance
  from CUDA nondeterminism. Acceptance criteria use thresholds, not exact values.
- **Scaling up:** to make the GPU earn its keep, raise `WorldConfig.n_conditions`
  to 32, `n_symptoms` to 96, `AgentConfig.d_model` to 256 / `n_layers` to 4, and
  `--train-samples` to 500k. The debate loop and checks need no changes.
- **Pinning:** keep torch pinned in the image; do not let `pip_install` float.

---

## 5. Implementation guardrails (learned during validation — do not regress)

1. **No belief replacement.** Agents must never overwrite their belief with the
   latest conditional output — that creates echo chambers and debate UNDERPERFORMS
   ensembling. Belief = fixed prior + claim-LLR at current κ.
2. **No evidence double-counting.** A claim's LLR is re-evaluated (replaced), never
   re-added, when κ changes. Accumulating the same claim each round causes a
   confirmation cascade and accuracy collapse (observed: 0.728 → 0.688).
3. **κ must be architecturally salient.** Broadcast the κ embedding to all positions.
   With a single pooled token, agents shortcut-learn average trust and the κ token is
   ignored (observed: trust-lift flat at 0.169 vs 0.160 across κ).
4. **Reliability schedule must span chance.** `0.90 − 0.86κ` makes κ=0.9 claims
   uninformative, forcing the model to learn the contentiousness semantics.
5. **Consensus stays claim-level.** Pooling full distributions is the oracle baseline,
   not the system. Report it, labeled as such.
6. **Check 2 is about agreement rate, not arbitrated accuracy.** See §1.

## 6. Definition of done

Phase 1: `README.md` quickstart works on a fresh clone; `pytest` green; local
`--mode all` prints three PASSes; `modal run modal_app.py::train_and_evaluate`
returns the metric dict; sweep table committed under `results/`.

Phase 2 (short-term goal): M6 checkpoint in the `maci-lm` Volume meeting the loss/
coherence acceptance; a transcript of an M7 chat session committed under `results/`;
one M8 before/after learning demonstration committed under `results/`.
