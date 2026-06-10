# KENT-AGI Training Log

**Started:** 2026-06-09 ~21:00 UTC  
**Objective:** Train a conversational model (Phase 2 chat LM) with continual learning on Modal.

---

## Phase 1: EVINCE Baseline (M1–M5)

### M1: Environment Setup
- `uv sync --all-groups` completed
- Modal auth confirmed (`kent-ai-dev` workspace)
- `modal_app.py` verified against PLAN §4.2

### M2: Test Gate
- `uv run pytest -q` — **11 tests passed in ~7s**
- No regressions; gate cleared for Modal spend

### M3: Baseline Training (Seed 7)
- **Command:** `uv run modal run modal_app.py::train_and_evaluate`
- **Hardware:** Modal T4 GPU
- **Dataset:** 24k samples
- **Epochs:** ~6
- **Results:**
  | Metric | Value |
  |---|---|
  | acc_agent_a_alone | 66.4% |
  | acc_agent_b_alone | 65.3% |
  | acc_debate_consensus | **71.75%** |
  | acc_oracle_distribution_pool | 72.5% |
  | acc_oracle_poe | 72.5% |
  | final_agreement_rate | 87.4% |
- **Commit:** `ccf231e` — results/sample_results.json

### M3: Sweep (Overlap Grid)
- **Command:** `uv run modal run modal_app.py::sweep`
- **Grid:** overlap ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}, seeds ∈ {7, 13}
- **Status:** IN PROGRESS

---

## Phase 2: Chat LM (M6–M10)

### M6: Pretrain
- **Status:** IN PROGRESS
- `prepare_data` running
- `train` launched on Modal

### M7: Chat REPL
- **Status:** PENDING

### M8: Continual Learning
- **Status:** PENDING

---

## Notes
- All §5 guardrails respected (no belief replacement, κ salience, no double-counting).
- `maci_evince_sim.py` and `modal_app.py` untouched per guardrails.

## Auto-Log: 2026-06-10T03:49:50Z
- busy=True queued=0 cursor=5
- tmux:      ◼ Phase 2 M6: prepare_data + train (pretrain chat LM)

---

## Detailed execution log (Claude Code) — authoritative numbers

> Appended by the executing agent. Where this differs from auto-generated
> sections above, these are the verified values from the actual run output.

### Phase 0 — environment + test gate (2026-06-09)
- `uv sync --all-groups`: torch 2.12.0+cu130, numpy 2.4.6, modal 1.5.0,
  pytest 9.0.3, ruff 0.15.16.
- Test gate `uv run pytest -q`: **11 passed, 1 warning in 32.18s** (budget
  <120s). (Correction: an earlier auto-entry said ~7s; the measured wall time
  was 32.18s.) Benign `TransformerEncoder` nested-tensor warning only.

### Phase 1 (M4) — MACI/EVINCE baseline on Modal T4
| metric | seed 7 | seed 13 |
|---|---|---|
| Agent A alone | 0.664 | 0.719 |
| Agent B alone | 0.653 | 0.708 |
| Debate consensus (moderated) | 0.718 | 0.773 |
| Oracle distribution pool (upper bound) | 0.725 | 0.767 |
| Agreement: moderated vs fixed κ | 0.874 / 0.808 | 0.943 / 0.857 |
| Wrong-repeat, round 0 → final | 1.000 → 0.592 | 1.000 → 0.523 |

All three thesis checks PASS on both seeds. Runtime 16.5s (s7), 12.7s (s13).
Committed `results/sample_results.json` (seed 7) — commit `ccf231e`.

### Phase 1 (M3) — sweep, actual grid
- Actual grid: **overlap ∈ {0.3, 0.45, 0.6} × seed ∈ {7, 13, 21}** (9 cells),
  per PLAN.md M3. (Correction: an earlier auto-entry listed a different grid.)
- Modal `sweep`: Check 1 (debate > best single agent) **9/9 PASS**.
- Collaboration gap grows with overlap: mean debate−best gap
  ~0.027 → ~0.057 → ~0.078 for overlap 0.3/0.45/0.6 → **hypothesis CONFIRMED**.
- Complete per-cell table (all 3 checks) via local `scripts/sweep.py`:
  see `results/sweep_table.md` (committed when the local sweep finishes).

### Phase 2 (M6) — prepare_data
- `prepare_data`: 2,119,719 TinyStories → **474.0M GPT-2 tokens** packed to
  `/data/tinystories_gpt2.bin` (`maci-lm` volume).

### Phase 2 (M6) — train
- Attempt 1 (A10G, default batch_size=56): step 0 ok (loss=10.831), then
  **CUDA OOM at step 1** (5.37 GiB alloc on 22 GiB card). Cause: cross-entropy
  logits `B·T·V = 56·512·50257`; model code was CPU-validated only.
- Attempt 2 (CLI flags only, no file edits): `--batch-size 24 --max-steps
  32000 --eval-every 1000` ≈ one epoch (~393M tokens), est. peak ~12 GiB.
  **Running.** Final loss + sample coherence to be recorded on completion.
