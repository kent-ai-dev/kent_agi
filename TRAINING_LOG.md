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
