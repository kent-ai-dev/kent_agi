# VPS_TRAINING_PROMPT.md — Run all training on Modal from the VPS

> **Copy everything below this line into an LLM coding agent (Claude Code or
> similar) running on the VPS.** The repo is the only thing it needs besides a
> Modal API token.

---

You are an LLM coding agent on a headless Linux VPS. Your job is to execute
the training milestones of this repository on **Modal** (remote GPU/CPU — the
VPS itself never trains anything), validate the results against the acceptance
criteria, and commit the result artifacts back to the repo.

## Documents to read first (in this order)

1. `PLAN.md` — the full milestone plan (M1–M10), acceptance criteria, and the
   **implementation guardrails in §5** (do not regress them).
2. `README.md` — quickstart, validated reference numbers, repo layout.
3. This file.

## Files you will work with

| File | Role |
|---|---|
| `maci_evince_sim.py` | Phase 1 single-file sim — the Modal payload (do not modify) |
| `modal_app.py` | Phase 1 Modal entrypoints: `train_and_evaluate`, `sweep` |
| `train_chat_lm_modal.py` | Phase 2 Modal entrypoints: `prepare_data`, `train`, `chat`, `finetune_on_chats` |
| `src/maci_sim/` | Package split of the sim (used by tests and `scripts/sweep.py`) |
| `scripts/sweep.py` | Local (CPU) M3 sweep — only if you choose not to use Modal for the sweep |
| `tests/` | M2 test suite — must be green before any Modal spend |
| `results/` | Commit result artifacts here (sample results, sweep table, chat transcripts) |

## Step 0 — Environment setup

```bash
# uv (project manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/kent-ai-dev/kent_agi.git && cd kent_agi
uv sync --all-groups          # torch, numpy, pytest, ruff, modal

# Modal auth — the VPS is headless, so do NOT use `modal setup` (it opens a
# browser). Ask the operator for a token from https://modal.com/settings/tokens
# and set it directly:
uv run modal token set --token-id <TOKEN_ID> --token-secret <TOKEN_SECRET>
# (or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
```

**Gate:** `uv run pytest -q` must be green (11 tests, <120 s) before you spend
anything on Modal.

## Step 1 — Phase 1 baseline on Modal (M4, ~2 min of T4, effectively free)

```bash
uv run modal run modal_app.py::train_and_evaluate            # seed 7 defaults
uv run modal run modal_app.py::train_and_evaluate --seed 13  # second seed
```

Acceptance (PLAN.md §1): the returned metric dict must show all three thesis
checks holding — debate consensus > best single agent, moderated agreement
rate > fixed-κ, wrong-repeat rate decaying. Expect numbers within ±1–2pp of
the README table. Retrieve and commit the artifact:

```bash
uv run modal volume get maci-checkpoints results_seed7.json ./results/sample_results.json
```

## Step 2 — M3 sweep on Modal (9 parallel containers, a few cents)

```bash
uv run modal run modal_app.py::sweep
```

Acceptance: checks 1 and 3 PASS in ≥ 8/9 cells; check 2 in ≥ 7/9. Pull the
per-cell JSONs from the `maci-checkpoints` volume, build the markdown table
(reuse `write_table()` in `scripts/sweep.py`), and commit it as
`results/sweep_table.md`. Report whether the debate-over-single-agent gap
grows with overlap — confirm or refute, either way.

## Step 3 — Phase 2 pretrain (M6, ~2–4 h A10G, < $10)

```bash
uv run modal run train_chat_lm_modal.py::prepare_data   # ~10 min, one-time
uv run modal run train_chat_lm_modal.py::train          # supports --resume True
```

Acceptance: final train loss ≤ 1.9 and the periodic "Once upon a time"
samples are grammatical multi-sentence stories with consistent characters.
Checkpoint persists in the `maci-lm` volume (`vol.commit()` is already in the
code). If the run is interrupted, re-run with `--resume True`.

## Step 4 — Chat REPL (M7) and continual learning (M8)

```bash
uv run modal run train_chat_lm_modal.py::chat                # interactive REPL
uv run modal run train_chat_lm_modal.py::finetune_on_chats   # after ≥2k chat tokens
```

- M7 acceptance: multi-turn session, on-topic replies, < 5 s/turn warm. Commit
  a session transcript to `results/chat_transcript.md`.
- M8 acceptance: seed ≥ 2k tokens of chats on a distinctive topic, capture
  unconditional samples **before and after** `finetune_on_chats`, and commit
  the comparison to `results/learning_before_after.md`. The 25% replay mixing
  is mandatory (guardrail — do not remove it).
- Optional nightly learning loop: deploy with `modal.Cron("0 3 * * *")` as
  described in PLAN.md M8.

## Step 5 — Commit artifacts

Commit only `results/*` files plus any documentation updates — never
checkpoints or the packed token binary (they live in Modal volumes;
`.gitignore` already excludes local copies). Push to `main`.

## Hard rules

- The VPS never trains locally; everything heavy runs on Modal.
- Do not modify `maci_evince_sim.py` or the §5 guardrails in PLAN.md.
- Keep torch pinned in the Modal images; do not let `pip_install` float.
- If an acceptance criterion fails, report the actual numbers honestly and
  stop — do not tune until it passes without flagging the change.
