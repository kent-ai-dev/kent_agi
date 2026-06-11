# kent_agi Nex-N2 Inspired Training Plan
## Version: 1.0 | Target: 77M param GPT-2 | Modal T4 GPU

---

## Architecture Baseline
- **Model:** 8L/8H/512D, 512 seq, 77M params
- **Tokenizer:** tiktoken (gpt2)
- **Base checkpoint:** chat_lm.pt (step 1000, TinyStories pretrained)
- **Compute:** Modal T4 (~$0.60/hr), 2–4 hrs per phase

---

## Phase A: Math Reasoning (GSM8K + MATH)

**Goal:** Learn chain-of-thought reasoning format.

**Datasets:**
- Primary: `openai/gsm8k` (8.5K train, 1K test)
- Secondary: `hendrycks/competition_math` (7.5K train)
- Format: `Q: {question}\nA: Let's think step by step. {reasoning} The answer is {answer}.`

**Hyperparameters:**
- Steps: 5,000
- LR: 6e-4 × 0.5 = 3e-4 (lower LR for fine-tuning)
- Batch: 64 × 512 tokens
- Warmup: 500 steps
- Schedule: cosine decay to 1e-5

**Expected outcome:** 3–5% exact match on GSM8K test (reasonable for 77M param)

---

## Phase B: Code (HumanEval + MBPP)

**Goal:** Learn Python syntax, function definitions, docstrings.

**Datasets:**
- Primary: `openai_humaneval` (164 problems → expand with `nuprl/MultiPL-E`)
- Secondary: `mbpp` (974 train, 500 test)
- Tertiary: `codeparrot/github-code` (sample 50K Python functions)
- Format: `# {prompt}\n{solution}\n# Test:\n{assertions}`

**Hyperparameters:**
- Steps: 5,000
- LR: 3e-4
- Same batch/warmup as Phase A

**Expected outcome:** 2–4% pass@1 on HumanEval

---

## Phase C: Agentic Tool Use (Synthetic + ToolBench)

**Goal:** Learn observe → think → action loop (Nex-N2 core insight).

**Datasets:**
- Primary: Synthetic terminal trajectories (generated via Python scripts)
- Secondary: `ToolBench` (HuggingFace `yizhongw/toolbench`)
- Format:
  ```
  <obs>ls -la</obs>
  <think>The user wants to see files. I'll list them.</think>
  <act>drwxr-xr-x 5 user user 4096 Jun 11 .</act>
  ```

**Synthetic data generation:**
- 10K trajectories of: file operations, git commands, python REPL, curl requests
- Each trajectory: 5–10 turns, random but valid commands
- Reward: correct command syntax + plausible output

**Hyperparameters:**
- Steps: 5,000
- LR: 3e-4

**Expected outcome:** >50% correct action selection on held-out trajectories

---

## Phase D: Adaptive Reasoning Gate

**Goal:** Learn to emit reasoning tokens only when needed (Nex-N2 "adaptive thinking").

**Dataset:**
- Mixed simple vs hard questions from GSM8K
- Simple (grade 1–2): direct answer, no CoT needed
- Hard (grade 6–8): requires step-by-step reasoning
- Format:
  ```
  <gate>skip</gate>  # for simple questions
  Answer: 42
  
  <gate>think</gate>  # for hard questions
  Let's break this down...
  Answer: 42
  ```

**Hyperparameters:**
- Steps: 3,000
- LR: 2e-4

**Expected outcome:** Model uses <think> on hard questions, skips on easy

---

## Data Mixing Ratio (All Phases Combined)

| Phase | % of batch | Rationale |
|---|---|---|
| Math | 35% | Reasoning foundation |
| Code | 30% | Structured output, syntax |
| Agentic | 25% | Tool-use loop |
| Adaptive | 10% | Meta-control |

**Total:** ~18K steps across all phases
**Total compute:** ~$5–8 on Modal T4
**Total time:** ~6–10 hours

---

## Evaluation Pipeline

After each 1K-step checkpoint:
1. **GSM8K:** Run 100 random test examples, exact match
2. **HumanEval:** Run pass@1 on 50 random problems
3. **Agentic:** Run 50 synthetic trajectories, action accuracy
4. **Adaptive:** Check token count on easy vs hard questions

Results logged to `training_results.jsonl`.

---

## Success Criteria

| Metric | Target | Acceptable |
|---|---|---|
| GSM8K exact match | >5% | >3% |
| HumanEval pass@1 | >2% | >1% |
| Agentic action accuracy | >50% | >40% |
| Think-token efficiency | 2× fewer on easy | 1.5× |

---

## Next After This Plan

If targets met:
1. Scale to 150M–300M param model (same architecture, wider/deeper)
2. Add more agentic environments (browser, file system, API calls)
3. Train with RLHF on tool-use trajectories
4. Deploy as Hermes sub-agent for coding tasks

If targets not met:
1. Increase model size to 150M
2. Add more synthetic data
3. Try instruction tuning before domain-specific training
