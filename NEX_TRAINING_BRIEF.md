# Nex-N2 Inspired Training Plan for kent_agi

## Current State
- 77M param GPT-2 (8L, 8H, 512D, 512 seq)
- Trained on TinyStories to coherence (step 1000, loss ~2.4)
- Checkpoint tested: produces coherent narrative English
- Modal GPU training pipeline functional (T4, ~$2/run)

## Nex-N2 Insights to Apply
1. **Synthetic agentic environments** → train on interaction trajectories, not static text
2. **Adaptive reasoning depth** → special `<think>` tokens, learn when to reason deeply
3. **Unified tool-use format** → consistent schema: observation → thought → action
4. **Post-train on code + terminal trajectories** → structured API/terminal calling
5. **Fast inference loop** → optimize for tool-use latency, not just perplexity

## Proposed Training Phases

### Phase A: Math Reasoning Foundation
- **Dataset:** GSM8K + MATH (HuggingFace `openai/gsm8k`, `hendrycks/competition_math`)
- **Format:** question + chain-of-thought + answer
- **Objective:** Learn step-by-step reasoning format
- **Duration:** 5k–10k steps
- **Why:** Math is the purest reasoning task; teaches structured thought chains

### Phase B: Code Generation
- **Dataset:** HumanEval + MBPP (HuggingFace `openai_humaneval`, `mbpp`)
- **Format:** prompt + solution with comments
- **Objective:** Learn code syntax, function definitions, docstrings
- **Duration:** 5k–10k steps
- **Why:** Code is deterministic; rewards precise reasoning

### Phase C: Agentic Tool Use (Nex Innovation)
- **Dataset:** Synthetic trajectories + ToolBench (HuggingFace `ToolBench`)
- **Format:** 
  ```
  <observation>terminal output</observation>
  <thought>I need to check disk space</thought>
  <action>df -h</action>
  ```
- **Objective:** Learn the observe-think-act loop
- **Duration:** 5k–10k steps
- **Why:** This is the Nex-N2 core insight — unified reasoning+action format

### Phase D: Adaptive Reasoning Gate
- **Dataset:** Mix of simple (direct answer) and hard (requires CoT) questions
- **Format:** 
  ```
  <gate>think</gate>  # or <gate>act</gate>
  <content>...</content>
  ```
- **Objective:** Learn to emit reasoning tokens only when needed
- **Duration:** 3k–5k steps
- **Why:** Saves inference budget; Nex-N2's "adaptive thinking"

## Data Pipeline
- Use HuggingFace `datasets` library
- Stream datasets (don't download full corpus)
- Tokenize with tiktoken (gpt2 encoding)
- Concatenate to 512-token blocks
- Mix phases: 40% math, 30% code, 20% agentic, 10% adaptive

## Training Config
- Base LR: 6e-4 (same as Phase 2 chat LM)
- Batch: 64 sequences × 512 tokens
- Warmup: 500 steps
- Max steps: 25k total across all phases
- Gradient clipping: 1.0
- Save every 1k steps

## Evaluation
- Math: exact match on GSM8K test
- Code: pass@1 on HumanEval
- Agentic: trajectory completion accuracy
- Adaptive: token efficiency (correct answers with fewer think tokens)

## Success Criteria
- GSM8K: >5% exact match (baseline for 77M param)
- HumanEval: >2% pass@1
- Agentic: >50% correct action selection
- Model emits `<think>` tokens only on hard questions
