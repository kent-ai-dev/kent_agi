"""
Nex-N2 Inspired Multi-Phase Training for kent_agi
Loads HuggingFace datasets, mixes them, trains on Modal T4.
"""
import os, json, time, math, random
from pathlib import Path

import modal
from modal import App, Image, Volume, gpu

# ----------------------------------------------------------------------------
# Modal config
# ----------------------------------------------------------------------------
app = App("kent-agi-nex-training")
vol = Volume.from_name("maci-nex", create_if_missing=True)
vol_base = Volume.from_name("maci-lm", create_if_missing=True)
DATA = "/data"
DATA_BASE = "/data_base"
CKPT_PATH = f"{DATA}/nex_checkpoint.pt"
CKPT_BASE_PATH = f"{DATA_BASE}/chat_lm.pt"
RESULTS_PATH = f"{DATA}/training_results.jsonl"

image = (
    Image.debian_slim()
    .pip_install("torch", "numpy", "tqdm", "datasets", "tiktoken", "transformers")
)

# ----------------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------------
def load_gsm8k(split="train", max_samples=None):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        answer = ex["answer"].split("####")[-1].strip()
        reasoning = ex["answer"].split("####")[0].strip()
        text = f"Q: {ex['question']}\nA: Let's think step by step. {reasoning} The answer is {answer}.\n"
        texts.append(text)
    return texts

def load_mbpp(split="train", max_samples=None):
    from datasets import load_dataset
    ds = load_dataset("mbpp", split=split, streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        text = f"# {ex['text']}\n{ex['code']}\n\n"
        texts.append(text)
    return texts

def load_synthetic_agentic(n=15000):
    """Generate synthetic terminal trajectories with diverse tool-use."""
    commands = [
        # File operations
        ("ls -la", "drwxr-xr-x 5 user user 4096 Jun 11 .\ndrwxr-xr-x 3 user user 4096 ..\n-rw-r--r-- 1 user user  220 README.md\n"),
        ("cat README.md", "# Project\nThis is a test project for trading research.\n\n## Setup\nRun `pip install -r requirements.txt`\n"),
        ("touch new_file.txt", ""),
        ("mkdir data", ""),
        ("rm old_file.txt", ""),
        ("mv file.txt archive/", ""),
        ("cp config.yaml config.yaml.bak", ""),
        # Git operations
        ("git status", "On branch main\nYour branch is up to date with 'origin/main'.\n\nnothing to commit, working tree clean\n"),
        ("git log --oneline -5", "a1b2c3d feat: add backtest engine\n9f8e7d6 fix: stop loss calculation\n4a5b6c7 docs: update README\n"),
        ("git add .", ""),
        ("git commit -m 'feat: new strategy'", "[main 1234567] feat: new strategy\n 2 files changed, 50 insertions(+), 10 deletions(-)\n"),
        ("git push origin main", "Enumerating objects: 15, done.\nWriting objects: 100% (15/15), 2.1 KiB | 2.1 MiB/s, done.\n"),
        # Python REPL
        ("python3 -c 'print(2+2)'", "4\n"),
        ("python3 -c 'import numpy; print(numpy.random.randn(3))'", "[ 0.123 -0.456  0.789]\n"),
        ("python3 -c 'import pandas; df = pandas.DataFrame({\"A\": [1,2,3]}); print(df.mean())'", "A    2.0\ndtype: float64\n"),
        ("python3 script.py", "Loading data...\nComputing signals...\nBacktest complete. Sharpe: 1.23\n"),
        # System info
        ("df -h", "Filesystem Size Used Avail Use%\n/dev/sda1 100G 20G 80G 20%\n/dev/sdb1 500G 300G 200G 60%\n"),
        ("free -h", "              total        used        free\nMem:          16Gi       8.2Gi       7.8Gi\n"),
        ("uptime", " 14:32:01 up 5 days,  2:15,  1 user,  load average: 0.45, 0.38, 0.30\n"),
        ("ps aux | grep python", "user    1234  0.5  2.1  23456  8901 ?  Ss 14:00  0:10 python3 main.py\n"),
        # Package management
        ("pip install numpy pandas", "Successfully installed numpy-1.24.0 pandas-2.0.0\n"),
        ("pip list | grep torch", "torch                  2.4.0\ntorchvision            0.19.0\n"),
        ("python3 -m pip freeze > requirements.txt", ""),
        # Network/API calls
        ("curl -s https://api.github.com/users/octocat", '{"login":"octocat","id":583231","type":"User"}\n'),
        ("curl -s https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", '{"bitcoin":{"usd":67500.00}}\n'),
        ("wget https://example.com/data.csv", "Resolving example.com... 93.184.216.34\nHTTP request sent, awaiting response... 200 OK\nSaving to: data.csv\n"),
        # Docker/container
        ("docker ps", "CONTAINER ID  IMAGE          COMMAND       CREATED        STATUS\nabc123def456  trading:latest \"/bin/bash\"   2 hours ago    Up 2 hours\n"),
        ("docker build -t trading .", "Sending build context to Docker daemon  12.3MB\nSuccessfully built abc123def456\n"),
        # Trading-specific
        ("python3 -c 'import yfinance as yf; data = yf.download(\"AAPL\", period=\"5d\"); print(data.tail())'", "                 Open   High    Low  Close    Volume\nDate\n2026-06-07  210.50 212.30 209.80 211.40  45678900\n2026-06-08  211.40 213.10 210.90 212.80  52345600\n"),
        ("python3 backtest.py --config config.yaml", "Loading config...\nDownloading data for 55 symbols...\nRunning backtest...\nTotal return: -3.70%\nSharpe: -0.62\n"),
        # Modal/cloud
        ("modal app list", "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n┃ App ID                    ┃ Description  ┃ State   ┃ Tasks ┃\n┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n│ ap-abc123                 │ training     │ running │ 1     │\n"),
        ("modal volume ls maci-lm", "chat_lm.pt\nchat_logs.jsonl\n"),
        # Hermes/AI assistant specific
        ("hermes status", "Hermes Agent v2.1.172\nProfile: default\nBridges: 7 active\n"),
        ("hermes bridge list", " gemini-snap-flash  : port 8131\n trading-desk       : port 8132\n wheel              : port 8133\n ofa                : port 8140\n ken_alpaca         : port 8141\n kalshi-trader      : port 8142\n kent-agi           : port 8143\n"),
    ]
    
    texts = []
    for _ in range(n):
        traj_len = random.randint(3, 10)
        traj = []
        for _ in range(traj_len):
            cmd, out = random.choice(commands)
            traj.append(f"<obs>{cmd}</obs>")
            traj.append(f" thinkingThe user wants me to execute {cmd.split()[0]}. I'll run it and report the output. thinking")
            traj.append(f"<act>{out.strip()}</act>")
        texts.append("\n".join(traj) + "\n")
    return texts

def load_adaptive_gate(n=5000):
    """Simple vs hard questions with gate tokens."""
    simple = [
        ("What is 2+2?", "4", "skip"),
        ("What is 10*5?", "50", "skip"),
        ("What is 100/4?", "25", "skip"),
    ]
    hard = [
        ("A train travels 60 mph for 2 hours and 40 mph for 3 hours. What is the average speed?", 
         "The total distance is 60*2 + 40*3 = 120 + 120 = 240 miles. Total time is 5 hours. Average speed = 240/5 = 48 mph.", "think"),
        ("If 3 workers can build 2 houses in 4 days, how many days for 5 workers to build 10 houses?",
         "3 workers build 2 houses in 4 days, so 1 worker builds 2/3 house in 4 days. 5 workers build 10/3 houses in 4 days. To build 10 houses: 10 / (10/3) * 4 = 12 days.", "think"),
    ]
    texts = []
    for _ in range(n):
        q, a, gate = random.choice(simple + hard)
        if gate == "skip":
            text = f"Q: {q}\n<gate>skip</gate>\nAnswer: {a}\n\n"
        else:
            text = f"Q: {q}\n<gate>think</gate>\n{a}\nAnswer: {a.split('=')[-1].strip() if '=' in a else a}\n\n"
        texts.append(text)
    return texts

# ----------------------------------------------------------------------------
# Tokenization and batching
# ----------------------------------------------------------------------------
def tokenize_texts(texts, enc, block_size=512):
    import numpy as np
    all_ids = []
    for text in texts:
        ids = enc.encode_ordinary(text)
        all_ids.extend(ids)
    
    # Split into blocks
    n_blocks = len(all_ids) // block_size
    all_ids = all_ids[:n_blocks * block_size]
    arr = np.array(all_ids, dtype=np.uint16)
    return arr.reshape(n_blocks, block_size)

def create_mixed_dataset(enc, block_size=512, samples_per_phase=5000):
    """Create mixed dataset with ratios from TRAINING_PLAN.md."""
    print("Loading datasets...")
    
    math_texts = load_gsm8k("train", samples_per_phase)
    code_texts = load_mbpp("train", samples_per_phase)
    agentic_texts = load_synthetic_agentic(samples_per_phase)
    adaptive_texts = load_adaptive_gate(samples_per_phase)
    
    # Mix ratios: 25% math, 25% code, 35% agentic, 15% adaptive
    # Agentic/tool-use emphasized per user request
    all_texts = []
    ratios = [0.25, 0.25, 0.35, 0.15]
    phases = [math_texts, code_texts, agentic_texts, adaptive_texts]
    
    # Interleave
    max_len = max(len(p) for p in phases)
    for i in range(max_len):
        for phase_texts, ratio in zip(phases, ratios):
            n_take = max(1, int(ratio * 10))  # approximate per-batch mixing
            start = (i * n_take) % len(phase_texts)
            end = min(start + n_take, len(phase_texts))
            all_texts.extend(phase_texts[start:end])
    
    random.shuffle(all_texts)
    print(f"Total texts: {len(all_texts)}")
    
    arr = tokenize_texts(all_texts, enc, block_size)
    print(f"Token blocks: {arr.shape[0]} x {arr.shape[1]}")
    return arr

# ----------------------------------------------------------------------------
# Model definition (same as train_chat_lm_modal.py)
# ----------------------------------------------------------------------------
def build_model_classes():
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    
    class GPTConfig:
        def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
            self.vocab_size = vocab_size; self.block_size = block_size
            self.n_layer = n_layer; self.n_head = n_head
            self.n_embd = n_embd; self.dropout = dropout
    
    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.n_head = config.n_head; self.n_embd = config.n_embd
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))
        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)
    
    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        def forward(self, x):
            return self.c_proj(F.gelu(self.c_fc(x)))
    
    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)
        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
    
    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
            self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.apply(self._init_weights)
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        def forward(self, idx, targets=None):
            device = idx.device
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
            for block in self.h:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                if eos_id is not None and idx_next.item() == eos_id:
                    break
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    
    return GPTConfig, GPT

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@app.function(image=image, gpu=gpu.T4(), timeout=3600, volumes={DATA: vol, DATA_BASE: vol_base})
def train_nex_phases(total_steps=18000, batch_size=64, block_size=512, lr=3e-4, 
                     warmup=500, save_every=1000, eval_every=1000):
    import torch
    import tiktoken
    import numpy as np
    from tqdm import tqdm
    
    print("Building model...")
    GPTConfig, GPT = build_model_classes()
    cfg = GPTConfig(vocab_size=50257, block_size=512, n_layer=8, n_head=8, 
                    n_embd=512, dropout=0.0)
    model = GPT(cfg).to("cuda")
    
    # Load base checkpoint from maci-lm (separate volume, read-only)
    if os.path.exists(CKPT_BASE_PATH):
        ck = torch.load(CKPT_BASE_PATH, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Loaded base checkpoint from {CKPT_BASE_PATH}")
    else:
        print("WARNING: No base checkpoint found, training from scratch!")
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")
    
    print("Loading datasets...")
    enc = tiktoken.get_encoding("gpt2")
    data_arr = create_mixed_dataset(enc, block_size=block_size, samples_per_phase=5000)
    
    # Convert to torch tensor
    data_tensor = torch.from_numpy(data_arr).long().to("cuda")
    n_blocks = data_tensor.shape[0]
    
    print(f"Training on {n_blocks} blocks for {total_steps} steps")
    
    model.train()
    losses = []
    
    for step in range(total_steps):
        # Sample batch
        ix = torch.randint(0, n_blocks, (batch_size,))
        x = data_tensor[ix]
        y = torch.cat([x[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long, device="cuda")], dim=1)
        
        # LR schedule: warmup + cosine decay
        if step < warmup:
            lri = lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / (total_steps - warmup)
            lri = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in opt.param_groups:
            param_group["lr"] = lri
        
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            print(f"step {step}/{total_steps} loss={avg_loss:.3f} lr={lri:.2e}")
        
        if step > 0 and step % save_every == 0:
            ck = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": {"vocab_size": 50257, "block_size": 512, "n_layer": 8, 
                        "n_head": 8, "n_embd": 512, "dropout": 0.0}
            }
            torch.save(ck, f"{DATA}/nex_checkpoint_{step}.pt")
            vol.commit()
            print(f"Saved checkpoint at step {step}")
        
        if step > 0 and step % eval_every == 0:
            # Quick eval: generate a math answer
            model.eval()
            test_prompt = "Q: What is 15 + 27?\nA: Let's think step by step."
            idx = torch.tensor([enc.encode_ordinary(test_prompt)], device="cuda")
            out = model.generate(idx, max_new_tokens=30, temperature=0.8)
            generated = enc.decode(out[0].tolist())
            print(f"  Eval: {generated[:100]}")
            model.train()
    
    # Final save
    ck = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": total_steps,
        "cfg": {"vocab_size": 50257, "block_size": 512, "n_layer": 8, 
                "n_head": 8, "n_embd": 512, "dropout": 0.0}
    }
    torch.save(ck, f"{DATA}/nex_final.pt")
    vol.commit()
    print(f"Training complete. Final checkpoint saved.")
    return {"steps": total_steps, "final_loss": sum(losses[-100:]) / 100}

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Starting Nex-N2 inspired training...")
    result = train_nex_phases.remote()
    print(f"Training complete: {result}")

if __name__ == "__main__":
    main()
