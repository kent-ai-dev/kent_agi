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
