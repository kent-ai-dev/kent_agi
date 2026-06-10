"""M3 sweep: overlap x seed grid for the three thesis checks (PLAN.md).

Grid: overlap in {0.3, 0.45, 0.6} x seed in {7, 13, 21}. Writes one JSON per
cell under results/sweep/ plus an aggregate markdown table at
results/sweep_table.md, and reports whether the debate-over-single-agent gap
grows with overlap (confirm or refute — either way, report it).

Usage:
    uv run python scripts/sweep.py
    uv run python scripts/sweep.py --overlaps 0.3 0.45 --seeds 7
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from maci_sim.agent import AgentConfig, TrainConfig, train_agent
from maci_sim.debate import DebateConfig, run_debate
from maci_sim.world import Vocab, World, WorldConfig

OVERLAPS = [0.3, 0.45, 0.6]
SEEDS = [7, 13, 21]


def run_cell(seed: int, overlap: float, epochs: int, train_samples: int,
             eval_samples: int, device: str, ckpt_root: str) -> dict:
    wcfg = WorldConfig(overlap=overlap, seed=seed)
    acfg, dcfg = AgentConfig(), DebateConfig()
    tcfg = TrainConfig(epochs=epochs, train_samples=train_samples)
    world, vocab = World(wcfg), Vocab(wcfg)
    prior_a, prior_b, prior_eval = world.biased_priors()

    ckpt_dir = os.path.join(ckpt_root, f"overlap{overlap}_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    agent_a = train_agent("A (common-bias)", world, vocab, prior_a, acfg, tcfg,
                          device, os.path.join(ckpt_dir, "agent_a.pt"),
                          seed=seed + 1)
    agent_b = train_agent("B (rare-bias)", world, vocab, prior_b, acfg, tcfg,
                          device, os.path.join(ckpt_dir, "agent_b.pt"),
                          seed=seed + 2)

    X_eval, y_eval = world.sample_cases(eval_samples, prior_eval)
    res = run_debate(agent_a, agent_b, world, vocab, X_eval, y_eval, dcfg,
                     device, moderate=True)
    res_ablate = run_debate(agent_a, agent_b, world, vocab, X_eval, y_eval,
                            dcfg, device, moderate=False)
    res.pop("consensus_labels", None)
    res_ablate.pop("consensus_labels", None)

    best_single = max(res["acc_agent_a_alone"], res["acc_agent_b_alone"])
    wrr = res["debate_log"]["wrong_repeat_rate"]
    cell = {
        "seed": seed,
        "overlap": overlap,
        "acc_agent_a_alone": res["acc_agent_a_alone"],
        "acc_agent_b_alone": res["acc_agent_b_alone"],
        "acc_debate_consensus": res["acc_debate_consensus"],
        "acc_debate_unmoderated": res_ablate["acc_debate_consensus"],
        "acc_oracle_distribution_pool": res["acc_oracle_distribution_pool"],
        "debate_gap": res["acc_debate_consensus"] - best_single,
        "final_agreement_rate": res["final_agreement_rate"],
        "final_agreement_rate_unmoderated": res_ablate["final_agreement_rate"],
        "wrong_repeat_first": wrr[0],
        "wrong_repeat_last": wrr[-1],
        "check1_debate_beats_single": res["acc_debate_consensus"] > best_single,
        "check2_moderation_drives_agreement":
            res["final_agreement_rate"] > res_ablate["final_agreement_rate"],
        "check3_wrong_repeat_decays": wrr[-1] < wrr[0],
        "debate_log": res["debate_log"],
        "ablation_log": res_ablate["debate_log"],
    }
    return cell


def write_table(cells: list[dict], path: str) -> str:
    lines = [
        "# M3 sweep — thesis checks across overlap x seed",
        "",
        "Grid from PLAN.md M3. Check 1: debate consensus > best single agent"
        " (Aph. #10). Check 2: moderated final agreement rate > fixed-kappa"
        " (Ch. 7). Check 3: wrong-claim repeat rate decays (Aph. #11).",
        "",
        "| overlap | seed | acc A | acc B | debate | gap | agree (mod) |"
        " agree (fixed) | wrong-repeat | C1 | C2 | C3 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
        lines.append(
            f"| {c['overlap']} | {c['seed']} | {c['acc_agent_a_alone']:.3f} |"
            f" {c['acc_agent_b_alone']:.3f} | {c['acc_debate_consensus']:.3f} |"
            f" {c['debate_gap']:+.3f} | {c['final_agreement_rate']:.3f} |"
            f" {c['final_agreement_rate_unmoderated']:.3f} |"
            f" {c['wrong_repeat_first']:.2f} -> {c['wrong_repeat_last']:.2f} |"
            f" {mark(c['check1_debate_beats_single'])} |"
            f" {mark(c['check2_moderation_drives_agreement'])} |"
            f" {mark(c['check3_wrong_repeat_decays'])} |")

    n = len(cells)
    c1 = sum(c["check1_debate_beats_single"] for c in cells)
    c2 = sum(c["check2_moderation_drives_agreement"] for c in cells)
    c3 = sum(c["check3_wrong_repeat_decays"] for c in cells)
    lines += ["", f"**Totals:** check 1: {c1}/{n} · check 2: {c2}/{n} ·"
                  f" check 3: {c3}/{n}", ""]

    # Does the debate-over-single-agent gap grow with overlap (harder tasks)?
    overlaps = sorted({c["overlap"] for c in cells})
    if len(overlaps) > 1:
        lines.append("**Gap vs overlap** (mean debate gap per overlap level):")
        lines.append("")
        means = []
        for o in overlaps:
            gaps = [c["debate_gap"] for c in cells if c["overlap"] == o]
            means.append(sum(gaps) / len(gaps))
            lines.append(f"- overlap {o}: mean gap {means[-1]:+.3f}"
                         f" (n={len(gaps)})")
        trend = ("CONFIRMED — the collaboration gap grows with task overlap"
                 if all(b >= a for a, b in zip(means, means[1:]))
                 else "NOT MONOTONE — the gap does not strictly grow with"
                      " overlap on this grid")
        lines += ["", f"Expectation from PLAN.md M3: {trend}.", ""]

    table = "\n".join(lines)
    with open(path, "w") as f:
        f.write(table)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="MACI/EVINCE M3 sweep")
    ap.add_argument("--overlaps", type=float, nargs="+", default=OVERLAPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-samples", type=int, default=24000)
    ap.add_argument("--eval-samples", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="./results/sweep")
    ap.add_argument("--ckpt-root", default="./checkpoints/sweep")
    ap.add_argument("--table", default="./results/sweep_table.md")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cells = []
    t0 = time.time()
    for overlap in args.overlaps:
        for seed in args.seeds:
            print(f"\n===== cell overlap={overlap} seed={seed} =====")
            cell = run_cell(seed, overlap, args.epochs, args.train_samples,
                            args.eval_samples, args.device, args.ckpt_root)
            out = os.path.join(args.out_dir,
                               f"sweep_overlap{overlap}_seed{seed}.json")
            with open(out, "w") as f:
                json.dump(cell, f, indent=2)
            print(f"cell done -> {out}  (debate {cell['acc_debate_consensus']:.3f},"
                  f" gap {cell['debate_gap']:+.3f})")
            cells.append(cell)

    table = write_table(cells, args.table)
    print("\n" + table)
    print(f"\nSweep done in {(time.time() - t0) / 60:.1f} min"
          f" -> {args.table}")


if __name__ == "__main__":
    main()
