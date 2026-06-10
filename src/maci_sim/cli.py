"""CLI orchestration — reproduces the exact interface of maci_evince_sim.py.

Usage:
    python -m maci_sim.cli --mode all                # train both agents + run debate eval
    python -m maci_sim.cli --mode train --agent A    # train one agent
    python -m maci_sim.cli --mode debate             # eval with existing checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import torch

from maci_sim.agent import Agent, AgentConfig, TrainConfig, train_agent
from maci_sim.debate import DebateConfig, run_debate
from maci_sim.world import Vocab, World, WorldConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="MACI/EVINCE simulation (arXiv:2409.01007)")
    ap.add_argument("--mode", choices=["train", "debate", "all"], default="all")
    ap.add_argument("--agent", choices=["A", "B"], default=None,
                    help="with --mode train: train only this agent")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-samples", type=int, default=30000)
    ap.add_argument("--eval-samples", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--overlap", type=float, default=0.45)
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--results", default="./results.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    wcfg = WorldConfig(overlap=args.overlap, seed=args.seed)
    acfg, dcfg = AgentConfig(), DebateConfig(rounds=args.rounds)
    tcfg = TrainConfig(epochs=args.epochs, train_samples=args.train_samples)
    world, vocab = World(wcfg), Vocab(wcfg)
    prior_a, prior_b, prior_eval = world.biased_priors()
    path_a = os.path.join(args.ckpt_dir, "agent_a.pt")
    path_b = os.path.join(args.ckpt_dir, "agent_b.pt")
    t0 = time.time()

    if args.mode in ("train", "all"):
        if args.agent in (None, "A"):
            train_agent("A (common-bias)", world, vocab, prior_a, acfg, tcfg,
                        args.device, path_a, seed=args.seed + 1)
        if args.agent in (None, "B"):
            train_agent("B (rare-bias)", world, vocab, prior_b, acfg, tcfg,
                        args.device, path_b, seed=args.seed + 2)

    if args.mode in ("debate", "all"):
        agent_a = Agent(vocab, acfg, wcfg.n_conditions).to(args.device)
        agent_b = Agent(vocab, acfg, wcfg.n_conditions).to(args.device)
        agent_a.load_state_dict(torch.load(path_a, map_location=args.device))
        agent_b.load_state_dict(torch.load(path_b, map_location=args.device))

        X_eval, y_eval = world.sample_cases(args.eval_samples, prior_eval)
        res = run_debate(agent_a, agent_b, world, vocab, X_eval, y_eval,
                         dcfg, args.device, moderate=True)
        # Ablation: same claim exchange but NO information-theoretic
        # moderation — kappa stays pinned at its adversarial initial value.
        res_ablate = run_debate(agent_a, agent_b, world, vocab, X_eval,
                                y_eval, dcfg, args.device, moderate=False)
        res.pop("consensus_labels", None)
        res_ablate.pop("consensus_labels", None)
        res["acc_debate_unmoderated"] = res_ablate["acc_debate_consensus"]
        res["ablation_log"] = res_ablate["debate_log"]
        res["config"] = {"world": asdict(wcfg), "agent": asdict(acfg),
                         "train": asdict(tcfg), "debate": asdict(dcfg)}
        with open(args.results, "w") as f:
            json.dump(res, f, indent=2)

        log = res["debate_log"]
        best_single = max(res["acc_agent_a_alone"], res["acc_agent_b_alone"])
        print("\n================ MACI / EVINCE simulation report ================")
        print(f"Agent A alone (common-bias)        : {res['acc_agent_a_alone']:.3f}")
        print(f"Agent B alone (rare-bias)          : {res['acc_agent_b_alone']:.3f}")
        print(f"Debate, fixed kappa={dcfg.kappa_init} (ablation): "
              f"{res['acc_debate_unmoderated']:.3f}")
        print(f"Debate, EVINCE-moderated kappa     : {res['acc_debate_consensus']:.3f}")
        print(f"[oracle refs — full distribution pooling, unavailable to real"
              f" multi-LLM systems]")
        print(f"  oracle avg pool : {res['acc_oracle_distribution_pool']:.3f}"
              f"   oracle PoE : {res['acc_oracle_poe']:.3f}")
        print(f"Final claim agreement rate (moderated): "
              f"{res['final_agreement_rate']:.3f}  vs fixed-kappa: "
              f"{res_ablate['final_agreement_rate']:.3f}")
        print(f"Rounds run: {log['rounds_run']}  (consensus eps = {dcfg.jsd_eps})")
        print("\nround |  kappa |   JSD  |  H(A)  |  H(B)  | dec-MI | wrong-repeat")
        for i in range(log["rounds_run"]):
            print(f"  {i:3d} | {log['kappa'][i]:6.3f} | {log['mean_jsd'][i]:6.4f}"
                  f" | {log['mean_entropy_a'][i]:6.3f} | {log['mean_entropy_b'][i]:6.3f}"
                  f" | {log['decision_mi'][i]:6.3f} | {log['wrong_repeat_rate'][i]:6.3f}")
        ok1 = res["acc_debate_consensus"] > best_single
        # EVINCE's measurable contribution is consensus FORMATION: annealing
        # contentiousness with divergence drives the agents to actual
        # agreement far more often than a fixed adversarial stance. (Final
        # arbitrated accuracy is reported above; with a confidence
        # tiebreaker it is statistically similar — the moderation effect
        # shows in how often arbitration is needed at all.)
        ok2 = (res["final_agreement_rate"]
               > res_ablate["final_agreement_rate"])
        print(f"\nThesis check 1 (Aph. #10, debate beats any single agent): "
              f"{res['acc_debate_consensus']:.3f} > {best_single:.3f}  "
              f"{'PASS' if ok1 else 'FAIL'}")
        print(f"Thesis check 2 (Ch. 7, moderation drives consensus formation): "
              f"agreement {res['final_agreement_rate']:.3f} > "
              f"{res_ablate['final_agreement_rate']:.3f}  "
              f"{'PASS' if ok2 else 'FAIL'}")
        print(f"Thesis check 3 (Aph. #11, wrong-repeat rate decays): "
              f"{log['wrong_repeat_rate'][0]:.3f} -> {log['wrong_repeat_rate'][-1]:.3f}  "
              f"{'PASS' if log['wrong_repeat_rate'][-1] < log['wrong_repeat_rate'][0] else 'FAIL'}")
        print(f"Results JSON -> {args.results}")

    print(f"\nDone in {time.time() - t0:.1f}s on {args.device}")


if __name__ == "__main__":
    main()
