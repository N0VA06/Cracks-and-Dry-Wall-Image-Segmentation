"""
scripts/run_multiseed.py
========================
Trains each model variant across N random seeds and aggregates:
  • mean ± std of mIoU and Dice per prompt
  • consistency score across seeds

Usage
-----
  python scripts/run_multiseed.py --seeds 42 123 999 --model pretrained
  python scripts/run_multiseed.py --seeds 42 123 999 --model all
"""

import sys
import os
import json
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_DIR, DATASET_ROOTS, PROMPT_TEMPLATES

MODELS = ["pretrained", "custom", "active", "sam"]


def run_seed(model_key: str, seed: int, dataset: str = "all") -> int:
    """Run training + evaluation for one seed. Returns exit code."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Patch seed in config temporarily via env var (read in main.py)
    env["OVERRIDE_SEED"] = str(seed)

    train_cmd = [
        sys.executable, os.path.join(project_root, "main.py"),
        "--mode", "train",
        "--model", model_key,
        "--dataset", dataset,
    ]
    eval_cmd = [
        sys.executable, os.path.join(project_root, "evaluate.py"),
        "--model", model_key,
        "--seed", str(seed),
    ]

    print(f"\n{'='*56}")
    print(f"  Seed {seed} | Model: {model_key}")
    print(f"{'='*56}")

    ret = subprocess.run(train_cmd, cwd=project_root, env=env).returncode
    if ret != 0:
        print(f"  Training failed (exit {ret}) — skipping eval")
        return ret

    ret = subprocess.run(eval_cmd, cwd=project_root, env=env).returncode
    return ret


def aggregate_results(model_key: str, seeds: list[int]) -> dict:
    """
    Load per-seed eval_results.json files and compute mean ± std.
    Assumes each seed's results were saved to LOG_DIR/eval_results.json
    and then renamed to eval_results_seed{N}.json by this script.
    """
    seed_data = []
    for seed in seeds:
        path = os.path.join(LOG_DIR, f"eval_results_seed{seed}.json")
        if not os.path.isfile(path):
            print(f"  Missing: {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        if model_key in d.get("results", {}):
            seed_data.append(d["results"][model_key])

    if not seed_data:
        return {}

    # Aggregate across seeds
    datasets = list(seed_data[0].keys())
    agg = {}
    for ds in datasets:
        ious  = [s[ds]["mean_iou"]  for s in seed_data if ds in s]
        dices = [s[ds]["mean_dice"] for s in seed_data if ds in s]
        stds  = [s[ds]["std_iou"]   for s in seed_data if ds in s]
        agg[ds] = {
            "prompt"         : seed_data[0][ds]["prompt"],
            "n_seeds"        : len(ious),
            "mean_iou"       : round(float(np.mean(ious)),  4),
            "std_iou_seeds"  : round(float(np.std(ious)),   4),
            "mean_dice"      : round(float(np.mean(dices)), 4),
            "std_dice_seeds" : round(float(np.std(dices)),  4),
            "mean_consistency": round(float(np.mean(stds)), 4),  # avg within-dataset IoU std
        }
    return agg


def print_multiseed_table(all_agg: dict):
    print("\n" + "═" * 90)
    print("  MULTI-SEED RESULTS  (mean ± std across seeds)")
    print("═" * 90)
    print(f"{'Model':12s} │ {'Dataset':8s} │ {'Prompt':22s} │ "
          f"{'mIoU':11s} │ {'Dice':11s} │ {'Consist↓':10s}")
    print("─" * 90)
    for model_key, ds_agg in all_agg.items():
        for ds, r in ds_agg.items():
            miou  = f"{r['mean_iou']:.4f}±{r['std_iou_seeds']:.4f}"
            mdice = f"{r['mean_dice']:.4f}±{r['std_dice_seeds']:.4f}"
            cons  = f"{r['mean_consistency']:.4f}"
            print(f"{model_key:12s} │ {ds:8s} │ {r['prompt']:22s} │ "
                  f"{miou:11s} │ {mdice:11s} │ {cons:10s}")
        print("─" * 90)
    print("═" * 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",   nargs="+", type=int, default=[42, 123, 999],
                   help="Seeds to train/evaluate over")
    p.add_argument("--model",   default="all",
                   choices=["all"] + MODELS,
                   help="Model(s) to run")
    p.add_argument("--dataset", default="all",
                   help="Dataset for training (default: all)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; only aggregate existing results")
    args = p.parse_args()

    model_keys = MODELS if args.model == "all" else [args.model]

    if not args.eval_only:
        for seed in args.seeds:
            for model_key in model_keys:
                run_seed(model_key, seed, dataset=args.dataset)
                # Rename result file to seed-specific name
                src = os.path.join(LOG_DIR, "eval_results.json")
                dst = os.path.join(LOG_DIR, f"eval_results_seed{seed}.json")
                if os.path.isfile(src):
                    os.replace(src, dst)

    # Aggregate
    all_agg = {}
    for model_key in model_keys:
        agg = aggregate_results(model_key, args.seeds)
        if agg:
            all_agg[model_key] = agg

    print_multiseed_table(all_agg)

    # Save aggregated results
    out = os.path.join(LOG_DIR, "multiseed_results.json")
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "results": all_agg}, f, indent=2)
    print(f"\nAggregated results saved → {out}")


if __name__ == "__main__":
    main()
