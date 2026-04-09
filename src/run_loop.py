"""Main loop: run the full post-hoc boosting pipeline iteratively."""

import argparse
import json
import os

import yaml

from evaluate_failures import collect_failures
from diagnose_generate import run as diagnose_generate
from finetune_boost import finetune


def run_loop(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bcfg = cfg["boosting"]
    all_results = []

    for iteration in range(1, bcfg["max_iterations"] + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # Determine model dir
        if iteration == 1:
            model_dir = os.path.join(cfg["output_dir"], "baseline")
        else:
            model_dir = os.path.join(cfg["output_dir"], f"boosted_iter_{iteration - 1}")

        if not os.path.exists(model_dir):
            print(f"Model not found at {model_dir}. Run train_baseline.py first.")
            break

        # Step 1: Evaluate failures
        print(f"\n--- Step 1: Evaluate failures ---")
        collect_failures(config_path, model_dir)

        # Step 2-3: Diagnose + Generate
        print(f"\n--- Step 2-3: Diagnose + Generate ---")
        patterns, valid_pairs = diagnose_generate(config_path, iteration)

        if not valid_pairs:
            print("No valid pairs generated. Stopping loop.")
            break

        # Step 4: Finetune
        print(f"\n--- Step 4: Finetune ---")
        results = finetune(config_path, iteration, model_dir)

        if results is None:
            break

        all_results.append(results)

        # Check stopping condition
        ood_improvement = results["hans_improvement"]
        print(f"\nOOD improvement: {ood_improvement:+.4f} (threshold: {bcfg['improvement_threshold']})")

        if ood_improvement < bcfg["improvement_threshold"]:
            print(f"Improvement below threshold. Stopping after iteration {iteration}.")
            break

    # Save cumulative results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    cumulative = {
        "iterations": len(all_results),
        "total_pairs_generated": sum(r["num_cf_pairs"] for r in all_results),
        "per_iteration": all_results,
    }

    if all_results:
        cumulative["baseline_test_acc"] = all_results[0]["pre_test_acc"]
        cumulative["baseline_hans_acc"] = all_results[0]["pre_hans_acc"]
        cumulative["final_test_acc"] = all_results[-1]["post_test_acc"]
        cumulative["final_hans_acc"] = all_results[-1]["post_hans_acc"]
        cumulative["total_test_improvement"] = all_results[-1]["post_test_acc"] - all_results[0]["pre_test_acc"]
        cumulative["total_hans_improvement"] = all_results[-1]["post_hans_acc"] - all_results[0]["pre_hans_acc"]

        print(f"Iterations: {len(all_results)}")
        print(f"Total pairs: {cumulative['total_pairs_generated']}")
        print(f"Test: {cumulative['baseline_test_acc']:.4f} → {cumulative['final_test_acc']:.4f} ({cumulative['total_test_improvement']:+.4f})")
        print(f"HANS: {cumulative['baseline_hans_acc']:.4f} → {cumulative['final_hans_acc']:.4f} ({cumulative['total_hans_improvement']:+.4f})")

    output_path = os.path.join(cfg["output_dir"], "cumulative_results.json")
    with open(output_path, "w") as f:
        json.dump(cumulative, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_loop(args.config)
