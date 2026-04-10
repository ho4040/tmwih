"""Baseline methods for comparison: random augmentation, hard-label distillation."""

import argparse
import json
import os
import time

import yaml
from openai import OpenAI

from diagnose_generate import call_llm, extract_json

RANDOM_AUG_SYSTEM = """You are an expert at generating NLI (Natural Language Inference) training examples.
Generate diverse, natural premise-hypothesis pairs with labels (entailment, neutral, contradiction)."""

RANDOM_AUG_PROMPT = """Generate {n} NLI training examples as premise-hypothesis pairs.
Each example should have a clear, unambiguous label.
Make the examples diverse in topic, sentence structure, and difficulty.

Output as JSON array:
[{{
  "premise": "...",
  "hypothesis": "...",
  "label": "entailment|neutral|contradiction"
}}]"""

HARD_DISTILL_SYSTEM = """You are an expert NLI annotator.
Given premise-hypothesis pairs, assign the correct NLI label (entailment, neutral, contradiction)."""

HARD_DISTILL_PROMPT = """Assign NLI labels to these premise-hypothesis pairs:

{pairs}

Output as JSON array:
[{{"index": i, "label": "entailment|neutral|contradiction"}}]"""


def generate_random_augmentation(config_path: str, total_samples: int = 200):
    """Generate random NLI pairs using LLM (no model diagnosis)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bcfg = cfg["boosting"]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_path = os.path.expanduser("~/work/my-tools/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip('"').strip("'")
                        break
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    client = OpenAI(base_url=cfg["openrouter_base_url"], api_key=api_key)
    model = bcfg["teacher_model"]

    all_samples = []
    batch_size = 20

    for i in range(0, total_samples, batch_size):
        n = min(batch_size, total_samples - i)
        print(f"  Random augmentation batch {i // batch_size + 1}: generating {n} samples...")
        prompt = RANDOM_AUG_PROMPT.format(n=n)
        response = call_llm(client, model, RANDOM_AUG_SYSTEM, prompt, temperature=0.9)

        parsed = extract_json(response)
        if parsed is None:
            print(f"  Warning: Failed to parse batch")
            continue

        if isinstance(parsed, dict):
            for key in ["examples", "data", "pairs", "results"]:
                if key in parsed:
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]

        samples = [s for s in parsed if isinstance(s, dict) and "premise" in s and "hypothesis" in s and "label" in s]
        all_samples.extend(samples)
        print(f"    → {len(samples)} samples parsed")

    print(f"Generated {len(all_samples)} random samples total")

    # Save
    output_dir = os.path.join(cfg["output_dir"], "baselines", "random_augmentation")
    os.makedirs(output_dir, exist_ok=True)

    # Convert to CF pair format for compatibility with finetune_boost
    pairs = []
    for s in all_samples:
        pairs.append({
            "anchor_premise": s["premise"],
            "anchor_hypothesis": s["hypothesis"],
            "anchor_label": s["label"],
            "cf_premise": s["premise"],
            "cf_hypothesis": s["hypothesis"],
            "cf_label": s["label"],
            "changed_feature": "random_augmentation",
            "source_pattern": "random",
        })

    with open(os.path.join(output_dir, "pairs_valid.json"), "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_dir}")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["random"], default="random")
    parser.add_argument("--total_samples", type=int, default=200)
    args = parser.parse_args()

    if args.method == "random":
        generate_random_augmentation(args.config, args.total_samples)
