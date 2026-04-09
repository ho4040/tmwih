"""Step 2-3: Diagnose failure patterns and generate counterfactual minimal pairs via LLM."""

import argparse
import json
import os
import time

import yaml
from openai import OpenAI

from data import LABEL_NAMES

DIAGNOSE_SYSTEM = """You are an expert NLP model diagnostician. You analyze why classification models make errors by examining their internal signals (attention and gradient attribution scores).

Given failure cases with per-token attention and gradient values, identify systematic patterns of over-reliance or missed signals that cause misclassification."""

DIAGNOSE_PROMPT = """Below are failure cases from an NLI (Natural Language Inference) model classifying premise-hypothesis pairs as entailment, neutral, or contradiction.

Each case includes:
- The premise and hypothesis
- The true label and the model's incorrect prediction
- Per-token attention and gradient attribution scores (higher = model relied more on that token)

Analyze these failures and identify 3-5 systematic weakness patterns.

Failure cases:
{failures}

Output as JSON array:
[{{
  "pattern_id": "short_identifier",
  "description": "What the model gets wrong",
  "overrelied_tokens": ["tokens the model over-relies on"],
  "missed_signal": "What linguistic signal the model misses",
  "example_indices": [indices of failure cases matching this pattern],
  "count": number of cases matching
}}]"""

GENERATE_SYSTEM = """You are an expert at generating counterfactual minimal pairs for NLI (Natural Language Inference).

A minimal pair consists of two premise-hypothesis pairs that differ minimally (1-2 words changed) but have different NLI labels (entailment, neutral, contradiction).

The pairs should be natural, grammatically correct, and the label difference should be unambiguous."""

GENERATE_PROMPT = """Generate {n} counterfactual minimal pairs targeting this specific model weakness:

Weakness: {description}
Over-relied tokens: {overrelied_tokens}
Missed signal: {missed_signal}

Each pair should:
1. Include the over-relied tokens but in contexts where they DON'T determine the label
2. Differ by only 1-2 words between anchor and counterfactual
3. Have clear, unambiguous NLI labels

Output as JSON array:
[{{
  "anchor_premise": "premise text",
  "anchor_hypothesis": "hypothesis text",
  "anchor_label": "entailment|neutral|contradiction",
  "cf_premise": "same or minimally changed premise",
  "cf_hypothesis": "minimally changed hypothesis",
  "cf_label": "different label",
  "changed_feature": "what was changed and why"
}}]"""

VALIDATE_SYSTEM = """You are an NLI annotation expert. Given premise-hypothesis pairs, verify if the assigned NLI labels are correct.

Labels:
- entailment: hypothesis is definitely true given the premise
- neutral: hypothesis might be true given the premise
- contradiction: hypothesis is definitely false given the premise"""

VALIDATE_PROMPT = """Verify the NLI labels for these pairs. For each pair, respond with "correct" or "incorrect" and a brief reason.

Pairs:
{pairs}

Output as JSON array:
[{{"index": i, "anchor_valid": true/false, "cf_valid": true/false, "reason": "..."}}]"""


def call_llm(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.7) -> str:
    """Call LLM via OpenRouter."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return "[]"


def diagnose(client: OpenAI, model: str, failures: list, temperature: float = 0.7) -> list:
    """Diagnose failure patterns using LLM."""
    # Format failures for the prompt (limit to avoid token overflow)
    formatted = []
    for i, f in enumerate(failures[:50]):  # max 50 for diagnosis
        top_tokens = sorted(f["token_attributions"], key=lambda x: x["gradient"], reverse=True)[:5]
        attn_str = ", ".join(f'{t["token"]}:{t["attention"]}' for t in top_tokens)
        grad_str = ", ".join(f'{t["token"]}:{t["gradient"]}' for t in top_tokens)
        formatted.append(
            f"[{i}] Premise: {f['premise']}\n"
            f"    Hypothesis: {f['hypothesis']}\n"
            f"    True: {f['true_label']} | Predicted: {f['predicted_label']} | Conf: {f['confidence']}\n"
            f"    Top Attention: [{attn_str}]\n"
            f"    Top Gradient: [{grad_str}]"
        )

    prompt = DIAGNOSE_PROMPT.format(failures="\n\n".join(formatted))
    print(f"Diagnosing {len(formatted)} failures...")

    response = call_llm(client, model, DIAGNOSE_SYSTEM, prompt, temperature)

    try:
        patterns = json.loads(response)
        if isinstance(patterns, dict) and "patterns" in patterns:
            patterns = patterns["patterns"]
        if not isinstance(patterns, list):
            patterns = [patterns]
    except json.JSONDecodeError:
        print("Warning: Failed to parse diagnosis response")
        patterns = []

    print(f"Found {len(patterns)} weakness patterns")
    return patterns


def generate_pairs(client: OpenAI, model: str, patterns: list, samples_per_iter: int, temperature: float = 0.7) -> list:
    """Generate counterfactual minimal pairs for each diagnosed pattern."""
    all_pairs = []
    per_pattern = max(1, samples_per_iter // len(patterns)) if patterns else 0
    batch_size = 20  # generate in small batches to avoid LLM refusal

    for pattern in patterns:
        remaining = per_pattern
        batch_num = 0
        while remaining > 0:
            n = min(batch_size, remaining)
            batch_num += 1
            print(f"  Pattern '{pattern['pattern_id']}' batch {batch_num}: generating {n} pairs...")
            prompt = GENERATE_PROMPT.format(
                n=n,
                description=pattern["description"],
                overrelied_tokens=", ".join(pattern.get("overrelied_tokens", [])),
                missed_signal=pattern.get("missed_signal", "unknown"),
            )

            response = call_llm(client, model, GENERATE_SYSTEM, prompt, temperature)

            try:
                pairs = json.loads(response)
                if isinstance(pairs, dict):
                    # Try common wrapper keys
                    for key in ["pairs", "data", "examples", "results"]:
                        if key in pairs:
                            pairs = pairs[key]
                            break
                    else:
                        pairs = [pairs]
                if not isinstance(pairs, list):
                    pairs = [pairs]
                # Filter out error responses
                pairs = [p for p in pairs if "anchor_premise" in p or "anchor_hypothesis" in p]
                for p in pairs:
                    p["source_pattern"] = pattern["pattern_id"]
                all_pairs.extend(pairs)
                remaining -= n
            except json.JSONDecodeError:
                print(f"  Warning: Failed to parse batch {batch_num}")
                remaining -= n  # skip to avoid infinite loop

    print(f"Generated {len(all_pairs)} pairs total")
    return all_pairs


def validate_pairs(client: OpenAI, model: str, pairs: list, temperature: float = 0.2) -> list:
    """Validate generated pairs using a separate LLM call."""
    if not pairs:
        return []

    # Validate in batches
    batch_size = 20
    valid_pairs = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        formatted = []
        for j, p in enumerate(batch):
            formatted.append(
                f"[{j}] Anchor: P=\"{p.get('anchor_premise', '')}\" H=\"{p.get('anchor_hypothesis', '')}\" Label={p.get('anchor_label', '')}\n"
                f"     CF:     P=\"{p.get('cf_premise', '')}\" H=\"{p.get('cf_hypothesis', '')}\" Label={p.get('cf_label', '')}"
            )

        prompt = VALIDATE_PROMPT.format(pairs="\n\n".join(formatted))
        response = call_llm(client, model, VALIDATE_SYSTEM, prompt, temperature)

        try:
            validations = json.loads(response)
            if isinstance(validations, dict) and "validations" in validations:
                validations = validations["validations"]
            if not isinstance(validations, list):
                validations = [validations]

            for v in validations:
                idx = v.get("index", -1)
                if 0 <= idx < len(batch) and v.get("anchor_valid") and v.get("cf_valid"):
                    valid_pairs.append(batch[idx])
        except json.JSONDecodeError:
            # If validation parsing fails, keep all pairs in this batch
            valid_pairs.extend(batch)

    print(f"Validated: {len(valid_pairs)}/{len(pairs)} pairs passed")
    return valid_pairs


def run(config_path: str, iteration: int = 1):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bcfg = cfg["boosting"]

    # Load API key from env
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try loading from .env file
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

    # Load failures
    failures_path = os.path.join(cfg["output_dir"], "failures.json")
    with open(failures_path) as f:
        data = json.load(f)
    failures = data["failures"]

    # Step 2: Diagnose
    patterns = diagnose(client, bcfg["teacher_model"], failures, bcfg["temperature"])

    # Step 3: Generate
    pairs = generate_pairs(client, bcfg["teacher_model"], patterns, bcfg["samples_per_iteration"], bcfg["temperature"])

    # Step 3.5: Validate
    valid_pairs = validate_pairs(client, bcfg["teacher_model"], pairs, temperature=0.2)

    # Save results
    output_dir = os.path.join(cfg["output_dir"], f"iteration_{iteration}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "patterns.json"), "w") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "pairs_all.json"), "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "pairs_valid.json"), "w") as f:
        json.dump(valid_pairs, f, indent=2, ensure_ascii=False)

    print(f"\nIteration {iteration} complete:")
    print(f"  Patterns: {len(patterns)}")
    print(f"  Generated: {len(pairs)}")
    print(f"  Valid: {len(valid_pairs)}")
    print(f"  Saved to: {output_dir}")

    return patterns, valid_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()
    run(args.config, args.iteration)
