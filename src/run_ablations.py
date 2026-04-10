"""Run all ablation experiments:
1. Contrastive loss ablation (CE only vs CE + contrastive)
2. Hard-label distillation baseline
3. LLM dependency (gpt-5.4 vs mini vs nano)
"""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from data import load_snli, load_hans, LABEL_MAP
from finetune_boost import CounterfactualPairDataset, contrastive_loss_fn, evaluate
from run_comparison import SimpleNLIDataset, finetune_simple
from diagnose_generate import call_llm, extract_json, DIAGNOSE_SYSTEM, DIAGNOSE_PROMPT, GENERATE_SYSTEM, GENERATE_PROMPT
from evaluate_failures import collect_failures
from openai import OpenAI


# ============================================================
# 1. Contrastive Loss Ablation
# ============================================================

def finetune_ce_only(config_path: str, pairs: list, model_dir: str, output_dir: str):
    """Finetune with CE loss only (no contrastive) on CF pairs."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = cfg["boosting"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    cf_dataset = CounterfactualPairDataset(pairs, tokenizer, cfg["baseline"]["max_seq_length"])
    print(f"  Loaded {len(cf_dataset)} pairs")

    if len(cf_dataset) == 0:
        return None

    cf_loader = DataLoader(cf_dataset, batch_size=bcfg["finetune_batch_size"], shuffle=True)

    orig_ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
    n_orig = int(len(cf_dataset) * bcfg["original_data_ratio"] / (1 - bcfg["original_data_ratio"]))
    n_orig = min(n_orig, len(orig_ds["train"]))
    orig_subset = orig_ds["train"].select(range(n_orig))
    orig_loader = DataLoader(orig_subset, batch_size=bcfg["finetune_batch_size"], shuffle=True)

    snli_ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
    test_loader = DataLoader(snli_ds["test"], batch_size=32)
    hans_ds = load_hans(tokenizer, cfg["baseline"]["max_seq_length"])
    hans_loader = DataLoader(hans_ds, batch_size=32)

    pre_test = evaluate(model, test_loader, device)
    pre_hans = evaluate(model, hans_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(bcfg["finetune_learning_rate"]), weight_decay=0.01)
    total_steps = (len(cf_loader) + len(orig_loader)) * bcfg["finetune_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    for epoch in range(bcfg["finetune_epochs"]):
        model.train()
        # CE only on both anchor and CF (no contrastive loss)
        for batch in cf_loader:
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            anchor_labels = batch["anchor_label"].to(device)
            cf_ids = batch["cf_input_ids"].to(device)
            cf_mask = batch["cf_attention_mask"].to(device)
            cf_labels = batch["cf_label"].to(device)

            anchor_out = model(input_ids=anchor_ids, attention_mask=anchor_mask, labels=anchor_labels)
            cf_out = model(input_ids=cf_ids, attention_mask=cf_mask, labels=cf_labels)
            loss = (anchor_out.loss + cf_out.loss) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        for batch in orig_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    post_test = evaluate(model, test_loader, device)
    post_hans = evaluate(model, hans_loader, device)

    results = {
        "num_pairs": len(cf_dataset),
        "pre_test_acc": pre_test,
        "pre_hans_acc": pre_hans,
        "post_test_acc": post_test,
        "post_hans_acc": post_hans,
        "test_improvement": post_test - pre_test,
        "hans_improvement": post_hans - pre_hans,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.yaml"), "w") as f:
        yaml.dump(results, f)

    print(f"  Result: test={post_test:.4f} ({post_test - pre_test:+.4f}), hans={post_hans:.4f} ({post_hans - pre_hans:+.4f})")
    return results


def run_contrastive_ablation(config_path: str):
    """Compare CE only vs CE + contrastive on informed pairs."""
    print("\n" + "=" * 60)
    print("ABLATION 1: Contrastive Loss")
    print("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    baseline_dir = os.path.join(cfg["output_dir"], "baseline")

    # Load all informed pairs
    informed_pairs = []
    for i in range(1, 20):
        path = os.path.join(cfg["output_dir"], f"iteration_{i}", "pairs_valid.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            informed_pairs.extend(json.load(f))

    results = {}
    for n in [200, 400]:
        if n > len(informed_pairs):
            continue
        subset = informed_pairs[:n]

        print(f"\n--- CE only, {n} pairs ---")
        r1 = finetune_ce_only(config_path, subset, baseline_dir,
                              os.path.join(cfg["output_dir"], "ablation_contrastive", f"ce_only_{n}"))

        print(f"\n--- CE + contrastive, {n} pairs ---")
        from run_comparison import finetune_with_pairs
        r2 = finetune_with_pairs(config_path, subset, baseline_dir,
                                  os.path.join(cfg["output_dir"], "ablation_contrastive", f"ce_contrastive_{n}"))

        results[n] = {"ce_only": r1, "ce_contrastive": r2}

    print("\n--- Contrastive Ablation Summary ---")
    for n, r in results.items():
        ce_hans = r["ce_only"]["post_hans_acc"] if r["ce_only"] else "—"
        cc_hans = r["ce_contrastive"]["post_hans_acc"] if r["ce_contrastive"] else "—"
        print(f"  {n} pairs: CE only HANS={ce_hans:.4f}, CE+CL HANS={cc_hans:.4f}")

    return results


# ============================================================
# 2. Hard-Label Distillation Baseline
# ============================================================

DISTILL_SYSTEM = """You are an expert NLI annotator. Given a premise, generate a hypothesis and assign the correct NLI label."""

DISTILL_PROMPT = """Generate {n} NLI examples. For each, I provide a premise from the SNLI dataset.
Write a hypothesis and assign the correct label (entailment, neutral, contradiction).

Premises:
{premises}

Output as JSON array:
[{{"premise": "the original premise", "hypothesis": "your hypothesis", "label": "entailment|neutral|contradiction"}}]"""


def generate_hard_distillation(config_path: str, total_samples: int = 400):
    """Generate hard-label distillation data: LLM labels on SNLI-like data."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bcfg = cfg["boosting"]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    client = OpenAI(base_url=cfg["openrouter_base_url"], api_key=api_key)
    model = bcfg["teacher_model"]

    # Load some premises from SNLI for context
    from datasets import load_dataset
    snli = load_dataset("stanfordnlp/snli", split="train")
    snli = snli.filter(lambda x: x["label"] != -1)
    premises = [snli[i]["premise"] for i in range(min(2000, len(snli)))]

    all_samples = []
    batch_size = 20
    premise_idx = 0

    for i in range(0, total_samples, batch_size):
        n = min(batch_size, total_samples - i)
        batch_premises = premises[premise_idx:premise_idx + n]
        premise_idx += n
        if premise_idx >= len(premises):
            premise_idx = 0

        premises_text = "\n".join(f"[{j}] {p}" for j, p in enumerate(batch_premises))
        prompt = DISTILL_PROMPT.format(n=n, premises=premises_text)

        print(f"  Hard-label distillation batch {i // batch_size + 1}: generating {n}...")
        response = call_llm(client, model, DISTILL_SYSTEM, prompt, temperature=0.7)

        parsed = extract_json(response)
        if parsed is None:
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
        print(f"    → {len(samples)} samples")

    print(f"Generated {len(all_samples)} hard-label distillation samples")

    output_dir = os.path.join(cfg["output_dir"], "baselines", "hard_distillation")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "samples.json"), "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    return all_samples


def run_hard_distillation(config_path: str):
    """Run hard-label distillation baseline."""
    print("\n" + "=" * 60)
    print("ABLATION 2: Hard-Label Distillation")
    print("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    baseline_dir = os.path.join(cfg["output_dir"], "baseline")

    # Generate or load distillation data
    samples_path = os.path.join(cfg["output_dir"], "baselines", "hard_distillation", "samples.json")
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} existing distillation samples")
    else:
        samples = generate_hard_distillation(config_path, total_samples=400)

    results = {}
    for n in [50, 100, 200, 400]:
        if n > len(samples):
            continue
        print(f"\n--- Hard-label distillation with {n} samples ---")
        subset = samples[:n]
        out_dir = os.path.join(cfg["output_dir"], "comparison", f"distill_{n}")
        r = finetune_simple(config_path, subset, baseline_dir, out_dir)
        if r:
            results[n] = r

    return results


# ============================================================
# 3. LLM Dependency
# ============================================================

def generate_informed_with_model(config_path: str, llm_model: str, total_samples: int = 200):
    """Generate informed CF pairs using a specific LLM model."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found")

    client = OpenAI(base_url=cfg["openrouter_base_url"], api_key=api_key)

    # Load failures
    failures_path = os.path.join(cfg["output_dir"], "failures.json")
    with open(failures_path) as f:
        data = json.load(f)
    failures = data["failures"]

    # Diagnose with this model
    from diagnose_generate import diagnose, generate_pairs, validate_pairs
    print(f"  Diagnosing with {llm_model}...")
    patterns = diagnose(client, llm_model, failures, temperature=0.7)

    print(f"  Generating pairs with {llm_model}...")
    pairs = generate_pairs(client, llm_model, patterns, total_samples, temperature=0.7)

    print(f"  Validating with {llm_model}...")
    valid_pairs = validate_pairs(client, llm_model, pairs, temperature=0.2)

    model_short = llm_model.split("/")[-1]
    output_dir = os.path.join(cfg["output_dir"], "ablation_llm", model_short)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pairs_valid.json"), "w") as f:
        json.dump(valid_pairs, f, indent=2, ensure_ascii=False)

    print(f"  {llm_model}: {len(pairs)} generated, {len(valid_pairs)} valid")
    return valid_pairs


def run_llm_dependency(config_path: str):
    """Test with different LLM tiers."""
    print("\n" + "=" * 60)
    print("ABLATION 3: LLM Dependency")
    print("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    baseline_dir = os.path.join(cfg["output_dir"], "baseline")
    models = [
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
    ]

    results = {}
    for llm_model in models:
        model_short = llm_model.split("/")[-1]
        print(f"\n--- LLM: {llm_model} ---")

        # Check for existing pairs
        pairs_path = os.path.join(cfg["output_dir"], "ablation_llm", model_short, "pairs_valid.json")
        if os.path.exists(pairs_path):
            with open(pairs_path) as f:
                pairs = json.load(f)
            print(f"  Loaded {len(pairs)} existing pairs")
        else:
            # For gpt-5.4 we can reuse iteration_1 pairs
            if llm_model == "openai/gpt-5.4":
                iter1_path = os.path.join(cfg["output_dir"], "iteration_1", "pairs_valid.json")
                if os.path.exists(iter1_path):
                    with open(iter1_path) as f:
                        pairs = json.load(f)
                    print(f"  Reusing {len(pairs)} pairs from iteration_1")
                else:
                    pairs = generate_informed_with_model(config_path, llm_model, 200)
            else:
                pairs = generate_informed_with_model(config_path, llm_model, 200)

        if not pairs:
            print(f"  No valid pairs for {llm_model}, skipping")
            continue

        # Finetune with 200 pairs (or all available)
        n = min(200, len(pairs))
        subset = pairs[:n]

        from run_comparison import finetune_with_pairs
        out_dir = os.path.join(cfg["output_dir"], "ablation_llm", f"{model_short}_finetune")
        r = finetune_with_pairs(config_path, subset, baseline_dir, out_dir)
        if r:
            r["llm_model"] = llm_model
            r["pairs_generated"] = len(pairs)
            results[model_short] = r

    print("\n--- LLM Dependency Summary ---")
    for model, r in results.items():
        print(f"  {model}: HANS={r['post_hans_acc']:.4f} ({r['hans_improvement']:+.4f}), pairs={r.get('pairs_generated', '?')}")

    return results


# ============================================================
# Main
# ============================================================

def run_all(config_path: str):
    all_results = {}

    # 1. Contrastive ablation
    all_results["contrastive"] = run_contrastive_ablation(config_path)

    # 2. Hard-label distillation
    all_results["distillation"] = run_hard_distillation(config_path)

    # 3. LLM dependency
    all_results["llm_dependency"] = run_llm_dependency(config_path)

    # Final summary
    print("\n" + "=" * 70)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 70)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out_path = os.path.join(cfg["output_dir"], "ablation_results.json")

    # Convert to serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(out_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_all(args.config)
