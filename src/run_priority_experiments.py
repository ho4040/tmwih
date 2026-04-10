"""Priority experiments for novelty:
1. DICT direct comparison (attention-only, CE-only, one-shot)
2. Attribution ablation (attention only vs gradient only vs both)
3. Multi-seed (3+ seeds with std dev)
4. Multi-task (FEVER)
"""

import argparse
import json
import os
import shutil
import statistics

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from openai import OpenAI

from data import load_snli, load_hans, load_fever, load_fever_symmetric, LABEL_MAP
from finetune_boost import CounterfactualPairDataset, evaluate
from run_comparison import finetune_with_pairs, finetune_simple
from run_ablations import finetune_ce_only
from diagnose_generate import (
    call_llm, extract_json, diagnose, generate_pairs, validate_pairs,
    DIAGNOSE_SYSTEM, GENERATE_SYSTEM, GENERATE_PROMPT,
)
from evaluate_failures import collect_failures


# ============================================================
# DICT-style diagnosis prompt (attention only, no gradient)
# ============================================================

DICT_DIAGNOSE_SYSTEM = """You are an expert NLP model diagnostician. You analyze why classification models make errors by examining their attention patterns.

Given failure cases with per-token attention scores, identify systematic patterns of over-reliance or missed signals that cause misclassification."""

DICT_DIAGNOSE_PROMPT = """Below are failure cases from an NLI model classifying premise-hypothesis pairs as entailment, neutral, or contradiction.

Each case includes:
- The premise and hypothesis
- The true label and the model's incorrect prediction
- Per-token attention scores (higher = model attended more to that token)

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

# Attribution ablation prompts

GRAD_ONLY_DIAGNOSE_PROMPT = """Below are failure cases from an NLI model classifying premise-hypothesis pairs.

Each case includes:
- The premise and hypothesis
- The true label and the model's incorrect prediction
- Per-token gradient attribution scores (higher = model was more sensitive to that token)

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


def _format_failures_attn_only(failures, max_n=50):
    """Format failures with attention scores only (DICT-style)."""
    formatted = []
    for i, f in enumerate(failures[:max_n]):
        top_tokens = sorted(f["token_attributions"], key=lambda x: x["attention"], reverse=True)[:5]
        attn_str = ", ".join(f'{t["token"]}:{t["attention"]}' for t in top_tokens)
        formatted.append(
            f"[{i}] Premise: {f['premise']}\n"
            f"    Hypothesis: {f['hypothesis']}\n"
            f"    True: {f['true_label']} | Predicted: {f['predicted_label']} | Conf: {f['confidence']}\n"
            f"    Top Attention: [{attn_str}]"
        )
    return formatted


def _format_failures_grad_only(failures, max_n=50):
    """Format failures with gradient scores only."""
    formatted = []
    for i, f in enumerate(failures[:max_n]):
        top_tokens = sorted(f["token_attributions"], key=lambda x: x["gradient"], reverse=True)[:5]
        grad_str = ", ".join(f'{t["token"]}:{t["gradient"]}' for t in top_tokens)
        formatted.append(
            f"[{i}] Premise: {f['premise']}\n"
            f"    Hypothesis: {f['hypothesis']}\n"
            f"    True: {f['true_label']} | Predicted: {f['predicted_label']} | Conf: {f['confidence']}\n"
            f"    Top Gradient: [{grad_str}]"
        )
    return formatted


def diagnose_dict_style(client, model, failures, temperature=0.7):
    """DICT-style diagnosis: attention only."""
    formatted = _format_failures_attn_only(failures)
    prompt = DICT_DIAGNOSE_PROMPT.format(failures="\n\n".join(formatted))
    print(f"DICT-style diagnosing {len(formatted)} failures (attention only)...")
    response = call_llm(client, model, DICT_DIAGNOSE_SYSTEM, prompt, temperature, log_dir="outputs/logs")
    parsed = extract_json(response)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        for key in ["patterns", "data", "results"]:
            if key in parsed:
                return parsed[key]
        return [parsed]
    return parsed if isinstance(parsed, list) else []


def diagnose_grad_only(client, model, failures, temperature=0.7):
    """Gradient-only diagnosis."""
    formatted = _format_failures_grad_only(failures)
    prompt = GRAD_ONLY_DIAGNOSE_PROMPT.format(failures="\n\n".join(formatted))
    print(f"Diagnosing {len(formatted)} failures (gradient only)...")
    response = call_llm(client, model, DICT_DIAGNOSE_SYSTEM, prompt, temperature, log_dir="outputs/logs")
    parsed = extract_json(response)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        for key in ["patterns", "data", "results"]:
            if key in parsed:
                return parsed[key]
        return [parsed]
    return parsed if isinstance(parsed, list) else []


def _get_client(cfg):
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
    return OpenAI(base_url=cfg["openrouter_base_url"], api_key=api_key)


# ============================================================
# 1. DICT Direct Comparison
# ============================================================

def _generate_dict_pairs(config_path, output_dir):
    """Generate CF pairs using DICT approach: attention-only diagnosis, one-shot."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    client = _get_client(cfg)
    teacher = cfg["boosting"]["teacher_model"]

    # Load failures
    failures_path = os.path.join(cfg["output_dir"], "failures.json")
    with open(failures_path) as f:
        data = json.load(f)
    failures = data["failures"]

    # DICT-style: attention-only diagnosis
    patterns = diagnose_dict_style(client, teacher, failures)

    # Generate pairs (same generation prompt)
    pairs = generate_pairs(client, teacher, patterns, cfg["boosting"]["samples_per_iteration"])

    # Validate
    valid_pairs = validate_pairs(client, teacher, pairs)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pairs_valid.json"), "w") as f:
        json.dump(valid_pairs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "patterns.json"), "w") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)

    print(f"DICT-style: {len(patterns)} patterns, {len(pairs)} generated, {len(valid_pairs)} valid")
    return valid_pairs


def run_dict_comparison(config_path):
    """Compare OUCH vs DICT at same sample counts."""
    print("\n" + "=" * 60)
    print("DICT DIRECT COMPARISON")
    print("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    baseline_dir = os.path.join(cfg["output_dir"], "baseline")
    out_base = os.path.join(cfg["output_dir"], "dict_comparison")

    # Generate DICT-style pairs
    dict_pairs_path = os.path.join(out_base, "dict_pairs", "pairs_valid.json")
    if os.path.exists(dict_pairs_path):
        with open(dict_pairs_path) as f:
            dict_pairs = json.load(f)
        print(f"Loaded {len(dict_pairs)} existing DICT pairs")
    else:
        dict_pairs = _generate_dict_pairs(config_path, os.path.join(out_base, "dict_pairs"))

    # Load OUCH pairs
    ouch_pairs = []
    for i in range(1, 20):
        path = os.path.join(cfg["output_dir"], f"iteration_{i}", "pairs_valid.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            ouch_pairs.extend(json.load(f))
    print(f"Loaded {len(ouch_pairs)} OUCH pairs")

    sample_counts = [50, 100, 200, 400]
    results = {"dict": {}, "ouch": {}}

    for n in sample_counts:
        # DICT: attention-only diagnosis + CE-only loss (DICT uses balanced CE, we use CE)
        if n <= len(dict_pairs):
            print(f"\n--- DICT with {n} pairs (CE only) ---")
            r = finetune_ce_only(config_path, dict_pairs[:n], baseline_dir,
                                 os.path.join(out_base, f"dict_{n}"))
            if r:
                results["dict"][n] = r

        # OUCH: both attribution + CE+contrastive
        if n <= len(ouch_pairs):
            print(f"\n--- OUCH with {n} pairs (CE + contrastive) ---")
            r = finetune_with_pairs(config_path, ouch_pairs[:n], baseline_dir,
                                     os.path.join(out_base, f"ouch_{n}"))
            if r:
                results["ouch"][n] = r

    # Summary
    print("\n" + "=" * 60)
    print("DICT vs OUCH Comparison")
    print("=" * 60)
    print(f"{'Samples':>8} | {'DICT HANS':>12} | {'OUCH HANS':>12} | {'Gap':>8}")
    print("-" * 50)
    for n in sample_counts:
        d_hans = f"{results['dict'][n]['post_hans_acc']:.4f}" if n in results['dict'] else "—"
        o_hans = f"{results['ouch'][n]['post_hans_acc']:.4f}" if n in results['ouch'] else "—"
        gap = ""
        if n in results['dict'] and n in results['ouch']:
            g = results['ouch'][n]['post_hans_acc'] - results['dict'][n]['post_hans_acc']
            gap = f"{g:+.4f}"
        print(f"{n:>8} | {d_hans:>12} | {o_hans:>12} | {gap:>8}")

    os.makedirs(out_base, exist_ok=True)
    with open(os.path.join(out_base, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)

    return results


# ============================================================
# 2. Attribution Ablation
# ============================================================

def run_attribution_ablation(config_path):
    """Compare attention only vs gradient only vs both."""
    print("\n" + "=" * 60)
    print("ATTRIBUTION ABLATION")
    print("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    client = _get_client(cfg)
    teacher = cfg["boosting"]["teacher_model"]
    baseline_dir = os.path.join(cfg["output_dir"], "baseline")
    out_base = os.path.join(cfg["output_dir"], "attribution_ablation")

    failures_path = os.path.join(cfg["output_dir"], "failures.json")
    with open(failures_path) as f:
        data = json.load(f)
    failures = data["failures"]

    conditions = {
        "attention_only": lambda: diagnose_dict_style(client, teacher, failures),
        "gradient_only": lambda: diagnose_grad_only(client, teacher, failures),
        "both": lambda: diagnose(client, teacher, failures),
    }

    results = {}
    for cond_name, diagnose_fn in conditions.items():
        print(f"\n--- {cond_name} ---")
        pairs_path = os.path.join(out_base, cond_name, "pairs_valid.json")

        if os.path.exists(pairs_path):
            with open(pairs_path) as f:
                valid_pairs = json.load(f)
            print(f"  Loaded {len(valid_pairs)} existing pairs")
        else:
            patterns = diagnose_fn()
            pairs = generate_pairs(client, teacher, patterns, cfg["boosting"]["samples_per_iteration"])
            valid_pairs = validate_pairs(client, teacher, pairs)
            os.makedirs(os.path.join(out_base, cond_name), exist_ok=True)
            with open(pairs_path, "w") as f:
                json.dump(valid_pairs, f, indent=2, ensure_ascii=False)
            with open(os.path.join(out_base, cond_name, "patterns.json"), "w") as f:
                json.dump(patterns, f, indent=2, ensure_ascii=False)

        if not valid_pairs:
            continue

        # Finetune with 200 pairs, CE only (to isolate attribution effect)
        n = min(200, len(valid_pairs))
        r = finetune_ce_only(config_path, valid_pairs[:n], baseline_dir,
                              os.path.join(out_base, f"{cond_name}_finetune"))
        if r:
            r["condition"] = cond_name
            r["num_pairs"] = len(valid_pairs)
            results[cond_name] = r

    print("\n--- Attribution Ablation Summary ---")
    for cond, r in results.items():
        print(f"  {cond}: HANS={r['post_hans_acc']:.4f} ({r['hans_improvement']:+.4f}), pairs={r['num_pairs']}")

    os.makedirs(out_base, exist_ok=True)
    with open(os.path.join(out_base, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)

    return results


# ============================================================
# 3. Multi-seed
# ============================================================

def train_baseline_with_seed(config_path, seed, output_suffix):
    """Train baseline with a specific seed."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["student_model"], num_labels=cfg["num_labels"]
    ).to(device)

    bcfg = cfg["baseline"]
    ds = load_snli(tokenizer, bcfg["max_seq_length"])
    train_loader = DataLoader(ds["train"], batch_size=bcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=bcfg["batch_size"])
    test_loader = DataLoader(ds["test"], batch_size=32)
    hans_ds = load_hans(tokenizer, bcfg["max_seq_length"])
    hans_loader = DataLoader(hans_ds, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(bcfg["learning_rate"]),
                                   weight_decay=float(bcfg["weight_decay"]))
    total_steps = len(train_loader) * bcfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * bcfg["warmup_ratio"]), total_steps)

    output_dir = os.path.join(cfg["output_dir"], f"baseline_{output_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    best_val_acc = 0.0

    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    def eval_acc(model, loader, device):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
                preds.extend(out.logits.argmax(-1).cpu().tolist())
                labels.extend(batch["label"].tolist())
        return accuracy_score(labels, preds)

    for epoch in range(bcfg["num_epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}"):
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["label"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_acc = eval_acc(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_acc = eval_acc(model, test_loader, device)
    hans_acc = eval_acc(model, hans_loader, device)

    results = {"seed": seed, "test_acc": test_acc, "hans_acc": hans_acc, "best_val_acc": best_val_acc}
    with open(os.path.join(output_dir, "results.yaml"), "w") as f:
        yaml.dump(results, f)

    print(f"Seed {seed}: test={test_acc:.4f}, hans={hans_acc:.4f}")
    return results


def run_ouch_on_baseline(config_path, baseline_dir, output_suffix):
    """Run OUCH pipeline on a specific baseline model."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    client = _get_client(cfg)
    teacher = cfg["boosting"]["teacher_model"]

    # Collect failures from this baseline
    failures_path = os.path.join(cfg["output_dir"], f"failures_{output_suffix}.json")
    if os.path.exists(failures_path):
        with open(failures_path) as f:
            data = json.load(f)
        failures = data["failures"]
    else:
        # Temporarily override output to collect failures
        from evaluate_failures import collect_failures as _collect
        # We need to override the model dir
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            baseline_dir, attn_implementation="eager"
        ).to(device)
        model.eval()

        from evaluate_failures import compute_gradient_attribution
        from data import decode_pair, LABEL_NAMES

        ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
        val_loader = DataLoader(ds["validation"], batch_size=1, shuffle=False)

        failures = []
        for batch in val_loader:
            if len(failures) >= cfg["boosting"]["max_failures"]:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].item()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                pred = outputs.logits.argmax(-1).item()
                confidence = torch.softmax(outputs.logits, -1).max().item()

            if pred == label:
                continue

            last_attn = outputs.attentions[-1]
            cls_attn = last_attn[0, :, 0, :].mean(dim=0)
            grad_attr = compute_gradient_attribution(model, input_ids, attention_mask, label)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
            premise, hypothesis = decode_pair(tokenizer, input_ids[0].cpu())

            token_info = []
            for i, tok in enumerate(tokens):
                if tok in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                    continue
                token_info.append({
                    "token": tok, "attention": round(cls_attn[i].item(), 4),
                    "gradient": round(grad_attr[i].item(), 4),
                })

            failures.append({
                "premise": premise, "hypothesis": hypothesis,
                "true_label": LABEL_NAMES[label], "predicted_label": LABEL_NAMES[pred],
                "confidence": round(confidence, 4), "token_attributions": token_info,
            })

        with open(failures_path, "w") as f:
            json.dump({"failures": failures}, f, indent=2, ensure_ascii=False)
        print(f"  Collected {len(failures)} failures for {output_suffix}")

    # Diagnose + Generate + Validate
    patterns = diagnose(client, teacher, failures)
    pairs = generate_pairs(client, teacher, patterns, cfg["boosting"]["samples_per_iteration"])
    valid_pairs = validate_pairs(client, teacher, pairs)

    out_dir = os.path.join(cfg["output_dir"], f"multiseed_{output_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pairs_valid.json"), "w") as f:
        json.dump(valid_pairs, f, indent=2, ensure_ascii=False)

    # Finetune with OUCH pairs
    if valid_pairs:
        n = min(200, len(valid_pairs))
        r = finetune_with_pairs(config_path, valid_pairs[:n], baseline_dir,
                                 os.path.join(out_dir, "finetune"))
        if r:
            r["num_pairs"] = len(valid_pairs)
        return r
    return None


def run_multi_seed(config_path, seeds=None):
    """Run baseline + OUCH with multiple seeds."""
    print("\n" + "=" * 60)
    print("MULTI-SEED EXPERIMENT")
    print("=" * 60)

    if seeds is None:
        seeds = [42, 123, 456]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_base = os.path.join(cfg["output_dir"], "multiseed")
    os.makedirs(out_base, exist_ok=True)

    all_results = {}
    for seed in seeds:
        suffix = f"seed{seed}"
        baseline_dir = os.path.join(cfg["output_dir"], f"baseline_{suffix}")
        results_path = os.path.join(baseline_dir, "results.yaml")

        # Train baseline
        if os.path.exists(results_path):
            with open(results_path) as f:
                bl_result = yaml.safe_load(f)
            print(f"Loaded existing baseline for seed {seed}: test={bl_result['test_acc']:.4f}, hans={bl_result['hans_acc']:.4f}")
        else:
            bl_result = train_baseline_with_seed(config_path, seed, suffix)

        # Run OUCH
        ouch_result_path = os.path.join(cfg["output_dir"], f"multiseed_{suffix}", "finetune", "results.yaml")
        if os.path.exists(ouch_result_path):
            with open(ouch_result_path) as f:
                ouch_result = yaml.safe_load(f)
            print(f"Loaded existing OUCH for seed {seed}")
        else:
            ouch_result = run_ouch_on_baseline(config_path, baseline_dir, suffix)

        all_results[seed] = {
            "baseline": bl_result,
            "ouch": ouch_result,
        }

    # Summary with mean ± std
    print("\n" + "=" * 60)
    print("MULTI-SEED RESULTS")
    print("=" * 60)

    bl_test = [r["baseline"]["test_acc"] for r in all_results.values()]
    bl_hans = [r["baseline"]["hans_acc"] for r in all_results.values()]
    ouch_hans = [r["ouch"]["post_hans_acc"] for r in all_results.values() if r["ouch"]]
    ouch_test = [r["ouch"]["post_test_acc"] for r in all_results.values() if r["ouch"]]

    print(f"\nBaseline Test:  {statistics.mean(bl_test):.4f} ± {statistics.stdev(bl_test):.4f}")
    print(f"Baseline HANS:  {statistics.mean(bl_hans):.4f} ± {statistics.stdev(bl_hans):.4f}")
    if ouch_hans:
        print(f"OUCH Test:      {statistics.mean(ouch_test):.4f} ± {statistics.stdev(ouch_test):.4f}")
        print(f"OUCH HANS:      {statistics.mean(ouch_hans):.4f} ± {statistics.stdev(ouch_hans):.4f}")
        deltas = [o - b for o, b in zip(ouch_hans, bl_hans[:len(ouch_hans)])]
        print(f"HANS Δ:         {statistics.mean(deltas):+.4f} ± {statistics.stdev(deltas):.4f}")

    with open(os.path.join(out_base, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)

    return all_results


# ============================================================
# 4. FEVER (Multi-task)
# ============================================================

def run_fever(config_path):
    """Run OUCH pipeline on FEVER task."""
    print("\n" + "=" * 60)
    print("MULTI-TASK: FEVER")
    print("=" * 60)

    # Create FEVER config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    fever_cfg = dict(cfg)
    fever_cfg["dataset"] = "fever"
    fever_cfg["output_dir"] = os.path.join(cfg["output_dir"], "fever")

    fever_config_path = os.path.join(os.path.dirname(config_path), "fever.yaml")
    with open(fever_config_path, "w") as f:
        yaml.dump(fever_cfg, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = fever_cfg["baseline"]
    baseline_dir = os.path.join(fever_cfg["output_dir"], "baseline")

    # Train FEVER baseline
    if os.path.exists(os.path.join(baseline_dir, "results.yaml")):
        print("FEVER baseline already trained")
        with open(os.path.join(baseline_dir, "results.yaml")) as f:
            bl_result = yaml.safe_load(f)
    else:
        import random
        import numpy as np
        torch.manual_seed(fever_cfg.get("seed", 42))
        torch.cuda.manual_seed_all(fever_cfg.get("seed", 42))
        random.seed(fever_cfg.get("seed", 42))
        np.random.seed(fever_cfg.get("seed", 42))

        tokenizer = AutoTokenizer.from_pretrained(fever_cfg["student_model"])
        model = AutoModelForSequenceClassification.from_pretrained(
            fever_cfg["student_model"], num_labels=fever_cfg["num_labels"]
        ).to(device)

        print("Loading FEVER...")
        fever_ds = load_fever(tokenizer, bcfg["max_seq_length"])
        train_loader = DataLoader(fever_ds["train"], batch_size=bcfg["batch_size"], shuffle=True)
        val_loader = DataLoader(fever_ds["validation"], batch_size=32)
        test_loader = DataLoader(fever_ds["test"], batch_size=32)

        print("Loading FEVER Symmetric...")
        fever_sym = load_fever_symmetric(tokenizer, bcfg["max_seq_length"])
        sym_loader = DataLoader(fever_sym, batch_size=32) if fever_sym else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(bcfg["learning_rate"]),
                                       weight_decay=float(bcfg["weight_decay"]))
        total_steps = len(train_loader) * bcfg["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * bcfg["warmup_ratio"]), total_steps)

        os.makedirs(baseline_dir, exist_ok=True)
        best_val_acc = 0.0

        from tqdm import tqdm

        for epoch in range(bcfg["num_epochs"]):
            model.train()
            for batch in tqdm(train_loader, desc=f"FEVER Epoch {epoch+1}"):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["label"].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            val_acc = evaluate(model, val_loader, device)
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(baseline_dir)
                tokenizer.save_pretrained(baseline_dir)

        model = AutoModelForSequenceClassification.from_pretrained(baseline_dir).to(device)
        test_acc = evaluate(model, test_loader, device)
        sym_acc = evaluate(model, sym_loader, device) if sym_loader else None

        bl_result = {"test_acc": test_acc, "sym_acc": sym_acc, "best_val_acc": best_val_acc}
        with open(os.path.join(baseline_dir, "results.yaml"), "w") as f:
            yaml.dump(bl_result, f)
        print(f"FEVER baseline: test={test_acc:.4f}, symmetric={sym_acc}")

    # Run OUCH on FEVER
    ouch_result = run_ouch_on_baseline(fever_config_path, baseline_dir, "fever")

    return {"baseline": bl_result, "ouch": ouch_result}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", choices=["all", "dict", "attribution", "multiseed", "fever"],
                        default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    if args.experiment == "all":
        run_dict_comparison(args.config)
        run_attribution_ablation(args.config)
        run_multi_seed(args.config, args.seeds)
        run_fever(args.config)
    elif args.experiment == "dict":
        run_dict_comparison(args.config)
    elif args.experiment == "attribution":
        run_attribution_ablation(args.config)
    elif args.experiment == "multiseed":
        run_multi_seed(args.config, args.seeds)
    elif args.experiment == "fever":
        run_fever(args.config)
