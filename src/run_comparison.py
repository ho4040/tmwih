"""Run comparison experiments: informed (ours) vs random augmentation at different sample counts."""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from data import load_snli, load_hans, LABEL_MAP
from finetune_boost import CounterfactualPairDataset, contrastive_loss_fn, evaluate
from baselines import generate_random_augmentation


class SimpleNLIDataset(Dataset):
    """Dataset for individual NLI samples (not pairs)."""

    def __init__(self, samples: list, tokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        for s in samples:
            label = LABEL_MAP.get(s.get("label", ""), -1)
            if label == -1:
                continue
            self.samples.append({**s, "label_id": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["premise"], s["hypothesis"],
            truncation=True, padding="max_length", max_length=self.max_seq_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": s["label_id"],
        }


def finetune_simple(config_path: str, samples: list, model_dir: str, output_dir: str):
    """Finetune with simple CE loss on individual samples (for random augmentation baseline)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = cfg["boosting"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    aug_dataset = SimpleNLIDataset(samples, tokenizer, cfg["baseline"]["max_seq_length"])
    print(f"  Loaded {len(aug_dataset)} samples for finetuning")

    if len(aug_dataset) == 0:
        return None

    aug_loader = DataLoader(aug_dataset, batch_size=bcfg["finetune_batch_size"], shuffle=True)

    # Original data for mixing
    orig_ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
    n_orig = int(len(aug_dataset) * bcfg["original_data_ratio"] / (1 - bcfg["original_data_ratio"]))
    n_orig = min(n_orig, len(orig_ds["train"]))
    orig_subset = orig_ds["train"].select(range(n_orig))
    orig_loader = DataLoader(orig_subset, batch_size=bcfg["finetune_batch_size"], shuffle=True)

    # Eval sets
    snli_ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
    test_loader = DataLoader(snli_ds["test"], batch_size=32)
    hans_ds = load_hans(tokenizer, cfg["baseline"]["max_seq_length"])
    hans_loader = DataLoader(hans_ds, batch_size=32)

    pre_test = evaluate(model, test_loader, device)
    pre_hans = evaluate(model, hans_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(bcfg["finetune_learning_rate"]), weight_decay=0.01)
    total_steps = (len(aug_loader) + len(orig_loader)) * bcfg["finetune_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    for epoch in range(bcfg["finetune_epochs"]):
        model.train()
        for batch in aug_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
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
        "num_samples": len(aug_dataset),
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


def finetune_with_pairs(config_path: str, pairs: list, model_dir: str, output_dir: str):
    """Finetune with CE + contrastive loss on CF pairs (for informed method)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = cfg["boosting"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    cf_dataset = CounterfactualPairDataset(pairs, tokenizer, cfg["baseline"]["max_seq_length"])
    print(f"  Loaded {len(cf_dataset)} pairs for finetuning")

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
    lam = float(bcfg["contrastive_lambda"])
    total_steps = (len(cf_loader) + len(orig_loader)) * bcfg["finetune_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    for epoch in range(bcfg["finetune_epochs"]):
        model.train()
        for batch in cf_loader:
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            anchor_labels = batch["anchor_label"].to(device)
            cf_ids = batch["cf_input_ids"].to(device)
            cf_mask = batch["cf_attention_mask"].to(device)
            cf_labels = batch["cf_label"].to(device)

            anchor_out = model(input_ids=anchor_ids, attention_mask=anchor_mask, labels=anchor_labels)
            cf_out = model(input_ids=cf_ids, attention_mask=cf_mask, labels=cf_labels)
            ce_loss = (anchor_out.loss + cf_out.loss) / 2
            cl_loss = contrastive_loss_fn(anchor_out.logits, cf_out.logits, anchor_labels, cf_labels)

            loss = ce_loss + lam * cl_loss
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


def run_comparison(config_path: str):
    """Run comparison at different sample counts."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    baseline_dir = os.path.join(cfg["output_dir"], "baseline")
    sample_counts = [50, 100, 200, 400]

    # --- Method 1: Random augmentation ---
    print("\n=== RANDOM AUGMENTATION ===")
    random_samples_path = os.path.join(cfg["output_dir"], "baselines", "random_augmentation", "pairs_valid.json")

    # Load raw samples (not pair format)
    raw_samples_path = os.path.join(cfg["output_dir"], "baselines", "random_augmentation", "raw_samples.json")
    if os.path.exists(raw_samples_path):
        with open(raw_samples_path) as f:
            random_samples = json.load(f)
        print(f"Loaded {len(random_samples)} existing random samples")
    else:
        random_samples = generate_random_augmentation(config_path, total_samples=max(sample_counts))
        # Save raw samples separately
        os.makedirs(os.path.dirname(raw_samples_path), exist_ok=True)
        # Extract raw format from pair format
        raw = []
        for p in random_samples:
            raw.append({
                "premise": p.get("anchor_premise", p.get("premise", "")),
                "hypothesis": p.get("anchor_hypothesis", p.get("hypothesis", "")),
                "label": p.get("anchor_label", p.get("label", "")),
            })
        with open(raw_samples_path, "w") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        random_samples = raw

    random_results = {}
    for n in sample_counts:
        if n > len(random_samples):
            print(f"\n--- Random with {n} samples: only {len(random_samples)} available, skipping ---")
            continue
        print(f"\n--- Random augmentation with {n} samples ---")
        subset = random_samples[:n]
        out_dir = os.path.join(cfg["output_dir"], "comparison", f"random_{n}")
        results = finetune_simple(config_path, subset, baseline_dir, out_dir)
        if results:
            random_results[n] = results

    # --- Method 2: Informed (ours) ---
    print("\n=== INFORMED (OURS) ===")
    informed_pairs_all = []
    for i in range(1, 20):
        path = os.path.join(cfg["output_dir"], f"iteration_{i}", "pairs_valid.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            pairs = json.load(f)
        informed_pairs_all.extend(pairs)
    print(f"Loaded {len(informed_pairs_all)} informed pairs from previous iterations")

    informed_results = {}
    for n in sample_counts:
        if n > len(informed_pairs_all):
            print(f"\n--- Informed with {n} samples: not enough pairs ({len(informed_pairs_all)} available), skipping ---")
            continue
        print(f"\n--- Informed with {n} samples ---")
        subset = informed_pairs_all[:n]
        out_dir = os.path.join(cfg["output_dir"], "comparison", f"informed_{n}")
        results = finetune_with_pairs(config_path, subset, baseline_dir, out_dir)
        if results:
            informed_results[n] = results

    # --- Summary ---
    print("\n" + "=" * 70)
    print("COST-PERFORMANCE COMPARISON")
    print("=" * 70)

    baseline_test = baseline_hans = None
    for d in [random_results, informed_results]:
        for v in d.values():
            baseline_test = v["pre_test_acc"]
            baseline_hans = v["pre_hans_acc"]
            break
        if baseline_test:
            break

    print(f"\nBaseline: test={baseline_test:.4f}, hans={baseline_hans:.4f}\n")
    print(f"{'Samples':>8} | {'Random':>20} | {'Informed (ours)':>20}")
    print(f"{'':>8} | {'Test':>9} {'HANS':>9} | {'Test':>9} {'HANS':>9}")
    print("-" * 60)

    for n in sample_counts:
        r_test = r_hans = i_test = i_hans = "    —"
        if n in random_results:
            r_test = f"{random_results[n]['post_test_acc']:.4f}"
            r_hans = f"{random_results[n]['post_hans_acc']:.4f}"
        if n in informed_results:
            i_test = f"{informed_results[n]['post_test_acc']:.4f}"
            i_hans = f"{informed_results[n]['post_hans_acc']:.4f}"
        print(f"{n:>8} | {r_test:>9} {r_hans:>9} | {i_test:>9} {i_hans:>9}")

    # Save comparison
    comparison = {
        "baseline_test_acc": baseline_test,
        "baseline_hans_acc": baseline_hans,
        "random": {str(k): v for k, v in random_results.items()},
        "informed": {str(k): v for k, v in informed_results.items()},
    }
    out_path = os.path.join(cfg["output_dir"], "comparison", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_comparison(args.config)
