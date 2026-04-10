"""Step 4: Finetune with counterfactual minimal pairs (CE + Contrastive loss)."""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data import load_snli, load_hans, LABEL_MAP, LABEL_NAMES


class CounterfactualPairDataset(Dataset):
    """Dataset of counterfactual minimal pairs for contrastive learning."""

    def __init__(self, pairs: list, tokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pairs = []

        for p in pairs:
            anchor_label = LABEL_MAP.get(p.get("anchor_label", ""), -1)
            cf_label = LABEL_MAP.get(p.get("cf_label", ""), -1)
            if anchor_label == -1 or cf_label == -1 or anchor_label == cf_label:
                continue
            self.pairs.append(p)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        anchor = self.tokenizer(
            p["anchor_premise"], p["anchor_hypothesis"],
            truncation=True, padding="max_length", max_length=self.max_seq_length, return_tensors="pt",
        )
        cf = self.tokenizer(
            p["cf_premise"], p["cf_hypothesis"],
            truncation=True, padding="max_length", max_length=self.max_seq_length, return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
            "anchor_label": LABEL_MAP[p["anchor_label"]],
            "cf_input_ids": cf["input_ids"].squeeze(0),
            "cf_attention_mask": cf["attention_mask"].squeeze(0),
            "cf_label": LABEL_MAP[p["cf_label"]],
        }


def contrastive_loss_fn(anchor_logits, cf_logits, anchor_labels, cf_labels):
    """Supervised contrastive loss on classification head logit space.

    For each pair, the anchor and counterfactual have different labels.
    We want the model to increase the logit gap for the correct classes:
    - anchor should have high logit for anchor_label and low for cf_label
    - cf should have high logit for cf_label and low for anchor_label
    """
    batch_size = anchor_logits.size(0)
    loss = 0.0
    for i in range(batch_size):
        # Anchor: increase target logit, decrease the cf's target logit
        anchor_target = anchor_logits[i, anchor_labels[i]]
        anchor_confuse = anchor_logits[i, cf_labels[i]]
        # CF: increase target logit, decrease the anchor's target logit
        cf_target = cf_logits[i, cf_labels[i]]
        cf_confuse = cf_logits[i, anchor_labels[i]]
        # Margin ranking: target should be higher than confusing class by margin
        loss += F.relu(1.0 - (anchor_target - anchor_confuse))
        loss += F.relu(1.0 - (cf_target - cf_confuse))
    return loss / (2 * batch_size)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return accuracy_score(all_labels, all_preds)


def finetune(config_path: str, iteration: int = 1, model_dir: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = cfg["boosting"]

    # Determine model directory
    if model_dir is None:
        if iteration == 1:
            model_dir = os.path.join(cfg["output_dir"], "baseline")
        else:
            model_dir = os.path.join(cfg["output_dir"], f"boosted_iter_{iteration - 1}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    # Load counterfactual pairs
    pairs_path = os.path.join(cfg["output_dir"], f"iteration_{iteration}", "pairs_valid.json")
    with open(pairs_path) as f:
        pairs = json.load(f)

    cf_dataset = CounterfactualPairDataset(pairs, tokenizer, cfg["baseline"]["max_seq_length"])
    print(f"Loaded {len(cf_dataset)} valid counterfactual pairs")

    if len(cf_dataset) == 0:
        print("No valid pairs to finetune with. Skipping.")
        return None

    cf_loader = DataLoader(cf_dataset, batch_size=bcfg["finetune_batch_size"], shuffle=True)

    # Load original training data (subset for mixing)
    orig_ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
    n_orig = int(len(cf_dataset) * bcfg["original_data_ratio"] / (1 - bcfg["original_data_ratio"]))
    n_orig = min(n_orig, len(orig_ds["train"]))
    orig_subset = orig_ds["train"].select(range(n_orig))
    orig_loader = DataLoader(orig_subset, batch_size=bcfg["finetune_batch_size"], shuffle=True)

    # Load eval sets
    snli_ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"])
    test_loader = DataLoader(snli_ds["test"], batch_size=32)
    hans_ds = load_hans(tokenizer, cfg["baseline"]["max_seq_length"])
    hans_loader = DataLoader(hans_ds, batch_size=32)

    # Pre-boosting eval
    pre_test_acc = evaluate(model, test_loader, device)
    pre_hans_acc = evaluate(model, hans_loader, device)
    print(f"Pre-boosting: test_acc={pre_test_acc:.4f} | hans_acc={pre_hans_acc:.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(bcfg["finetune_learning_rate"]), weight_decay=0.01)
    total_steps = (len(cf_loader) + len(orig_loader)) * bcfg["finetune_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    lam = bcfg["contrastive_lambda"]

    # Training loop
    for epoch in range(bcfg["finetune_epochs"]):
        model.train()
        total_ce_loss = 0
        total_cl_loss = 0
        steps = 0

        # Contrastive training on CF pairs
        pbar = tqdm(cf_loader, desc=f"Boost epoch {epoch + 1}/{bcfg['finetune_epochs']} [CF pairs]")
        for batch in pbar:
            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            anchor_labels = batch["anchor_label"].to(device)
            cf_ids = batch["cf_input_ids"].to(device)
            cf_mask = batch["cf_attention_mask"].to(device)
            cf_labels = batch["cf_label"].to(device)

            # CE loss on both anchor and counterfactual
            anchor_out = model(input_ids=anchor_ids, attention_mask=anchor_mask, labels=anchor_labels)
            cf_out = model(input_ids=cf_ids, attention_mask=cf_mask, labels=cf_labels)
            ce_loss = (anchor_out.loss + cf_out.loss) / 2

            # Contrastive loss on logits
            cl_loss = contrastive_loss_fn(anchor_out.logits, cf_out.logits, anchor_labels, cf_labels)

            loss = ce_loss + lam * cl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_ce_loss += ce_loss.item()
            total_cl_loss += cl_loss.item()
            steps += 1
            pbar.set_postfix(ce=f"{ce_loss.item():.4f}", cl=f"{cl_loss.item():.4f}")

        # CE training on original data (prevent catastrophic forgetting)
        pbar = tqdm(orig_loader, desc=f"Boost epoch {epoch + 1}/{bcfg['finetune_epochs']} [Original]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_ce_loss += loss.item()
            steps += 1

        avg_ce = total_ce_loss / steps
        avg_cl = total_cl_loss / max(1, len(cf_loader))
        test_acc = evaluate(model, test_loader, device)
        hans_acc = evaluate(model, hans_loader, device)
        print(f"Epoch {epoch + 1}: ce={avg_ce:.4f} cl={avg_cl:.4f} | test={test_acc:.4f} hans={hans_acc:.4f}")

    # Post-boosting eval
    post_test_acc = evaluate(model, test_loader, device)
    post_hans_acc = evaluate(model, hans_loader, device)
    print(f"\nPost-boosting: test_acc={post_test_acc:.4f} | hans_acc={post_hans_acc:.4f}")
    print(f"Improvement: test={post_test_acc - pre_test_acc:+.4f} | hans={post_hans_acc - pre_hans_acc:+.4f}")

    # Save boosted model
    output_dir = os.path.join(cfg["output_dir"], f"boosted_iter_{iteration}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {
        "iteration": iteration,
        "num_cf_pairs": len(cf_dataset),
        "pre_test_acc": pre_test_acc,
        "pre_hans_acc": pre_hans_acc,
        "post_test_acc": post_test_acc,
        "post_hans_acc": post_hans_acc,
        "test_improvement": post_test_acc - pre_test_acc,
        "hans_improvement": post_hans_acc - pre_hans_acc,
    }
    with open(os.path.join(output_dir, "results.yaml"), "w") as f:
        yaml.dump(results, f)

    print(f"Saved to {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--model_dir", default=None)
    args = parser.parse_args()
    finetune(args.config, args.iteration, args.model_dir)
