"""Step 0: Train baseline BERT encoder on SNLI."""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data import load_snli, load_hans, LABEL_NAMES


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


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["student_model"], num_labels=cfg["num_labels"]
    ).to(device)

    # Load data
    bcfg = cfg["baseline"]
    print("Loading SNLI...")
    ds = load_snli(tokenizer, bcfg["max_seq_length"], cfg.get("max_train_samples"))

    train_loader = DataLoader(ds["train"], batch_size=bcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=bcfg["batch_size"])
    test_loader = DataLoader(ds["test"], batch_size=bcfg["batch_size"])

    # Load HANS for OOD evaluation
    print("Loading HANS...")
    hans_ds = load_hans(tokenizer, bcfg["max_seq_length"])
    hans_loader = DataLoader(hans_ds, batch_size=bcfg["batch_size"])

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(bcfg["learning_rate"]),
        weight_decay=float(bcfg["weight_decay"]),
    )
    total_steps = len(train_loader) * bcfg["num_epochs"]
    warmup_steps = int(total_steps * bcfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_val_acc = 0.0
    output_dir = os.path.join(cfg["output_dir"], "baseline")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(bcfg["num_epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{bcfg['num_epochs']}")

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

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        val_acc = evaluate(model, val_loader, device)
        hans_acc = evaluate(model, hans_loader, device)

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} | val_acc={val_acc:.4f} | hans_acc={hans_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    # Final test evaluation
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_acc = evaluate(model, test_loader, device)
    hans_acc = evaluate(model, hans_loader, device)
    print(f"\nFinal: test_acc={test_acc:.4f} | hans_acc={hans_acc:.4f}")

    # Save results
    results = {
        "test_acc": test_acc,
        "hans_acc": hans_acc,
        "best_val_acc": best_val_acc,
    }
    with open(os.path.join(output_dir, "results.yaml"), "w") as f:
        yaml.dump(results, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
