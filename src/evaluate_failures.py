"""Step 1: Evaluate converged model and collect failure cases with attribution."""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from data import load_snli, decode_pair, LABEL_NAMES


def compute_gradient_attribution(model, input_ids, attention_mask, target_label):
    """Compute gradient-based attribution (input x gradient, normalized)."""
    model.zero_grad()
    embeddings = model.bert.embeddings.word_embeddings(input_ids)
    embeddings.retain_grad()

    # Forward through the rest of the model
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits

    # Backward for the target class
    target_score = logits[0, target_label]
    target_score.backward()

    # Input x Gradient attribution
    grad = embeddings.grad  # (1, seq_len, hidden_dim)
    attr = (embeddings * grad).sum(dim=-1).squeeze(0)  # (seq_len,)
    attr = attr.abs()
    attr = attr / (attr.sum() + 1e-10)
    return attr.detach()


def collect_failures(config_path: str, model_dir: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = cfg["boosting"]

    if model_dir is None:
        model_dir = os.path.join(cfg["output_dir"], "baseline")

    # Load model and tokenizer (use eager attention for output_attentions support)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    ).to(device)
    model.eval()

    # Load validation set
    ds = load_snli(tokenizer, cfg["baseline"]["max_seq_length"], cfg.get("max_val_samples"))
    val_loader = DataLoader(ds["validation"], batch_size=1, shuffle=False)

    failures = []
    total_correct = 0
    total_seen = 0

    print(f"Collecting failures (max {bcfg['max_failures']})...")
    for batch in tqdm(val_loader):
        if len(failures) >= bcfg["max_failures"]:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].item()
        total_seen += 1

        # Forward pass with attention
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            pred = outputs.logits.argmax(dim=-1).item()
            confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        if pred == label:
            total_correct += 1
            continue

        # Extract attention (last layer, [CLS] token, averaged over heads)
        last_attn = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
        cls_attn = last_attn[0, :, 0, :].mean(dim=0)  # (seq_len,)

        # Compute gradient attribution (input x gradient)
        grad_attr = compute_gradient_attribution(model, input_ids, attention_mask, label)

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        premise, hypothesis = decode_pair(tokenizer, input_ids[0].cpu())

        # Build token-level attribution dict (skip special tokens and padding)
        token_info = []
        for i, tok in enumerate(tokens):
            if tok in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                continue
            token_info.append({
                "token": tok,
                "attention": round(cls_attn[i].item(), 4),
                "gradient": round(grad_attr[i].item(), 4),
            })

        failures.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": LABEL_NAMES[label],
            "predicted_label": LABEL_NAMES[pred],
            "confidence": round(confidence, 4),
            "token_attributions": token_info,
        })

    val_acc = total_correct / total_seen if total_seen > 0 else 0
    print(f"Val accuracy: {val_acc:.4f} ({total_correct}/{total_seen})")
    print(f"Collected {len(failures)} failure cases")

    # Save failures
    output_path = os.path.join(cfg["output_dir"], "failures.json")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"val_accuracy": val_acc, "failures": failures}, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_dir", default=None)
    args = parser.parse_args()
    collect_failures(args.config, args.model_dir)
