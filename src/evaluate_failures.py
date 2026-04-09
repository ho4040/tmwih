"""Step 1: Evaluate converged model and collect failure cases with attribution."""

import argparse
import json
import os

import torch
import yaml
from captum.attr import LayerIntegratedGradients
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from data import load_snli, decode_pair, LABEL_NAMES


def collect_failures(config_path: str, model_dir: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bcfg = cfg["boosting"]

    if model_dir is None:
        model_dir = os.path.join(cfg["output_dir"], "baseline")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Setup Integrated Gradients
    def forward_func(input_embeds, attention_mask, token_type_ids=None):
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings.word_embeddings)

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
        # attentions shape: (num_layers, batch, num_heads, seq_len, seq_len)
        last_attn = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
        cls_attn = last_attn[0, :, 0, :].mean(dim=0)  # (seq_len,) — avg over heads

        # Compute Integrated Gradients
        token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
        attributions = lig.attribute(
            input_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            target=label,
            n_steps=bcfg["ig_steps"],
            return_convergence_delta=False,
        )
        # Sum over embedding dim to get per-token attribution
        ig_attr = attributions.sum(dim=-1).squeeze(0)  # (seq_len,)
        ig_attr = ig_attr.abs()  # use absolute values
        ig_attr = ig_attr / (ig_attr.sum() + 1e-10)  # normalize

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
                "gradient": round(ig_attr[i].item(), 4),
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
