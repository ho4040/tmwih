"""Dataset loading and preprocessing for SNLI, FEVER + OOD evaluation sets."""

import json
import os
import urllib.request

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def load_snli(tokenizer: PreTrainedTokenizer, max_seq_length: int = 128, max_samples: int | None = None):
    """Load and tokenize SNLI dataset."""
    ds = load_dataset("stanfordnlp/snli")

    # Filter out samples with label -1 (no gold label)
    ds = ds.filter(lambda x: x["label"] != -1)

    if max_samples:
        for split in ds:
            if len(ds[split]) > max_samples:
                ds[split] = ds[split].select(range(max_samples))

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])
    ds.set_format("torch")
    return ds


def load_hans(tokenizer: PreTrainedTokenizer, max_seq_length: int = 128, data_dir: str = "data"):
    """Load HANS evaluation set (OOD diagnostic for NLI) from raw TSV."""
    hans_path = os.path.join(data_dir, "hans.tsv")
    if not os.path.exists(hans_path):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
        print(f"Downloading HANS from {url}...")
        urllib.request.urlretrieve(url, hans_path)

    df = pd.read_csv(hans_path, sep="\t")
    # Map labels: entailment -> 0, non-entailment -> 2 (contradiction)
    label_map = {"entailment": 0, "non-entailment": 2}
    df["label"] = df["gold_label"].map(label_map)
    df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})

    ds = Dataset.from_pandas(df[["premise", "hypothesis", "label"]])

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])
    ds.set_format("torch")
    return ds


def load_kaushik_cad(tokenizer: PreTrainedTokenizer, max_seq_length: int = 128):
    """Load Kaushik et al. counterfactually augmented NLI data (test split)."""
    # The CAD dataset is available from the original paper's GitHub
    # For now, we'll download and cache it
    try:
        ds = load_dataset("tomekkorbak/counterfactually-augmented-snli", split="test")
    except Exception:
        # Fallback: try alternative source
        print("Warning: Could not load Kaushik CAD dataset. Skipping.")
        return None

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    ds = ds.map(tokenize, batched=True)
    ds.set_format("torch")
    return ds


FEVER_LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 2, "NOT ENOUGH INFO": 1}
FEVER_LABEL_NAMES = ["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]


def load_fever(tokenizer: PreTrainedTokenizer, max_seq_length: int = 128):
    """Load and tokenize FEVER dataset for fact verification."""
    ds = load_dataset("fever/fever", "v1.0")

    # Filter to only verifiable claims with labels
    def has_valid_label(x):
        return x.get("label", "") in FEVER_LABEL_MAP

    ds = ds.filter(has_valid_label)

    # Map labels to integers
    def map_labels(batch):
        batch["label"] = [FEVER_LABEL_MAP[l] for l in batch["label"]]
        return batch

    ds = ds.map(map_labels, batched=True)

    def tokenize(batch):
        # FEVER: claim is the hypothesis, evidence is the premise
        premises = [e if isinstance(e, str) else str(e) for e in batch.get("evidence", batch.get("claim", []))]
        hypotheses = batch["claim"] if "evidence" in batch else [""] * len(premises)
        return tokenizer(
            premises, hypotheses,
            truncation=True, padding="max_length", max_length=max_seq_length,
        )

    cols_to_remove = [c for c in ds["train"].column_names if c not in ["label"]]
    ds = ds.map(tokenize, batched=True, remove_columns=cols_to_remove)
    ds.set_format("torch")
    return ds


def load_fever_symmetric(tokenizer: PreTrainedTokenizer, max_seq_length: int = 128, data_dir: str = "data"):
    """Load FEVER Symmetric evaluation set (OOD for fact verification)."""
    sym_path = os.path.join(data_dir, "fever_symmetric.jsonl")
    if not os.path.exists(sym_path):
        os.makedirs(data_dir, exist_ok=True)
        # Try downloading from GitHub
        urls = [
            "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl",
            "https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2/fever_symmetric_dev.jsonl",
        ]
        downloaded = False
        for url in urls:
            try:
                print(f"Downloading FEVER Symmetric from {url}...")
                urllib.request.urlretrieve(url, sym_path)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if not downloaded:
            print("Warning: Could not download FEVER Symmetric. Skipping.")
            return None

    # Parse JSONL
    records = []
    with open(sym_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            label = obj.get("label", obj.get("gold_label", ""))
            if label in FEVER_LABEL_MAP:
                records.append({
                    "premise": obj.get("evidence", obj.get("evidence_sentence", "")),
                    "hypothesis": obj.get("claim", ""),
                    "label": FEVER_LABEL_MAP[label],
                })

    if not records:
        print("Warning: No valid FEVER Symmetric records found")
        return None

    ds = Dataset.from_list(records)

    def tokenize(batch):
        return tokenizer(
            batch["premise"], batch["hypothesis"],
            truncation=True, padding="max_length", max_length=max_seq_length,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])
    ds.set_format("torch")
    print(f"Loaded {len(ds)} FEVER Symmetric examples")
    return ds


def decode_pair(tokenizer: PreTrainedTokenizer, input_ids) -> tuple[str, str]:
    """Decode a tokenized premise-hypothesis pair back to text."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    # For BERT with [SEP] separator, split on [SEP]
    parts = text.split(tokenizer.sep_token)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return text, ""
