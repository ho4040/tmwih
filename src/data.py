"""Dataset loading and preprocessing for SNLI + OOD evaluation sets."""

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


def load_hans(tokenizer: PreTrainedTokenizer, max_seq_length: int = 128):
    """Load HANS evaluation set (OOD diagnostic for NLI)."""
    try:
        ds = load_dataset("jhu-cogsci/hans", split="validation", trust_remote_code=True)
    except Exception:
        ds = load_dataset("hans", split="validation", trust_remote_code=True)

    # HANS has only entailment (0) and non-entailment (1)
    # Map non-entailment to contradiction (2) for 3-class compatibility
    def map_labels(example):
        example["label"] = 0 if example["label"] == 0 else 2
        return example

    ds = ds.map(map_labels)

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    # Remove all columns except those added by tokenization + label
    cols_to_remove = [c for c in ds.column_names if c not in ["label", "input_ids", "attention_mask", "token_type_ids"]]
    ds = ds.map(tokenize, batched=True, remove_columns=cols_to_remove)
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


def decode_pair(tokenizer: PreTrainedTokenizer, input_ids) -> tuple[str, str]:
    """Decode a tokenized premise-hypothesis pair back to text."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    # For BERT with [SEP] separator, split on [SEP]
    parts = text.split(tokenizer.sep_token)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return text, ""
