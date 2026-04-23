"""
Download databricks/databricks-dolly-15k and write train/valid JSONL
for mlx-lm LoRA training.

Run from repo root: python foundry/qwen2.5-dolly/prepare_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_dataset
from transformers import AutoTokenizer

from shared.data_utils import format_chat_example, train_val_split, write_jsonl

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "databricks/databricks-dolly-15k"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")

    train_raw, valid_raw = train_val_split(list(dataset))

    for split, rows in [("train", train_raw), ("valid", valid_raw)]:
        records = [{"text": format_chat_example(ex, tokenizer)} for ex in rows]
        write_jsonl(records, os.path.join(DATA_DIR, f"{split}.jsonl"))

    print("\nSample (first training example):")
    import json
    with open(os.path.join(DATA_DIR, "train.jsonl")) as f:
        print(json.loads(f.readline())["text"][:500], "...")


if __name__ == "__main__":
    main()
