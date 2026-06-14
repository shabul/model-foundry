"""
Combine scraped + synthetic data, apply Mistral chat template, save to data/sft/.

Run from repo root:
    python foundry/desi-finance-advisor/phase2_sft/prepare_sft_data.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from shared.data_utils import train_val_split, write_jsonl

from transformers import AutoTokenizer

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
SFT_DIR = os.path.join(DATA_DIR, "sft")

SYSTEM_PROMPT = (
    "You are Desi Finance Bhai — a warm, knowledgeable Indian personal finance advisor. "
    "Respond in simple English with occasional Hindi words like bhai, yaar, seedha, matlab. "
    "Be factually accurate, mention caveats like 'consult a CA' where needed, "
    "and keep answers conversational and concise."
)


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  Skipping (not found): {path}")
        return []
    with open(path) as f:
        data = [json.loads(line) for line in f]
    print(f"  Loaded {len(data):5d} examples ← {path}")
    return data


def format_example(ex: dict, tokenizer) -> str:
    # Mistral-7B-Instruct doesn't support system role — prepend to user message
    user = f"{SYSTEM_PROMPT}\n\n{ex['instruction'].strip()}"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["response"].strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\nLoading data sources:")
    scraped = load_jsonl(os.path.join(CLEANED_DIR, "scraped_pairs.jsonl"))
    synthetic = load_jsonl(os.path.join(CLEANED_DIR, "synthetic.jsonl"))

    all_examples = scraped + synthetic
    print(f"\n  Total: {len(all_examples)} examples")

    if not all_examples:
        print("ERROR: No data found. Run Phase 1 scripts first.")
        sys.exit(1)

    train_raw, valid_raw = train_val_split(all_examples, train_ratio=0.9)

    os.makedirs(SFT_DIR, exist_ok=True)
    for split, rows in [("train", train_raw), ("valid", valid_raw)]:
        records = [{"text": format_example(ex, tokenizer)} for ex in rows]
        write_jsonl(records, os.path.join(SFT_DIR, f"{split}.jsonl"))

    print(f"\nSFT data ready: {len(train_raw)} train / {len(valid_raw)} val")
    print("\nSample (first 400 chars of first training example):")
    with open(os.path.join(SFT_DIR, "train.jsonl")) as f:
        print(json.loads(f.readline())["text"][:400], "...")
    print("\nNext: python foundry/desi-finance-advisor/phase2_sft/train_sft.py")


if __name__ == "__main__":
    main()
