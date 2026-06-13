"""
Format raw_advocate.jsonl into chat-template JSONL for mlx-lm LoRA training.

Run from repo root:
    python foundry/devils-advocate/prepare_data.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoTokenizer

from shared.data_utils import train_val_split, write_jsonl

MODEL_ID = "mlx-community/gemma-2-9b-it-4bit"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_FILE = os.path.join(DATA_DIR, "raw_advocate.jsonl")

SYSTEM_PROMPT = (
    "You are a sophisticated Devil's Advocate. Your purpose is to intelligently "
    "challenge the user's premise. Do not agree. Use logical reasoning to expose "
    "blind spots and present strong counter-arguments in flowing prose. "
    "Be intellectual, provocative, and structured."
)


def format_example(ex: dict, tokenizer) -> str:
    # Gemma-2 doesn't support system role in many implementations/templates.
    # We merge it into the first user message.
    full_user_content = f"{SYSTEM_PROMPT}\n\nPremise: {ex['instruction']}"
    messages = [
        {"role": "user", "content": full_user_content},
        {"role": "assistant", "content": ex["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    if not os.path.exists(RAW_FILE):
        print(f"ERROR: {RAW_FILE} not found.")
        print("Run generate_dataset.py first.")
        sys.exit(1)

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    with open(RAW_FILE) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} raw examples")

    train_raw, valid_raw = train_val_split(examples, train_ratio=0.9)
    for split, rows in [("train", train_raw), ("valid", valid_raw)]:
        records = [{"text": format_example(ex, tokenizer)} for ex in rows]
        write_jsonl(records, os.path.join(DATA_DIR, f"{split}.jsonl"))

    print("\nSample (first training example):")
    with open(os.path.join(DATA_DIR, "train.jsonl")) as f:
        print(json.loads(f.readline())["text"][:600], "...")
    print(f"\nNext: python foundry/devils-advocate/train.py")


if __name__ == "__main__":
    main()
