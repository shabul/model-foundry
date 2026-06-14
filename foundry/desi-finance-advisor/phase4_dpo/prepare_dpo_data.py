"""
Format DPO triplets for mlx-lm DPO training and split into train/valid.

mlx-lm DPO expects JSONL with keys: prompt, chosen, rejected.
The prompt should be the raw instruction text (mlx-lm applies the template).

Run from repo root:
    python foundry/desi-finance-advisor/phase4_dpo/prepare_dpo_data.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from shared.data_utils import train_val_split, write_jsonl

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IN_FILE = os.path.join(DATA_DIR, "dpo", "triplets.jsonl")
OUT_DIR = os.path.join(DATA_DIR, "dpo")


def coerce_str(val) -> str:
    """Gemini occasionally returns a list of dicts instead of a plain string."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                parts.append(item.get("response") or item.get("text") or str(item))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(val).strip()


def extract_instruction(prompt: str) -> str:
    """Pull the bare user question out of a formatted Mistral prompt."""
    if "[INST]" in prompt:
        inner = prompt.split("[INST]", 1)[-1].split("[/INST]")[0].strip()
        # Drop system prompt prefix (separated by double newline)
        if "\n\n" in inner:
            inner = inner.split("\n\n", 1)[-1].strip()
        return inner
    return prompt.strip()


def main():
    if not os.path.exists(IN_FILE):
        print(f"ERROR: {IN_FILE} not found. Run phase3 scripts first.")
        sys.exit(1)

    with open(IN_FILE) as f:
        triplets = [json.loads(line) for line in f]
    print(f"Loaded {len(triplets)} DPO triplets")

    # Reformat: strip prompt back to raw instruction so mlx-lm can apply template
    formatted = []
    for t in triplets:
        chosen = coerce_str(t["chosen"])
        rejected = coerce_str(t["rejected"])
        if chosen and rejected and chosen != rejected:
            formatted.append({
                "prompt": extract_instruction(t["prompt"]),
                "chosen": chosen,
                "rejected": rejected,
            })
    print(f"Valid triplets   : {len(formatted)}")

    train, valid = train_val_split(formatted, train_ratio=0.9)
    write_jsonl(train, os.path.join(OUT_DIR, "train.jsonl"))
    write_jsonl(valid, os.path.join(OUT_DIR, "valid.jsonl"))

    print(f"\nDPO data ready: {len(train)} train / {len(valid)} val")
    print("Next: python foundry/desi-finance-advisor/phase4_dpo/train_dpo.py")


if __name__ == "__main__":
    main()
