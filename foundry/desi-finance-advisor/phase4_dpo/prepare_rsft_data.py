"""
Prepare Rejection-Sampling SFT (RSFT) data from DPO triplets.

mlx-lm 0.31.3 does not have a DPODataset class, so we pivot to RSFT:
train on only the Gemini-labelled *chosen* responses.  The data format
used is CompletionsDataset  { "prompt": ..., "completion": ... }  where
mlx-lm computes loss only on the completion tokens.

The triplets already store the prompt in full Mistral format:
    <s>[INST] SYSTEM\n\nQUESTION [/INST]
so we can use it verbatim as the prompt and append the chosen response
as the completion.

Run from repo root:
    python foundry/desi-finance-advisor/phase4_dpo/prepare_rsft_data.py
"""

import json
import os
import random
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IN_FILE  = os.path.join(DATA_DIR, "dpo", "triplets.jsonl")
OUT_DIR  = os.path.join(DATA_DIR, "rsft")

TRAIN_RATIO = 0.9
SEED = 42


def coerce_str(val) -> str:
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


def main():
    if not os.path.exists(IN_FILE):
        print(f"ERROR: {IN_FILE} not found. Run phase3 scripts first.")
        sys.exit(1)

    with open(IN_FILE) as f:
        triplets = [json.loads(line) for line in f]
    print(f"Loaded {len(triplets)} DPO triplets from triplets.jsonl")

    records = []
    for t in triplets:
        prompt  = t["prompt"]   # already full <s>[INST]...[/INST] format
        chosen  = coerce_str(t["chosen"])
        if not chosen:
            continue
        # CompletionsDataset concatenates prompt + completion; loss is on completion
        records.append({
            "prompt":     prompt,
            "completion": f" {chosen} </s>",
        })
    print(f"Valid records    : {len(records)}")

    random.seed(SEED)
    random.shuffle(records)
    split = max(1, int(len(records) * TRAIN_RATIO))
    train, valid = records[:split], records[split:]

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, subset in [("train", train), ("valid", valid)]:
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w") as f:
            for rec in subset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {name:5s}: {len(subset):3d} records → {path}")

    print(f"\nRSFT data ready: {len(train)} train / {len(valid)} val")
    print("Next: python foundry/desi-finance-advisor/phase4_dpo/train_rsft.py")


if __name__ == "__main__":
    main()
