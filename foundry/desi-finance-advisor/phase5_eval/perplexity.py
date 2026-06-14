"""
Compare SFT vs DPO model perplexity on the validation set.
Lower perplexity = better language quality.

Run from repo root:
    python foundry/desi-finance-advisor/phase5_eval/perplexity.py
"""

import json
import math
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
SFT_ADAPTER = os.path.join(os.path.dirname(__file__), "..", "adapters", "sft")
DPO_ADAPTER = os.path.join(os.path.dirname(__file__), "..", "adapters", "dpo")
VALID_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sft", "valid.jsonl")

MAX_EXAMPLES = 100
MAX_TOKENS_PER_EXAMPLE = 512


def compute_perplexity(model, tokenizer, examples: list[dict]) -> float:
    total_loss = 0.0
    total_tokens = 0

    for ex in examples:
        tokens = tokenizer.encode(ex["text"])
        if len(tokens) > MAX_TOKENS_PER_EXAMPLE:
            tokens = tokens[:MAX_TOKENS_PER_EXAMPLE]
        if len(tokens) < 4:
            continue

        input_ids = mx.array(tokens[:-1])[None]   # [1, T-1]
        targets = mx.array(tokens[1:])             # [T-1]

        logits = model(input_ids)                  # [1, T-1, vocab]
        logits = logits[0]                         # [T-1, vocab]

        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_tokens += len(targets)

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def main():
    if not os.path.exists(VALID_FILE):
        print(f"ERROR: {VALID_FILE} not found.")
        sys.exit(1)

    with open(VALID_FILE) as f:
        examples = [json.loads(line) for line in f][:MAX_EXAMPLES]
    print(f"Validation examples: {len(examples)}\n")

    results = {}
    for name, adapter in [("SFT", SFT_ADAPTER), ("DPO", DPO_ADAPTER)]:
        if not os.path.isdir(adapter):
            print(f"Skipping {name}: adapter not found at {adapter}")
            continue
        print(f"Loading {name} model ({adapter})...")
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
        ppl = compute_perplexity(model, tokenizer, examples)
        results[name] = ppl
        print(f"  {name} perplexity: {ppl:.3f}")
        del model

    print(f"\n{'='*40}")
    for name, ppl in results.items():
        print(f"  {name:5s}  ppl = {ppl:.3f}")

    if "SFT" in results and "DPO" in results:
        delta = results["DPO"] - results["SFT"]
        sign = "↑ worse" if delta > 0 else "↓ better"
        print(f"\n  DPO vs SFT: {delta:+.3f}  ({sign})")
        if delta < 2.0:
            print("  DPO did not significantly hurt language quality.")

    out = os.path.join(os.path.dirname(__file__), "perplexity_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
