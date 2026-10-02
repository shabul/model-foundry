"""Audit source balance and token budgets without running the model."""

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from decision_encoder.data.schema import read_jsonl
from decision_encoder.serialization import serialize


def main():
    tok = AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base", revision="8949b909ec900327062f0ebf497f51aef5e6f0c8"
    )
    report = {}
    for split in ["train", "validation", "calibration"]:
        rows = read_jsonl(f"data/processed/{split}.jsonl")
        counts = Counter()
        overflows = Counter()
        impossible = []
        abstention = Counter()
        for row in rows:
            counts[row["source"]] += 1
            try:
                encoded = serialize(row, tok, 512, "truncate_state")
                overflows[row["source"]] += int(encoded["truncated"])
            except ValueError:
                impossible.append(row["id"])
            for i, o in enumerate(row["options"]):
                if o.get("is_abstain"):
                    abstention["available"] += 1
                    abstention["positive"] += int(row["target_probabilities"][i] > 0.5)
        report[split] = {
            "rows": len(rows),
            "sources": dict(counts),
            "state_truncated_at_512": dict(overflows),
            "question_options_overflow": impossible,
            "abstention": dict(abstention),
        }
        print(split, report[split], flush=True)
    Path("reports/dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
