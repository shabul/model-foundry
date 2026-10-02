"""Create held-out renderings of reserved synthetic test groups, after selection."""

import json
import random
from pathlib import Path

from decision_encoder.data.schema import read_jsonl, write_jsonl


def main():
    if not Path("reports/selection.json").exists():
        raise RuntimeError("Freeze checkpoint selection before constructing final stress evaluation")
    rows = [r for r in read_jsonl("data/processed/test.jsonl") if r["source"] == "synthetic"]
    random.Random(725).shuffle(rows)
    selected = []
    seen = set()
    for row in rows:
        if row["group_id"] in seen:
            continue
        seen.add(row["group_id"])
        fields = row["metadata"]["structured_state"]
        evidence = "; ".join(
            f"{key.replace('_', ' ')}: {json.dumps(value)}" for key, value in sorted(fields.items())
        )
        row["state"] = (
            f"Operations audit excerpt. Applicable rule: {row['metadata']['rule']}\nEvidence recorded by the operator: {evidence}.\nThe record does not specify any other relevant facts."
        )
        row["id"] += "-heldout-rendering"
        row["metadata"]["template"] = "heldout-audit-excerpt"
        row["metadata"]["evaluation_claim"] = "surface-form robustness on reserved causal-state groups"
        selected.append(row)
        if len(selected) >= 500:
            break
    write_jsonl("data/processed/stress.jsonl", selected)
    print(f"Created {len(selected)} distinct synthetic test-group renderings")


if __name__ == "__main__":
    main()
