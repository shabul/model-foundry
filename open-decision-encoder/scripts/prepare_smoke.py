"""Rebuild the fixed 128-example deliberate-overfit fixture."""

from decision_encoder.data.schema import write_jsonl
from decision_encoder.data.synthetic import generate_synthetic

rows = []
for row in generate_synthetic(1000):
    if row["decision_type"] == "choice" and row["metadata"]["abstention_reason"] is None:
        rows.append(row)
    if len(rows) == 128:
        break
write_jsonl("data/processed/smoke.jsonl", rows)
