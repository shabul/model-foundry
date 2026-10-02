"""Canonical data contract. Validation is shared by builders and training."""

import hashlib
import json
import math
from pathlib import Path

ABSTAIN_ID = "__abstain__"
ABSTAIN_OPTION = {
    "id": ABSTAIN_ID,
    "label": "None of these options",
    "description": "None of the other options is sufficiently supported by the available evidence.",
    "is_abstain": True,
}
TRAIN_SOURCES = {"banking77", "massive", "typed_decisions", "synthetic"}


def validate_options(options):
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("At least two options required")
    ids = []
    for o in options:
        for field in ["id", "label", "description"]:
            if not isinstance(o.get(field), str) or not o[field].strip():
                raise ValueError(f"Option {field} must be a nonempty string")
        if (o["id"] == ABSTAIN_ID) != bool(o.get("is_abstain", False)):
            raise ValueError("Reserved abstention ID and is_abstain must agree")
        if "value" in o and (
            isinstance(o["value"], bool)
            or not isinstance(o["value"], (float, int))
            or not math.isfinite(o["value"])
        ):
            raise ValueError("Ordinal values must be finite numbers")
        ids.append(o["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")


def validate_record(r, require_target=True):
    for field in ["state", "question"]:
        if not isinstance(r.get(field), str) or not r[field].strip():
            raise ValueError(f"{field} must be nonempty text")
    validate_options(r["options"])
    if not require_target:
        return r
    for field in ["id", "source", "source_split", "group_id"]:
        if not isinstance(r.get(field), str) or not r[field]:
            raise ValueError(f"Missing provenance: {field}")
    if r.get("decision_type") not in {"choice", "boolean", "ordinal"}:
        raise ValueError("Unknown decision type")
    if r["decision_type"] == "ordinal" and any("value" not in o for o in r["options"]):
        raise ValueError("Ordinal candidates require numeric values")
    y = r.get("target_probabilities")
    if not isinstance(y, list) or len(y) != len(r["options"]):
        raise ValueError("Target count must match candidate count")
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in y
    ):
        raise ValueError("Targets must be finite nonnegative numbers")
    if not math.isclose(sum(y), 1.0, abs_tol=1e-6):
        raise ValueError("Target probabilities must sum to one")
    return r


def state_hash(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def split_group(group, seed=42):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16) / 16**12
    return "train" if value < 0.8 else "validation" if value < 0.9 else "calibration"


def read_jsonl(path):
    with open(path) as f:
        return [validate_record(json.loads(line)) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            validate_record(row)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assert_no_leakage(splits):
    seen_groups, seen_states, seen_ids = {}, {}, set()
    for split, rows in splits.items():
        for row in rows:
            validate_record(row)
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate record ID: {row['id']}")
            seen_ids.add(row["id"])
            if split in {"train", "validation", "calibration"}:
                if row["source"] not in TRAIN_SOURCES or row["source_split"] == "test":
                    raise ValueError("Forbidden training/development source or split")
            key = (row["source"], row["group_id"])
            h = state_hash(row["state"])
            for k, seen in [(key, seen_groups), (h, seen_states)]:
                if k in seen and seen[k] != split:
                    raise ValueError(f"Cross-split leakage: {row['id']}")
                seen[k] = split
