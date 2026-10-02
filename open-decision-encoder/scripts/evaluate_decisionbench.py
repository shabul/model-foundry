"""Final-only external evaluation. Never used by training or synthetic generation."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from decision_encoder.data.schema import validate_record
from decision_encoder.inference import DecisionPredictor
from decision_encoder.metrics import metrics
from decision_encoder.runtime import accelerator_lock, sha256_file
from decision_encoder.serialization import serialize

REPO = "Hanno-Labs/decision-bench"
REVISION = "071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb"


def transform(row):
    candidates = json.loads(row["candidates_json"])
    options = []
    for c in candidates:
        option = {
            "id": str(c["id"]),
            "label": str(c["label"]),
            "description": str(c.get("description") or c["label"]),
        }
        if c.get("ordinal_value") is not None:
            option["value"] = float(c["ordinal_value"])
        options.append(option)
    r = {
        "id": row["row_id"],
        "source": "decisionbench",
        "source_split": "eval",
        "group_id": row["row_id"],
        "state": row["state_json"],
        "question": row["instruction"],
        "options": options,
        "target_probabilities": list(row["gold_probabilities"]),
        "decision_type": {
            "binary_classification": "boolean",
            "candidate_selection": "choice",
            "ordinal_scoring": "ordinal",
        }[row["primitive"]],
        "metadata": {
            "task_id": row["task_id"],
            "domain": row["domain"],
            "reasoning_required": row["reasoning_required"],
        },
    }
    return validate_record(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", default="reports/selection.json")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    selection = json.loads(Path(a.selection).read_text())
    checkpoint = selection["checkpoint"]
    if sha256_file(Path(checkpoint) / "head.safetensors") != selection["head_sha256"]:
        raise ValueError("Checkpoint changed after selection")
    if sha256_file(Path(checkpoint) / "encoder/model.safetensors") != selection["encoder_sha256"]:
        raise ValueError("Encoder changed after selection")
    dataset = load_dataset(REPO, revision=REVISION, split="eval")
    predictor = DecisionPredictor.from_pretrained(checkpoint, max_length=a.max_length, overflow="error")
    counts = Counter()
    excluded = Counter()
    records = []
    predictions = []
    errors = []
    tasks = defaultdict(list)
    with accelerator_lock(predictor.device):
        pending = []

        def flush():
            if pending:
                predictions.extend(predictor.predict_records(pending))
                records.extend(pending)
                pending.clear()

        for idx, row in enumerate(dataset):
            if a.limit and idx >= a.limit:
                break
            counts[row["task_id"]] += 1
            try:
                record = transform(row)
                serialize(record, predictor.collator.tokenizer, a.max_length, "error")
            except ValueError as exc:
                if "budget" not in str(exc):
                    raise
                excluded[row["task_id"]] += 1
                errors.append({"row_id": row["row_id"], "reason": str(exc)})
                continue
            pending.append(record)
            if len(pending) >= 4:
                flush()
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(dataset)}", flush=True)
        flush()
    report = {
        "source": REPO,
        "revision": REVISION,
        "checkpoint": checkpoint,
        "max_length": a.max_length,
        "input_variant": "standard",
        "selection_sha256": sha256_file(a.selection),
        "attempted": sum(counts.values()),
        "scored": len(records),
        "excluded_for_context": sum(excluded.values()),
        "coverage": len(records) / max(1, sum(counts.values())),
        "task_counts": dict(counts),
        "task_excluded": dict(excluded),
        "claim": "Zero-shot relative to our fine-tuning sources; encoder pretraining contamination is unknown.",
    }
    if records:
        report["metrics"] = metrics(
            [r["probabilities"] for r in predictions],
            [r["target_probabilities"] for r in records],
            [r["options"] for r in records],
        )
        for i, r in enumerate(records):
            tasks[r["metadata"]["task_id"]].append(i)
        report["per_task"] = {
            task: metrics(
                [predictions[i]["probabilities"] for i in inds],
                [records[i]["target_probabilities"] for i in inds],
                [records[i]["options"] for i in inds],
            )
            for task, inds in tasks.items()
        }
    Path("reports/decisionbench.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/decisionbench-exclusions.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["per_task", "task_counts", "task_excluded"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
