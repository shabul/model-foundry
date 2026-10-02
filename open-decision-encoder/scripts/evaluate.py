"""Evaluate a selected checkpoint, preserving raw and calibrated metrics."""

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

from scipy.special import softmax

from decision_encoder.data.schema import read_jsonl
from decision_encoder.inference import DecisionPredictor
from decision_encoder.metrics import metrics, permutation_metrics
from decision_encoder.runtime import accelerator_lock, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="reports/evaluation.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--overflow", choices=["error", "truncate_state"], default="error")
    p.add_argument("--limit", type=int)
    p.add_argument("--permutations", type=int, default=10)
    p.add_argument("--permutation-examples", type=int, default=100)
    a = p.parse_args()
    rows = read_jsonl(a.data)
    random.Random(42).shuffle(rows)
    rows = rows[: a.limit] if a.limit else rows
    predictor = DecisionPredictor.from_pretrained(a.checkpoint, max_length=a.max_length, overflow=a.overflow)
    results = []
    with accelerator_lock(predictor.device):
        for start in range(0, len(rows), a.batch_size):
            results.extend(predictor.predict_records(rows[start : start + a.batch_size]))
            if start % 400 == 0:
                print(f"Evaluated {start}/{len(rows)}", flush=True)
        targets = [r["target_probabilities"] for r in rows]
        options = [r["options"] for r in rows]
        probabilities = [r["probabilities"] for r in results]
        raw = [softmax(r["logits"]).tolist() for r in results]
        report = {
            "checkpoint": a.checkpoint,
            "data_sha256": sha256_file(a.data),
            "n": len(rows),
            "temperature": predictor.model.temperature,
            "max_length": a.max_length,
            "overflow": a.overflow,
            "truncated": sum(r["truncated"] for r in results),
            "raw": metrics(raw, targets, options),
            "calibrated": metrics(probabilities, targets, options),
            "groups": {},
        }
        for dimension in ["source", "decision_type", "candidate_count"]:
            groups = defaultdict(list)
            for i, row in enumerate(rows):
                key = str(len(row["options"])) if dimension == "candidate_count" else row[dimension]
                groups[key].append(i)
            report["groups"][dimension] = {
                key: metrics(
                    [probabilities[i] for i in ids], [targets[i] for i in ids], [options[i] for i in ids]
                )
                for key, ids in groups.items()
            }
        rng = random.Random(314)
        stresses = []
        for row, result in list(zip(rows, results))[: a.permutation_examples]:
            for _ in range(a.permutations):
                order = list(range(len(row["options"])))
                rng.shuffle(order)
                perm = copy.deepcopy(row)
                perm["options"] = [row["options"][i] for i in order]
                perm["target_probabilities"] = [row["target_probabilities"][i] for i in order]
                output = predictor.predict_records([perm])[0]
                stresses.append(
                    permutation_metrics(
                        result["probabilities"],
                        output["probabilities"],
                        result["option_ids"],
                        output["option_ids"],
                    )
                )
        if stresses:
            report["permutations"] = {
                "examples": min(len(rows), a.permutation_examples),
                "draws": len(stresses),
                "flip_rate": sum(s["flip"] for s in stresses) / len(stresses),
                "mean_kl": sum(s["kl"] for s in stresses) / len(stresses),
                "maximum_probability_deviation": max(s["max_probability_deviation"] for s in stresses),
            }
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    predictions_path = Path("artifacts") / (path.stem + "-predictions.jsonl")
    predictions_path.parent.mkdir(exist_ok=True)
    with predictions_path.open("w") as f:
        for row, result in zip(rows, results):
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "source": row["source"],
                        **result,
                        "target_probabilities": row["target_probabilities"],
                    }
                )
                + "\n"
            )
    print(json.dumps({k: report[k] for k in ["n", "truncated", "raw", "calibrated"]}, indent=2))


if __name__ == "__main__":
    main()
