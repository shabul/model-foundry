"""Fit one temperature on held-out calibration groups, never test data."""

import argparse
import json
from pathlib import Path

from decision_encoder.calibration import fit_temperature
from decision_encoder.data.schema import assert_no_leakage, read_jsonl
from decision_encoder.inference import DecisionPredictor
from decision_encoder.runtime import accelerator_lock, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/processed/calibration.jsonl")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--overflow", choices=["error", "truncate_state"], default="error")
    a = p.parse_args()
    manifest = json.loads(Path("data/manifests/dataset.json").read_text())
    if sha256_file(a.data) != manifest["splits"]["calibration"]["sha256"]:
        raise ValueError("Calibration data must match the frozen calibration manifest")
    rows = read_jsonl(a.data)
    assert_no_leakage({"calibration": rows})
    predictor = DecisionPredictor.from_pretrained(a.checkpoint, max_length=a.max_length, overflow=a.overflow)
    outputs = []
    with accelerator_lock(predictor.device):
        for start in range(0, len(rows), 4):
            outputs.extend(predictor.predict_records(rows[start : start + 4], calibrated=False))
    fit = fit_temperature([r["logits"] for r in outputs], [r["target_probabilities"] for r in rows])
    fit.update(
        data_sha256=sha256_file(a.data),
        checkpoint=str(a.checkpoint),
        truncated=sum(r["truncated"] for r in outputs),
        max_length=a.max_length,
        overflow=a.overflow,
    )
    (Path(a.checkpoint) / "temperature.json").write_text(json.dumps(fit, indent=2) + "\n")
    Path("reports/calibration.json").write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
