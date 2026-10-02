"""Synchronized warmed inference measurements for a saved checkpoint."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from decision_encoder.inference import DecisionPredictor
from decision_encoder.runtime import accelerator_lock, synchronize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repeats", type=int, default=30)
    a = p.parse_args()
    predictor = DecisionPredictor.from_pretrained(a.checkpoint, max_length=1024)
    results = []
    with accelerator_lock(predictor.device):
        for words in [32, 128, 256]:
            for k in [2, 4, 8]:
                row = {
                    "state": "The system completed the request successfully. " * max(1, words // 8),
                    "question": "Which action should be taken?",
                    "options": [
                        {"id": str(i), "label": f"Action {i}", "description": f"Use action {i}."}
                        for i in range(k)
                    ],
                }
                for _ in range(3):
                    predictor.predict_records([row])
                samples = []
                for _ in range(a.repeats):
                    synchronize(predictor.device)
                    start = time.perf_counter()
                    predictor.predict_records([row])
                    synchronize(predictor.device)
                    samples.append(time.perf_counter() - start)
                encoded = predictor.collator([row])
                results.append(
                    {
                        "candidate_count": k,
                        "input_tokens": int(encoded["attention_mask"].sum()),
                        "p50_ms": float(np.percentile(samples, 50) * 1000),
                        "p95_ms": float(np.percentile(samples, 95) * 1000),
                        "examples_per_second": 1 / float(np.mean(samples)),
                        "allocated_bytes": torch.mps.current_allocated_memory()
                        if str(predictor.device) == "mps"
                        else None,
                        "driver_bytes": torch.mps.driver_allocated_memory()
                        if str(predictor.device) == "mps"
                        else None,
                    }
                )
    report = {
        "checkpoint": a.checkpoint,
        "device": str(predictor.device),
        "batch_size": 1,
        "repeats": a.repeats,
        "timing_scope": "end-to-end tokenization, transfer, forward, CPU result; warmed and synchronized",
        "memory_scope": "current allocated/driver snapshots, not whole-system peak",
        "measurements": results,
    }
    Path("reports/benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
