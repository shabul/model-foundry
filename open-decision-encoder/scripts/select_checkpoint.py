"""Freeze checkpoint selection using validation NLL before final tests."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from decision_encoder.runtime import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["checkpoints/frozen", "checkpoints/last4"])
    a = p.parse_args()
    candidates = []
    for directory in a.runs:
        path = Path(directory)
        run = json.loads((path / "run.json").read_text())
        cfg = json.loads((path / "best/decision_config.json").read_text())
        candidates.append(
            {
                "checkpoint": str(path / "best"),
                "validation_nll": cfg["metadata"]["validation"]["nll"],
                "run_sha256": sha256_file(path / "run.json"),
                "validation_sha256": run["validation_sha256"],
            }
        )
    if len({r["validation_sha256"] for r in candidates}) != 1:
        raise ValueError("Checkpoints used different validation data")
    selected = dict(min(candidates, key=lambda c: c["validation_nll"]))
    selected.update(
        {
            "candidates": candidates.copy(),
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "criterion": "minimum validation soft-target NLL; no external test feedback",
            "head_sha256": sha256_file(Path(selected["checkpoint"]) / "head.safetensors"),
            "encoder_sha256": sha256_file(Path(selected["checkpoint"]) / "encoder/model.safetensors"),
        }
    )
    # Avoid referencing selected recursively through candidates.
    selected["candidates"] = [
        {
            k: v
            for k, v in r.items()
            if k in {"checkpoint", "validation_nll", "run_sha256", "validation_sha256"}
        }
        for r in candidates
    ]
    Path("reports/selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
