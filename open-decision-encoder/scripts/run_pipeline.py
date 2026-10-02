"""Sequential local pipeline. Each subprocess must succeed before the next stage starts."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(script, *args):
    command = [sys.executable, f"scripts/{script}.py", *args]
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--full-if-improved", action="store_true")
    args = p.parse_args()
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    if not json.loads(Path("reports/hardware.json").read_text())["passed"]:
        raise RuntimeError("Hardware gate failed")
    if not json.loads(Path("reports/smoke.json").read_text())["passed"]:
        raise RuntimeError("Tiny-overfit gate failed")
    runs = ["checkpoints/frozen", "checkpoints/last4"]
    if not args.skip_training:
        run("train", "--config", "configs/frozen.yaml")
        run("train", "--config", "configs/last4.yaml")
        if args.full_if_improved:
            frozen = json.loads(Path("checkpoints/frozen/run.json").read_text())["best_validation_nll"]
            last4 = json.loads(Path("checkpoints/last4/run.json").read_text())["best_validation_nll"]
            if last4 < frozen * 0.95:
                run("train", "--config", "configs/full.yaml")
                runs.append("checkpoints/full")
    run("select_checkpoint", "--runs", *runs)
    checkpoint = json.loads(Path("reports/selection.json").read_text())["checkpoint"]
    run("calibrate", "--checkpoint", checkpoint, "--overflow", "truncate_state")
    run("build_dataset", "--include-test")
    run(
        "evaluate",
        "--checkpoint",
        checkpoint,
        "--data",
        "data/processed/test.jsonl",
        "--overflow",
        "truncate_state",
    )
    run("build_stress_suite")
    run(
        "evaluate",
        "--checkpoint",
        checkpoint,
        "--data",
        "data/processed/stress.jsonl",
        "--output",
        "reports/stress.json",
        "--overflow",
        "error",
    )
    run("benchmark_mps", "--checkpoint", checkpoint)
    run("evaluate_decisionbench")
    Path("reports/pipeline_complete.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint": checkpoint,
                "publication": "local only; no Hub upload performed",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
