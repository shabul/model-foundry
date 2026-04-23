"""
Retrain with adjusted hyperparameters based on eval verdict.
Reads the last eval report, bumps params, trains, and pushes
to a versioned HF repo (e.g. shabul/qwen2.5-3b-feynman-explainer-v2).

Run from repo root:
    python foundry/feynman-explainer/eval/retrain.py --repo shabul/qwen2.5-3b-feynman-explainer
"""

import argparse
import os
import re
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from shared.hub_utils import fuse_and_push, upload_file

CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "..", "config", "lora_config.yaml")
EVAL_DIR     = os.path.dirname(__file__)
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "adapters")
MODEL_CARD   = os.path.join(os.path.dirname(__file__), "..", "MODEL_CARD.md")
BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"

# Param ladder — each retry escalates aggressively
RETRAIN_LADDER = [
    {"iters": 2000, "learning_rate": "3e-4", "lora_rank": 16, "note": "More iters + higher LR"},
    {"iters": 2500, "learning_rate": "3e-4", "lora_rank": 32, "note": "Higher rank for more capacity"},
    {"iters": 3000, "learning_rate": "4e-4", "lora_rank": 32, "note": "Max push"},
]


def read_last_version() -> int:
    reports = [f for f in os.listdir(EVAL_DIR) if re.match(r"report_v\d+\.md", f)]
    if not reports:
        return 0
    return max(int(re.search(r"v(\d+)", r).group(1)) for r in reports)


def read_composite_from_report(version: int) -> float | None:
    path = os.path.join(EVAL_DIR, f"report_v{version}.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    m = re.search(r"fine-tuned=([\d.]+)", content)
    return float(m.group(1)) if m else None


def update_config(iters: int, lr: str, rank: int):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["iters"] = iters
    cfg["learning_rate"] = float(lr)
    cfg["lora_parameters"]["rank"]  = rank
    cfg["lora_parameters"]["alpha"] = rank * 2
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Config updated: iters={iters}, lr={lr}, rank={rank}, alpha={rank*2}")


def run_training():
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", CONFIG_PATH]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Training failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Base HF repo, e.g. shabul/qwen2.5-3b-feynman-explainer")
    parser.add_argument("--step", type=int, default=None,
                        help="Force a specific ladder step (0, 1, 2). Default: auto from last eval.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    last_v = read_last_version()

    if args.step is not None:
        step = args.step
    else:
        # Auto-pick ladder step based on how many retrains we've done
        step = max(0, last_v - 1)

    if step >= len(RETRAIN_LADDER):
        print(f"Already at max ladder step ({step}). Check eval reports manually.")
        sys.exit(0)

    params = RETRAIN_LADDER[step]
    new_version = last_v + 1
    new_repo    = f"{args.repo}-v{new_version}"

    print(f"\n{'='*60}")
    print(f"Retrain v{new_version}")
    print(f"Ladder step  : {step} — {params['note']}")
    print(f"iters        : {params['iters']}")
    print(f"learning_rate: {params['learning_rate']}")
    print(f"LoRA rank    : {params['lora_rank']}")
    print(f"Target repo  : {new_repo}")
    print(f"{'='*60}\n")

    update_config(params["iters"], params["learning_rate"], params["lora_rank"])
    run_training()

    print(f"\nPushing v{new_version} → {new_repo}")
    fuse_and_push(BASE_MODEL, ADAPTER_PATH, new_repo, args.private)
    if os.path.exists(MODEL_CARD):
        upload_file(MODEL_CARD, new_repo, "README.md", f"Model card v{new_version}")

    print(f"\nDone. Now run eval on v{new_version}:")
    print(f"  python foundry/feynman-explainer/eval/evaluate.py --version {new_version} --adapter {ADAPTER_PATH}")
    print(f"\nCompare at:")
    print(f"  https://huggingface.co/{args.repo}")
    print(f"  https://huggingface.co/{new_repo}")


if __name__ == "__main__":
    main()
