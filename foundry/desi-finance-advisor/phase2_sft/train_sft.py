"""
LoRA SFT fine-tuning for Desi Finance Advisor via mlx-lm.

Run from repo root:
    python foundry/desi-finance-advisor/phase2_sft/train_sft.py
"""

import os
import subprocess
import sys

CONFIG = os.path.join(os.path.dirname(__file__), "sft_config.yaml")


def main():
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", CONFIG, *sys.argv[1:]]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
