"""
LoRA fine-tuning via mlx-lm for Devil's Advocate.

Run from repo root: python foundry/devils-advocate/train.py
"""

import os
import subprocess
import sys

CONFIG = os.path.join(os.path.dirname(__file__), "config", "lora_config.yaml")


def main():
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", CONFIG, *sys.argv[1:]]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
