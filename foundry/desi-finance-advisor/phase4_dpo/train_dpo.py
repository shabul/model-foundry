"""
DPO fine-tuning starting from the SFT adapter via mlx-lm.

mlx-lm >= 0.21 supports DPO via the --method dpo flag.
Data format: JSONL with {prompt, chosen, rejected} keys.

Run from repo root:
    python foundry/desi-finance-advisor/phase4_dpo/train_dpo.py
"""

import os
import subprocess
import sys

CONFIG = os.path.join(os.path.dirname(__file__), "dpo_config.yaml")


def main():
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", CONFIG, *sys.argv[1:]]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nNote: if mlx-lm reported an unrecognised argument for DPO,")
        print("upgrade mlx-lm:  pip install -U mlx-lm")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
