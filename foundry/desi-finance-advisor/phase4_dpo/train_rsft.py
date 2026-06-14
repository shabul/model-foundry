"""
Rejection-Sampling SFT (RSFT) — second-pass fine-tuning on Gemini-chosen responses.

mlx-lm 0.31.3 does not support DPO natively (no DPODataset class).
Instead we do RSFT: train only on the highest-quality (chosen) responses
identified by the Gemini judge, starting from the SFT iter-300 adapter.
This achieves a similar alignment effect to DPO in that the model learns
specifically from the best outputs rather than random samples.

Run from repo root:
    python foundry/desi-finance-advisor/phase4_dpo/train_rsft.py
"""

import os
import subprocess
import sys

CONFIG = os.path.join(os.path.dirname(__file__), "rsft_config.yaml")


def main():
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", CONFIG, *sys.argv[1:]]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
