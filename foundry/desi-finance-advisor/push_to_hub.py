"""
Fuse DPO adapters and push final model to Hugging Face Hub.

Run from repo root:
    python foundry/desi-finance-advisor/push_to_hub.py --repo shabul/mistral-7b-desi-finance-advisor
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.hub_utils import fuse_and_push

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "adapters", "dpo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo ID, e.g. shabul/mistral-7b-desi-finance-advisor")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    fuse_and_push(BASE_MODEL, ADAPTER_PATH, args.repo, args.private)


if __name__ == "__main__":
    main()
