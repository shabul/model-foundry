"""
Fuse LoRA adapters and push to Hugging Face Hub.

Run from repo root:
    python foundry/feynman-explainer/push_to_hub.py --repo shabul/qwen2.5-3b-feynman-explainer
    python foundry/feynman-explainer/push_to_hub.py --repo shabul/qwen2.5-3b-feynman-explainer --private
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.hub_utils import fuse_and_push, upload_file

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "adapters")
MODEL_CARD = os.path.join(os.path.dirname(__file__), "MODEL_CARD.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    fuse_and_push(BASE_MODEL, ADAPTER_PATH, args.repo, args.private)

    if os.path.exists(MODEL_CARD):
        upload_file(MODEL_CARD, args.repo, "README.md", "Add model card")


if __name__ == "__main__":
    main()
