"""
Fuse and push Sherlock Debugger model to Hugging Face Hub.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.hub_utils import fuse_and_push

BASE_MODEL = "mlx-community/gemma-2-9b-it-4bit"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "adapters")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Target HF repo ID")
    args = parser.parse_args()

    fuse_and_push(BASE_MODEL, ADAPTER_PATH, args.repo)


if __name__ == "__main__":
    main()
