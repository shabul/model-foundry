"""
Push the synthetic Devil's Advocate dataset to Hugging Face Hub.

Run from repo root:
    python foundry/devils-advocate/push_dataset_to_hub.py --repo shabul/devils-advocate-dataset
"""

import argparse
import os
import sys

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Target HF dataset repo ID")
    args = parser.parse_args()

    print(f"Loading dataset from {DATA_DIR}")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "valid": os.path.join(DATA_DIR, "valid.jsonl")
    })

    print(f"Pushing to https://huggingface.co/datasets/{args.repo}")
    dataset.push_to_hub(args.repo)


if __name__ == "__main__":
    main()
