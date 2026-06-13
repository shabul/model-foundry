"""
Push the synthetic Sherlock Debugger dataset to Hugging Face Hub.
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

    dataset = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "valid": os.path.join(DATA_DIR, "valid.jsonl")
    })
    dataset.push_to_hub(args.repo)


if __name__ == "__main__":
    main()
