"""
Upload the Feynman dataset to a Hugging Face dataset repo.

Run from repo root:
    python foundry/feynman-explainer/push_dataset_to_hub.py --repo shabul/feynman-explainer-dataset
    python foundry/feynman-explainer/push_dataset_to_hub.py --repo shabul/feynman-explainer-dataset --private
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.hub_utils import ensure_hf_login

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_CARD = os.path.join(os.path.dirname(__file__), "DATASET_CARD.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF dataset repo id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    ensure_hf_login()
    api = HfApi()

    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    if os.path.exists(DATASET_CARD):
        api.upload_file(
            path_or_fileobj=DATASET_CARD,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    api.upload_folder(
        folder_path=DATA_DIR,
        path_in_repo="data",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Upload dataset files",
    )

    print(f"\nDone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
