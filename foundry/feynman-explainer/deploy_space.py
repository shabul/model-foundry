"""
Create a Hugging Face Space and upload the Gradio chat app.

Run from repo root:
    python foundry/feynman-explainer/deploy_space.py
    python foundry/feynman-explainer/deploy_space.py --space shabul/feynman-explainer
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, login

SPACE_DIR = os.path.join(os.path.dirname(__file__), "space")
DEFAULT_SPACE = "shabul/feynman-explainer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default=DEFAULT_SPACE,
                        help=f"HF Space repo id (default: {DEFAULT_SPACE})")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        login()

    api = HfApi()

    # Create Space if it doesn't exist
    try:
        api.create_repo(
            repo_id=args.space,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"Space ready: https://huggingface.co/spaces/{args.space}")
    except Exception as e:
        print(f"Space creation note: {e}")

    # Upload all files from space/
    for fname in ["README.md", "app.py", "requirements.txt"]:
        local = os.path.join(SPACE_DIR, fname)
        if not os.path.exists(local):
            print(f"  Skipping missing: {fname}")
            continue
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=fname,
            repo_id=args.space,
            repo_type="space",
            commit_message=f"Deploy Feynman Explainer Gradio app ({fname})",
        )
        print(f"  ✓ Uploaded {fname}")

    print(f"\nSpace deployed → https://huggingface.co/spaces/{args.space}")
    print("It will build in ~2-3 minutes. First cold load takes ~30-60s (model download).")


if __name__ == "__main__":
    main()
