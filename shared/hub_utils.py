"""
Shared Hugging Face Hub utilities — fuse LoRA adapters and upload.
"""

import os
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, login


def ensure_hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("HF_TOKEN not set — launching interactive login ...")
        login()


def fuse_and_push(base_model: str, adapter_path: str, repo_id: str, private: bool = False):
    """Fuse LoRA adapters into the base model and upload to HF Hub."""
    ensure_hf_login()

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}. Run train.py first.")

    adapters = [f for f in os.listdir(adapter_path) if f.endswith(".safetensors")]
    if not adapters:
        raise FileNotFoundError(f"No .safetensors files in {adapter_path}")
    print(f"Adapters found: {adapters}")

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", base_model,
            "--adapter-path", adapter_path,
            "--save-path", tmp,
            "--upload-repo", repo_id,
        ]
        if private:
            cmd.append("--private")

        print("Fusing + uploading:", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("mlx_lm.fuse failed")

    print(f"\nDone → https://huggingface.co/{repo_id}")


def upload_file(local_path: str, repo_id: str, path_in_repo: str, commit_message: str = "Update file"):
    ensure_hf_login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Uploaded {local_path} → {repo_id}/{path_in_repo}")
