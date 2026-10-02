"""Reproducibility and single-process accelerator ownership."""

import contextlib
import fcntl
import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name="auto"):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


@contextlib.contextmanager
def accelerator_lock(device):
    if str(device) != "mps":
        yield
        return
    path = Path(tempfile.gettempdir()) / f"open-decision-encoder-mps-{os.getuid()}.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another decision-encoder MPS job is running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(root):
    root = Path(root)
    files = sorted(
        p
        for folder in ["src", "scripts", "configs"]
        for p in (root / folder).rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    )
    digest = hashlib.sha256(
        json.dumps([(str(p.relative_to(root)), sha256_file(p)) for p in files]).encode()
    ).hexdigest()

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest,
        "torch": torch.__version__,
    }
