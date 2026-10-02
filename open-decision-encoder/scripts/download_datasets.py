"""Pin and download permitted training sources; benchmark data is never downloaded here."""

import argparse
import json
from pathlib import Path

import requests
from datasets import load_dataset
from huggingface_hub import HfApi

from decision_encoder.runtime import sha256_file

SOURCES = {
    "banking77": ("PolyAI/banking77", None, "cc-by-4.0"),
    "massive": ("AmazonScience/massive", "en-US", "cc-by-4.0"),
    "typed_decisions": ("LocalLLaMA/typed-decisions", "all", "apache-2.0"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    p.add_argument("--refresh-revisions", action="store_true")
    args = p.parse_args()
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/manifests/sources.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.sources:
        repo, config, license_name = SOURCES[name]
        pinned = manifest.get(name, {}) if not args.refresh_revisions else {}
        revision = pinned.get("revision") or HfApi().dataset_info(repo).sha
        extra = {}
        if name == "banking77":
            response = requests.get(
                "https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master", timeout=30
            )
            response.raise_for_status()
            commit = pinned.get("upstream_commit") or response.json()["sha"]
            files = {}
            for split in ["train", "test"]:
                url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{commit}/banking_data/{split}.csv"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                file = out / f"banking77-{split}.csv"
                file.write_bytes(response.content)
                files[split] = str(file)
            dataset = load_dataset("csv", data_files=files)
            extra = {"upstream_commit": commit, "file_sha256": {s: sha256_file(f) for s, f in files.items()}}
        elif name == "massive":
            parquet_revision = (
                pinned.get("parquet_revision")
                or HfApi().dataset_info(repo, revision="refs/convert/parquet").sha
            )
            files = {
                s: f"hf://datasets/{repo}@{parquet_revision}/en-US/{s}/0000.parquet"
                for s in ["train", "validation", "test"]
            }
            dataset = load_dataset("parquet", data_files=files)
            extra = {"parquet_revision": parquet_revision}
        else:
            dataset = load_dataset(repo, config, revision=revision)
        dataset.save_to_disk(str(out / name))
        manifest[name] = {
            "repo": repo,
            "config": config,
            "revision": revision,
            "license": license_name,
            "splits": {s: len(d) for s, d in dataset.items()},
            "features": str(dataset[next(iter(dataset))].features),
            "usage": "original test evaluation-only; train-derived grouped splits",
            **extra,
        }
        print(name, json.dumps(manifest[name]), flush=True)
    Path("data/manifests/sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
