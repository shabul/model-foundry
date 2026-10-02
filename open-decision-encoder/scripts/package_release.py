"""Package selected local artifacts; uploads require an explicit --repo argument."""

import argparse
import json
import shutil
from pathlib import Path

from decision_encoder.runtime import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/release")
    p.add_argument("--repo", help="Optional explicit Hugging Face destination; otherwise local only")
    a = p.parse_args()
    selection = json.loads(Path("reports/selection.json").read_text())
    source = Path(selection["checkpoint"])
    if sha256_file(source / "head.safetensors") != selection["head_sha256"]:
        raise ValueError("Selection no longer matches weights")
    if sha256_file(source / "encoder/model.safetensors") != selection["encoder_sha256"]:
        raise ValueError("Encoder changed after selection")
    if not Path("reports/evaluation.json").exists():
        raise ValueError("Complete final evaluation before packaging")
    evaluation = json.loads(Path("reports/evaluation.json").read_text())
    calibration = json.loads((source / "temperature.json").read_text())
    if Path(evaluation["checkpoint"]).resolve() != source.resolve():
        raise ValueError("Evaluation belongs to a different checkpoint")
    if evaluation["temperature"] != calibration["temperature"] or not calibration.get("data_sha256"):
        raise ValueError("Calibrated evaluation does not match release temperature")
    if not evaluation["n"] or not Path("MODEL_CARD.md").exists():
        raise ValueError("Measured evaluation and model card are required")
    baseline = json.loads(Path("checkpoints/frozen/run.json").read_text())["baseline"]["nll"]
    if selection["validation_nll"] >= baseline * 0.95:
        raise ValueError("Selected checkpoint did not meaningfully improve validation NLL")
    target = Path(a.output)
    if target.exists():
        raise ValueError(f"Release directory already exists: {target}")
    shutil.copytree(source, target)
    if (source.parent / "source_snapshot").exists():
        shutil.copytree(source.parent / "source_snapshot", target / "training_source_snapshot")
    for name in ["src", "scripts", "configs", "reports"]:
        shutil.copytree(name, target / name, ignore=shutil.ignore_patterns("*.log", "__pycache__"))
    for name in [
        "pyproject.toml",
        "uv.lock",
        "IMPLEMENTATION.md",
        "README.md",
        "MODEL_CARD.md",
        "LICENSE",
        "NOTICE",
    ]:
        if Path(name).exists():
            shutil.copy2(
                name,
                target
                / ("README.md" if name == "MODEL_CARD.md" else "USAGE.md" if name == "README.md" else name),
            )
    shutil.copytree("data/manifests", target / "data/manifests")
    shutil.copytree("data/taxonomies", target / "data/taxonomies")
    files = {
        str(path.relative_to(target)): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    (target / "release_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(f"Local release: {target} ({len(files)} files)")
    if a.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=a.repo, folder_path=str(target), commit_message="Release Open Decision Encoder"
        )
        print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
