"""Publishing must not bypass calibration or validation readiness checks."""

import importlib.util
import json
from pathlib import Path

import pytest

from decision_encoder.runtime import sha256_file

SPEC = importlib.util.spec_from_file_location(
    "ode_package_release", Path(__file__).parents[1] / "scripts/package_release.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def prepare(tmp_path, monkeypatch, calibrated=True, nll=0.5):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["package_release.py"])
    source = Path("checkpoints/selected/best")
    (source / "encoder").mkdir(parents=True)
    (source / "head.safetensors").write_bytes(b"test head")
    (source / "encoder/model.safetensors").write_bytes(b"test encoder")
    (source / "temperature.json").write_text(
        json.dumps({"temperature": 1.2, **({"data_sha256": "hash"} if calibrated else {})})
    )
    Path("reports").mkdir()
    Path("reports/selection.json").write_text(
        json.dumps(
            {
                "checkpoint": str(source),
                "validation_nll": nll,
                "head_sha256": sha256_file(source / "head.safetensors"),
                "encoder_sha256": sha256_file(source / "encoder/model.safetensors"),
            }
        )
    )
    Path("reports/evaluation.json").write_text(
        json.dumps({"checkpoint": str(source), "temperature": 1.2, "n": 100})
    )
    Path("MODEL_CARD.md").write_text("Measured model card")
    Path("checkpoints/frozen").mkdir()
    Path("checkpoints/frozen/run.json").write_text(json.dumps({"baseline": {"nll": 1.5}}))


def test_release_rejects_uncalibrated_weights(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, calibrated=False)
    with pytest.raises(ValueError, match="Calibrated evaluation"):
        module.main()
    assert not Path("artifacts/release").exists()


def test_release_rejects_unimproved_model(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, nll=1.49)
    with pytest.raises(ValueError, match="meaningfully improve"):
        module.main()
    assert not Path("artifacts/release").exists()
