import copy
import json

import numpy as np
import pytest
import torch

from decision_encoder.calibration import fit_temperature
from decision_encoder.collator import DecisionCollator
from decision_encoder.data.schema import assert_no_leakage, split_group, validate_record
from decision_encoder.data.synthetic import generate_synthetic
from decision_encoder.inference import DecisionPredictor
from decision_encoder.metrics import metrics, permutation_metrics
from decision_encoder.modeling import DecisionEncoder
from decision_encoder.serialization import serialize


@pytest.mark.parametrize("target", [[-1, 2], [float("nan"), 0], [0.1, 0.2], [1]])
def test_bad_targets(record, target):
    record["target_probabilities"] = target
    with pytest.raises(ValueError):
        validate_record(record)


def test_duplicate_options(record):
    record["options"][1]["id"] = "yes"
    with pytest.raises(ValueError):
        validate_record(record)


def test_mask_in_user_text_is_not_candidate(record, tokenizer):
    record["state"] = "[MASK] The state is ready."
    result = serialize(record, tokenizer)
    assert len(result["candidate_positions"]) == 2
    assert result["input_ids"].count(tokenizer.mask_token_id) == 3
    assert all(result["input_ids"][p] == tokenizer.mask_token_id for p in result["candidate_positions"])


def test_overflow(record, tokenizer):
    record["state"] = "ready " * 500
    with pytest.raises(ValueError):
        serialize(record, tokenizer, 64)
    result = serialize(record, tokenizer, 64, "truncate_state")
    assert result["truncated"] and len(result["input_ids"]) == 64
    record["options"][0]["description"] = "ready " * 100
    with pytest.raises(ValueError):
        serialize(record, tokenizer, 64, "truncate_state")


def test_variable_candidates_and_gradients(record, tokenizer, model):
    other = copy.deepcopy(record)
    other["options"].append({"id": "maybe", "label": "Maybe", "description": "Unclear"})
    other["target_probabilities"] = [0.0, 0.0, 1.0]
    batch = DecisionCollator(tokenizer)([record, other])
    batch.pop("metadata")
    out = model(**batch)
    assert torch.allclose(out["probabilities"].sum(-1), torch.ones(2))
    assert out["probabilities"][0, 2] == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.set_trainable("frozen")
    model.train()
    assert not model.encoder.training
    model(**batch)["loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.head[-1].weight.grad is not None


def test_roundtrip(record, tokenizer, model, tmp_path):
    predictor = DecisionPredictor(model, tokenizer, device="cpu")
    a = predictor.decide(record["state"], record["question"], record["options"])
    model.save_pretrained(tmp_path / "saved", tokenizer)
    restored = DecisionPredictor.from_pretrained(tmp_path / "saved", device="cpu")
    b = restored.decide(record["state"], record["question"], record["options"])
    assert a == b
    assert isinstance(DecisionEncoder.from_pretrained(tmp_path / "saved"), DecisionEncoder)


def test_shuffling_targets(record, tokenizer):
    col = DecisionCollator(tokenizer, shuffle=True)
    for _ in range(10):
        batch = col([record])
        target = dict(zip(batch["metadata"]["option_ids"][0], batch["targets"][0].tolist()))
        assert target["yes"] == pytest.approx(0.7)
        assert target["no"] == pytest.approx(0.3)


def test_permutation_alignment():
    m = permutation_metrics([0.7, 0.2, 0.1], [0.1, 0.7, 0.2], ["a", "b", "c"], ["c", "a", "b"])
    assert m == {"flip": 0.0, "kl": 0.0, "max_probability_deviation": 0.0}


def test_metrics_and_calibration():
    m = metrics([[0.8, 0.2], [0.1, 0.9]], [[1.0, 0.0], [0.0, 1.0]])
    assert m["accuracy"] == 1.0
    assert m["nll"] == pytest.approx(-np.log(0.8 * 0.9) / 2)
    assert m["brier"] == pytest.approx(0.05)
    fit = fit_temperature([[4.0, 0.0], [0.0, 4.0]], [[0.7, 0.3], [0.3, 0.7]])
    assert fit["calibrated_nll"] < fit["raw_nll"]
    assert fit["temperature"] > 1.0


def test_leakage(record):
    other = copy.deepcopy(record)
    other["id"] = "other"
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage({"train": [record], "validation": [other]})
    other["source"] = "DecisionBench"
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_leakage({"train": [other]})
    other["source"] = "typed_decisions"
    other["source_split"] = "test"
    with pytest.raises(ValueError, match="Forbidden"):
        assert_no_leakage({"train": [other]})


def test_synthetic_reproducible_and_grouped():
    a, b = list(generate_synthetic(200)), list(generate_synthetic(200))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    splits = {s: [] for s in ["train", "validation", "calibration"]}
    for r in a:
        validate_record(r)
        splits[split_group(r["group_id"])].append(r)
    assert_no_leakage(splits)
    assert len({r["metadata"]["domain"] for r in a}) == 10


def test_hub_checkpoint_loading_is_artifact_only(record, tokenizer, model, tmp_path, monkeypatch):
    model.save_pretrained(tmp_path / "hub", tokenizer)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    predictor = DecisionPredictor.from_pretrained(
        "owner/decision-model", revision="immutable-sha", device="cpu"
    )
    result = predictor.decide(record["state"], record["question"], record["options"])
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert len(calls) == 1
    assert calls[0]["revision"] == "immutable-sha"
    assert "*.py" not in calls[0]["allow_patterns"]
    assert "head.safetensors" in calls[0]["allow_patterns"]
