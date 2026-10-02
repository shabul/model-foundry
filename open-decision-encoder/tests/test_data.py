import json

import pytest

from decision_encoder.data.public import transform_intent
from decision_encoder.data.schema import validate_record
from decision_encoder.data.typed_decisions import transform


def test_abstention_has_positive_and_negative_examples():
    tax = {
        str(i): {"id": str(i), "label": f"Label {i}", "description": f"Intent {i}", "group": str(i // 4)}
        for i in range(10)
    }
    seen = set()
    for r in transform_intent("A request", "0", tax, "banking77", "train", "row", variants=100):
        validate_record(r)
        for i, o in enumerate(r["options"]):
            if o.get("is_abstain"):
                seen.add(r["target_probabilities"][i])
    assert seen == {0.0, 1.0}


def test_typed_soft_targets_and_ordinal_values():
    row = {
        "id": "row",
        "workflow": "invoice_processing",
        "state": json.dumps({"amount": 10}),
        "questions": json.dumps(
            {
                "approved": {"type": "noul", "instructions": "Approved?"},
                "risk": {"type": "score", "instructions": "Risk?", "criteria": ["Low", "High"]},
            }
        ),
        "gold": json.dumps(
            {
                "approved": {"probabilities": {"false": 0.2, "true": 0.8}},
                "risk": {"probabilities": {"0": 0.3, "1": 0.7}},
            }
        ),
    }
    records = list(transform(row, "train"))
    assert records[0]["group_id"] == records[1]["group_id"]
    for r in records:
        validate_record(r)
        y = dict(zip([o["id"] for o in r["options"]], r["target_probabilities"]))
        assert y == ({"false": 0.2, "true": 0.8} if r["decision_type"] == "boolean" else {"0": 0.3, "1": 0.7})
        if r["decision_type"] == "ordinal":
            assert {o["value"] for o in r["options"]} == {0.0, 1.0}


def test_unknown_typed_criteria_rejected():
    row = {
        "id": "x",
        "state": "{}",
        "questions": json.dumps({"q": {"type": "choice", "instructions": "Q"}}),
        "gold": "{}",
    }
    with pytest.raises(ValueError, match="criteria"):
        list(transform(row, "train"))


def test_synthetic_test_groups_are_reserved():
    from decision_encoder.data.schema import assert_no_leakage
    from decision_encoder.data.synthetic import generate_synthetic

    splits = {s: [] for s in ["train", "validation", "calibration", "test"]}
    for r in generate_synthetic(1000, reserve_test=True):
        splits[r["metadata"]["split"]].append(r)
    assert splits["test"]
    assert all(r["source_split"] == "test" for r in splits["test"])
    assert_no_leakage(splits)
