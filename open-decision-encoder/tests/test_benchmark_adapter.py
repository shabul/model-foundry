"""Schema-only external adapter tests using invented fixtures, never benchmark examples."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "ode_external_eval", Path(__file__).parents[1] / "scripts/evaluate_decisionbench.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_optional_description_and_ordinal_value():
    row = {
        "row_id": "schema-fixture",
        "task_id": "fixture",
        "domain": "fixture",
        "reasoning_required": False,
        "state_json": '{"measurement": 2}',
        "instruction": "Choose a level.",
        "primitive": "ordinal_scoring",
        "candidates_json": json.dumps(
            [
                {"id": "a", "label": "A", "description": None, "ordinal_value": 10},
                {"id": "b", "label": "B", "description": "Second level", "ordinal_value": 20},
            ]
        ),
        "gold_probabilities": [0.25, 0.75],
    }
    result = module.transform(row)
    assert result["source"] == "decisionbench"
    assert result["source_split"] == "eval"
    assert result["options"][0]["description"] == "A"
    assert [o["value"] for o in result["options"]] == [10.0, 20.0]
    assert result["target_probabilities"] == [0.25, 0.75]
