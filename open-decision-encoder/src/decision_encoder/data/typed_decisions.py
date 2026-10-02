"""Preserve teacher distributions and group all questions from a state together."""

import hashlib
import json
import random

from decision_encoder.data.schema import state_hash


def transform(row, source_split, seed=42):
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    state = json.dumps(json.loads(row["state"]), sort_keys=True, ensure_ascii=False)
    for question_id, question in questions.items():
        kind = question["type"]
        criteria = question.get("criteria")
        if criteria is None and kind == "noul":
            criteria = {"false": "The statement is false.", "true": "The statement is true."}
        if criteria is None:
            raise ValueError("Missing non-boolean criteria")
        criteria = (
            {str(i): value for i, value in enumerate(criteria)} if isinstance(criteria, list) else criteria
        )
        options = [
            {"id": str(key), "label": str(key).replace("_", " "), "description": description}
            for key, description in criteria.items()
        ]
        if kind == "score":
            for o in options:
                o["value"] = float(o["id"])
        distribution = gold[question_id]["probabilities"]
        y = [float(distribution[o["id"]]) for o in options]
        total = sum(y)
        if abs(total - 1) > 1e-4 or any(v < 0 for v in y):
            raise ValueError("Invalid source distribution")
        y = [v / total for v in y]  # Source rounds to six decimals.
        row_seed = int(hashlib.sha256(f"{seed}:{row['id']}:{question_id}".encode()).hexdigest()[:12], 16)
        order = list(range(len(options)))
        random.Random(row_seed).shuffle(order)
        yield {
            "id": f"typed-{row['id']}-{question_id}",
            "source": "typed_decisions",
            "source_split": source_split,
            "group_id": state_hash(state),
            "decision_type": {"choice": "choice", "noul": "boolean", "score": "ordinal"}[kind],
            "state": state,
            "question": question["instructions"],
            "options": [options[j] for j in order],
            "target_probabilities": [y[j] for j in order],
            "metadata": {
                "source_row_id": row["id"],
                "workflow": row["workflow"],
                "question_id": question_id,
                "seed": row_seed,
                "gold_semantics": "teacher_distribution",
            },
        }
