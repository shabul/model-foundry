"""Public intent transformations with deterministic candidate sampling."""

import hashlib
import random

from decision_encoder.data.schema import ABSTAIN_OPTION, state_hash


def transform_intent(state, gold_id, taxonomy, source, source_split, row_id, seed=42, variants=2):
    group = state_hash(state)
    for variant in range(variants):
        row_seed = int(hashlib.sha256(f"{seed}:{source}:{row_id}:{variant}".encode()).hexdigest()[:12], 16)
        rng = random.Random(row_seed)
        k = rng.randint(2, 8)
        gold = taxonomy[gold_id]
        pool = [x for x in taxonomy if x != gold_id]
        nearby = [x for x in pool if taxonomy[x]["group"] == gold["group"]]
        negatives = []
        for _ in range(k - 1):
            available = [x for x in nearby if x not in negatives] if rng.random() < 0.7 else []
            if not available:
                available = [x for x in pool if x not in negatives]
            negatives.append(rng.choice(available))
        ids = [gold_id] + negatives
        options = [{key: taxonomy[i][key] for key in ["id", "label", "description"]} for i in ids]
        target = 0
        roll = rng.random()
        abstain = roll < 0.125
        if abstain:
            options.pop(0)
            options.append(dict(ABSTAIN_OPTION))
            target = len(options) - 1
        elif roll < 0.5:
            options.append(dict(ABSTAIN_OPTION))
        y = [float(j == target) for j in range(len(options))]
        order = list(range(len(options)))
        rng.shuffle(order)
        yield {
            "id": f"{source}-{source_split}-{row_id}-{variant}",
            "source": source,
            "source_split": source_split,
            "group_id": group,
            "decision_type": "choice",
            "state": state,
            "question": "Which issue best describes this request?"
            if source == "banking77"
            else "Which action best describes this request?",
            "options": [options[j] for j in order],
            "target_probabilities": [y[j] for j in order],
            "metadata": {
                "original_label": gold_id,
                "source_row_id": str(row_id),
                "seed": row_seed,
                "abstention_reason": "gold_removed" if abstain else None,
                "augmentation": "mixed_negatives",
                "transform_version": 1,
            },
        }
