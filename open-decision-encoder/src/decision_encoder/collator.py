"""Variable-token and variable-candidate batching."""

import copy
import random

import torch

from decision_encoder.data.schema import validate_record
from decision_encoder.serialization import serialize


class DecisionCollator:
    def __init__(self, tokenizer, max_length=512, shuffle=False, seed=42, overflow="error"):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.shuffle, self.rng, self.overflow = shuffle, random.Random(seed), overflow

    def __call__(self, records):
        if not records:
            raise ValueError("Empty batch")
        examples = copy.deepcopy(records)
        has_targets = ["target_probabilities" in r for r in examples]
        if any(has_targets) and not all(has_targets):
            raise ValueError("Cannot mix labeled and unlabeled examples")
        for r in examples:
            validate_record(r, require_target=all(has_targets))
            if self.shuffle:
                order = list(range(len(r["options"])))
                self.rng.shuffle(order)
                r["options"] = [r["options"][i] for i in order]
                if all(has_targets):
                    r["target_probabilities"] = [r["target_probabilities"][i] for i in order]
        encoded = [serialize(r, self.tokenizer, self.max_length, self.overflow) for r in examples]
        length = max(len(r["input_ids"]) for r in encoded)
        candidates = max(len(r["candidate_positions"]) for r in encoded)
        batch = {
            "input_ids": torch.full((len(records), length), self.tokenizer.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(records), length), dtype=torch.long),
            "candidate_positions": torch.zeros((len(records), candidates), dtype=torch.long),
            "candidate_mask": torch.zeros((len(records), candidates), dtype=torch.bool),
        }
        if all(has_targets):
            batch["targets"] = torch.zeros((len(records), candidates), dtype=torch.float32)
        for i, (enc, record) in enumerate(zip(encoded, examples)):
            n, k = len(enc["input_ids"]), len(enc["candidate_positions"])
            batch["input_ids"][i, :n] = torch.tensor(enc["input_ids"])
            batch["attention_mask"][i, :n] = 1
            batch["candidate_positions"][i, :k] = torch.tensor(enc["candidate_positions"])
            batch["candidate_mask"][i, :k] = True
            if all(has_targets):
                batch["targets"][i, :k] = torch.tensor(record["target_probabilities"])
        batch["metadata"] = {
            "option_ids": [[o["id"] for o in r["options"]] for r in examples],
            "truncated": [r["truncated"] for r in encoded],
        }
        return batch
