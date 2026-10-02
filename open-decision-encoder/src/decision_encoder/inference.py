"""Public inference API; no confidence threshold is silently applied."""

import copy
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from decision_encoder.collator import DecisionCollator
from decision_encoder.data.schema import ABSTAIN_ID, ABSTAIN_OPTION
from decision_encoder.metrics import stable_argmax
from decision_encoder.modeling import DecisionEncoder
from decision_encoder.runtime import device_for


class DecisionPredictor:
    def __init__(self, model, tokenizer, device="auto", max_length=512, overflow="error"):
        self.device = device_for(device)
        self.model = model.to(self.device).eval()
        self.collator = DecisionCollator(tokenizer, max_length=max_length, overflow=overflow)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(
            DecisionEncoder.from_pretrained(path),
            AutoTokenizer.from_pretrained(Path(path) / "tokenizer"),
            **kwargs,
        )

    @torch.inference_mode()
    def predict_records(self, records, calibrated=True):
        batch = self.collator(records)
        metadata = batch.pop("metadata")
        out = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        temperature = self.model.temperature if calibrated else 1.0
        probs = torch.softmax(out["logits"] / temperature, -1).cpu().numpy()
        logits = out["logits"].cpu().numpy()
        return [
            {
                "probabilities": p[: len(ids)].tolist(),
                "logits": z[: len(ids)].tolist(),
                "option_ids": ids,
                "truncated": trunc,
            }
            for p, z, ids, trunc in zip(probs, logits, metadata["option_ids"], metadata["truncated"])
        ]

    def decide(self, state, question, options, allow_abstain=False, calibrated=True):
        options = copy.deepcopy(options)
        for o in options:
            o.setdefault("id", o.get("label"))
            o.setdefault("description", o.get("label"))
        if allow_abstain and not any(o["id"] == ABSTAIN_ID for o in options):
            options.append(copy.deepcopy(ABSTAIN_OPTION))
        r = self.predict_records([{"state": state, "question": question, "options": options}], calibrated)[0]
        p = np.array(r["probabilities"])
        selected = stable_argmax(p, r["option_ids"])
        ordered = np.sort(p)
        result = {
            "probabilities": dict(zip(r["option_ids"], r["probabilities"])),
            "selected": options[selected]["id"],
            "entropy": float(-np.sum(p * np.log(np.clip(p, 1e-12, 1)))),
            "margin": float(ordered[-1] - ordered[-2]),
            "abstained": bool(options[selected].get("is_abstain")),
            "truncated": r["truncated"],
            "temperature": self.model.temperature if calibrated else 1.0,
            "calibration_fitted": bool(calibrated and self.model.calibration.get("data_sha256")),
        }
        if all("value" in o for o in options):
            result["expected_score"] = float(sum(v * o["value"] for v, o in zip(p, options)))
        return result
