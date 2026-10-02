"""Bidirectional candidate scoring; no autoregressive generation."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel


class DecisionEncoder(nn.Module):
    def __init__(self, encoder, head_hidden=256, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.head_hidden, self.dropout = head_hidden, dropout
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.config.hidden_size),
            nn.Linear(encoder.config.hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.frozen = False
        self.temperature = 1.0
        self.calibration = {}

    @classmethod
    def from_base(cls, name="answerdotai/ModernBERT-base", revision=None):
        kwargs = {"reference_compile": False} if "modernbert" in name.lower() else {}
        encoder = AutoModel.from_pretrained(
            name,
            revision=revision,
            attn_implementation="eager" if "deberta" in name.lower() else "sdpa",
            **kwargs,
        )
        return cls(encoder)

    def set_trainable(self, mode):
        if mode not in {"frozen", "last4", "full"}:
            raise ValueError("mode must be frozen, last4, or full")
        for p in self.encoder.parameters():
            p.requires_grad = mode == "full"
        for p in self.head.parameters():
            p.requires_grad = True
        if mode == "last4":
            layers = getattr(self.encoder, "layers", None)
            if layers is None:
                layers = self.encoder.encoder.layer
            for layer in layers[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
            norm = getattr(self.encoder, "final_norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True
        self.frozen = mode == "frozen"
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask, candidate_positions, candidate_mask, targets=None):
        if not candidate_mask.any(dim=-1).all():
            raise ValueError("Every row requires valid candidates")
        if self.frozen:
            with torch.no_grad():
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        gathered = hidden.gather(1, candidate_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits = self.head(gathered).squeeze(-1).float().masked_fill(~candidate_mask, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "probabilities": probabilities}
        if targets is not None:
            logp = torch.log_softmax(logits, dim=-1).masked_fill(~candidate_mask, 0.0)
            output["loss"] = -(targets * logp).sum(-1).mean()
        return output

    def save_pretrained(self, path, tokenizer=None, metadata=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
            str(path / "head.safetensors"),
        )
        (path / "decision_config.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "head_hidden": self.head_hidden,
                    "dropout": self.dropout,
                    "metadata": metadata or {},
                },
                indent=2,
            )
            + "\n"
        )
        (path / "temperature.json").write_text(json.dumps({"temperature": self.temperature}) + "\n")
        if tokenizer is not None:
            tokenizer.save_pretrained(path / "tokenizer")

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)
        cfg = json.loads((path / "decision_config.json").read_text())
        if cfg["schema_version"] != 1:
            raise ValueError("Unsupported checkpoint schema")
        architecture = AutoConfig.from_pretrained(path / "encoder").model_type
        model = cls(
            AutoModel.from_pretrained(
                path / "encoder", attn_implementation="eager" if "deberta" in architecture else "sdpa"
            ),
            cfg["head_hidden"],
            cfg["dropout"],
        )
        model.head.load_state_dict(load_file(str(path / "head.safetensors")), strict=True)
        model.calibration = json.loads((path / "temperature.json").read_text())
        model.temperature = float(model.calibration["temperature"])
        if not 0 < model.temperature < float("inf"):
            raise ValueError("Invalid checkpoint temperature")
        return model
