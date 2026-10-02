"""Offline tests against the actual encoder architecture, without downloading weights."""

import torch
from transformers import ModernBertConfig, ModernBertModel

from decision_encoder.collator import DecisionCollator
from decision_encoder.modeling import DecisionEncoder


def tiny_modernbert():
    config = ModernBertConfig(
        vocab_size=17,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        max_position_embeddings=128,
        local_attention=16,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        reference_compile=False,
    )
    config._attn_implementation = "sdpa"
    return DecisionEncoder(ModernBertModel(config), head_hidden=16, dropout=0.0)


def test_last4_freezes_earlier_layers_and_embeddings(record, tokenizer):
    model = tiny_modernbert()
    model.set_trainable("last4")
    assert all(not p.requires_grad for p in model.encoder.embeddings.parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[0].parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[1].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[2].parameters())
    batch = DecisionCollator(tokenizer)([record])
    batch.pop("metadata")
    result = model(**batch)
    expected = -(batch["targets"] * result["probabilities"].log()).sum(-1).mean()
    assert torch.allclose(result["loss"], expected)
    result["loss"].backward()
    assert all(p.grad is None for p in model.encoder.layers[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.layers[-1].parameters())


def test_padding_does_not_change_valid_probabilities(record, tokenizer):
    model = tiny_modernbert().eval()
    batch = DecisionCollator(tokenizer)([record])
    batch.pop("metadata")
    with torch.no_grad():
        baseline = model(**batch)["probabilities"]
        batch["input_ids"] = torch.cat([batch["input_ids"], torch.full((1, 5), 13, dtype=torch.long)], dim=1)
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], torch.zeros((1, 5), dtype=torch.long)], dim=1
        )
        padded = model(**batch)["probabilities"]
    assert torch.allclose(baseline, padded, atol=1e-6)
