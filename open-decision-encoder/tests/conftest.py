import pytest
from transformers import BertConfig, BertModel, BertTokenizerFast

from decision_encoder.modeling import DecisionEncoder


@pytest.fixture
def tokenizer(tmp_path):
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "yes",
        "no",
        "state",
        "question",
        "options",
        "label",
        "description",
        ":",
        ".",
        "the",
        "is",
        "ready",
    ]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    return BertTokenizerFast(vocab_file=str(path))


@pytest.fixture
def model():
    return DecisionEncoder(
        BertModel(
            BertConfig(
                vocab_size=17,
                hidden_size=24,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
            )
        ),
        head_hidden=12,
        dropout=0.0,
    )


@pytest.fixture
def record():
    return {
        "id": "test-1",
        "source": "synthetic",
        "source_split": "train",
        "group_id": "g1",
        "decision_type": "boolean",
        "state": "The state is ready.",
        "question": "Ready?",
        "options": [
            {"id": "yes", "label": "Yes", "description": "Ready"},
            {"id": "no", "label": "No", "description": "Not ready"},
        ],
        "target_probabilities": [0.7, 0.3],
        "metadata": {},
    }
