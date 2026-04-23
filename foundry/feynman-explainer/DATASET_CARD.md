---
pretty_name: Feynman Explainer Synthetic Dataset
language:
- en
license: apache-2.0
task_categories:
- text-generation
- question-answering
tags:
- synthetic
- education
- explanations
- feynman
- instruction-tuning
- mlx
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: raw
    path: data/raw_feynman.jsonl
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/valid.jsonl
dataset_info:
- config_name: default
  features:
  - name: instruction
    dtype: string
  - name: response
    dtype: string
  - name: category
    dtype: string
---

# Feynman Explainer Synthetic Dataset

A compact synthetic instruction dataset for training models to explain concepts in a Feynman-style voice: analogy first, intuition before jargon, and flowing prose instead of bullets.

This dataset was created for the `qwen2.5-3b-feynman-explainer` fine-tune and includes both the raw synthetic examples and the chat-formatted train/validation splits used for MLX LoRA training.

## What is in the repo

- `data/raw_feynman.jsonl`: 575 raw examples with `instruction`, `response`, and `category`
- `data/train.jsonl`: 517 chat-formatted training rows
- `data/valid.jsonl`: 58 chat-formatted validation rows

## Coverage

The prompts span seven subject areas:

| Category | Examples |
|---|---:|
| ML & AI | 119 |
| Statistics | 98 |
| Math | 80 |
| CS | 80 |
| Physics | 78 |
| Biology | 60 |
| Economics | 60 |

Average response length in the raw set is about 310 words.

## Schema

### Raw split

Each row in `raw_feynman.jsonl` has:

```json
{
  "instruction": "What is gradient descent?",
  "response": "Imagine you're standing on a foggy mountain...",
  "category": "ML & AI"
}
```

### Train and validation splits

The prepared files contain one field:

```json
{
  "text": "<|im_start|>system\nYou are a Feynman-style explainer..."
}
```

These rows are already formatted with the Qwen chat template and are ready for MLX LoRA training.

## How it was created

1. A fixed list of concepts was curated across multiple domains.
2. Google Gemini was prompted to generate explanation-style answers for each concept.
3. The raw generations were stored in `raw_feynman.jsonl`.
4. The raw set was converted into chat-template format and split into train/validation data for training.

Because the data is synthetic, it should be treated as style-tuning material rather than a source of ground-truth factual supervision.

## Intended use

This dataset is appropriate for:

- style transfer toward analogy-driven explanations
- instruction tuning for educational assistants
- experiments in teaching-oriented response formatting

This dataset is not appropriate as a benchmark for factual accuracy or as a substitute for expert-reviewed educational content.

## Limitations

- The data is synthetic and may contain factual mistakes or oversimplifications.
- The tone is intentionally stylized and can over-prefer analogy.
- Coverage is broad but shallow; it is designed for explanation style, not domain completeness.

## Related model

This dataset was used to train:

- [`shabul/qwen2.5-3b-feynman-explainer`](https://huggingface.co/shabul/qwen2.5-3b-feynman-explainer)

## Author

Shabul Abdul  
[huggingface.co/shabul](https://huggingface.co/shabul)
