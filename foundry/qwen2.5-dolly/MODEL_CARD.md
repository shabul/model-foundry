---
base_model: Qwen/Qwen2.5-3B-Instruct
datasets:
- databricks/databricks-dolly-15k
language:
- en
library_name: mlx
tags:
- lora
- fine-tuned
- instruction-following
- apple-silicon
- mlx-lm
- qwen2.5
license: apache-2.0
---

# qwen2.5-3b-dolly-finetuned

> LoRA fine-tune of Qwen2.5-3B-Instruct on Dolly-15k — trained entirely on a MacBook Pro M5 using Apple MLX.  
> Built by **Shabul Abdul**, Sr. Data Scientist.

---

## What this is

A lightweight instruction-following model fine-tuned with LoRA on [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) — 15k human-written instruction/response pairs spanning brainstorming, classification, QA, summarization, creative writing, and more.

The goal: adapt Qwen2.5-3B-Instruct's output style toward human-annotated, concise, direct answers — without touching 99.8% of the model's weights.

---

## Training snapshot

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Method** | LoRA (rank 8, alpha 16) |
| **Dataset** | `databricks/databricks-dolly-15k` — 13,500 train / 1,500 val |
| **Hardware** | Apple M5 MacBook Pro · 24 GB unified memory |
| **Framework** | `mlx-lm` (Apple MLX) |
| **Training time** | ~13 minutes |
| **Iterations** | 1,000 steps |
| **Batch size** | 2 |
| **Max seq length** | 1,024 tokens |
| **Learning rate** | 1e-4 (constant) |
| **LoRA layers** | 16 |

---

## Parameters — the 0.216% that changed

```
Total model parameters : 3,085,939,200   (3.1 B)
Trainable (LoRA only)  :     6,652,000   (6.7 M)
Frozen                 : 3,079,287,200   (99.784%)
Peak GPU memory        :        17.8 GB
```

Only the LoRA adapter matrices were updated. The base Qwen2.5 weights are untouched — the adapter is a learned "nudge" on top.

---

## Loss curve

| Checkpoint | Val Loss | Train Loss |
|:---:|:---:|:---:|
| Init (iter 1) | 2.725 | — |
| iter 200 | 1.523 | 1.476 |
| iter 400 | 1.482 | 1.509 |
| **iter 600** | **1.446** | 1.552 |
| iter 800 | 1.569 | 1.438 |
| iter 1000 | 1.596 | 1.479 |

**Val loss drop: 2.725 → 1.446 = 46.9% reduction**

The model converged fast — best val loss hit at iter 600. Slight uptick afterward indicates the sweet spot for this dataset/LR combo is around 600–700 steps. The checkpoint at `0000500_adapters.safetensors` (iter 500) is also a strong candidate.

---

## Throughput

```
Speed          ~1.2 – 1.8 it/sec
Tokens/sec     ~485 – 580 tok/sec
Total tokens   388,263 trained tokens
```

MLX's unified memory architecture means the full model + gradients + activations live in the same 24 GB pool shared with the OS and other apps — no PCIe bottleneck.

---

## How to run

```python
from mlx_lm import load, generate

model, tokenizer = load("shabul/qwen2.5-3b-dolly-finetuned")

prompt = "Explain the difference between supervised and unsupervised learning."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=text, max_tokens=512, verbose=True)
```

---

## What changed vs the base model

The base `Qwen2.5-3B-Instruct` is already a capable instruction model. This fine-tune shifts its output distribution toward the Dolly annotation style:

- **Tone**: more direct, less hedging
- **Length**: calibrated to Dolly's avg response length
- **Structure**: plain paragraphs over markdown-heavy formatting
- **Coverage**: particularly tuned on brainstorming, open QA, summarization, and creative writing categories

For highly domain-specific tasks (medical, legal, internal company data), the same pipeline can be re-run with a custom dataset by swapping one line in `prepare_data.py`.

---

## Reproducing this

```bash
git clone https://huggingface.co/shabul/qwen2.5-3b-dolly-finetuned
pip install mlx-lm datasets transformers huggingface_hub

python prepare_data.py   # format Dolly-15k → JSONL
python train.py          # LoRA fine-tune (~13 min on M5)
python push_to_hub.py --repo <your-username>/your-model-name
```

Full source: [`model-fine-tuning`](https://github.com/shabul)

---

## Author

**Shabul Abdul** — Sr. Data Scientist  
[huggingface.co/shabul](https://huggingface.co/shabul)

---

*Fine-tuned on Apple Silicon. No cloud GPUs were harmed.*
