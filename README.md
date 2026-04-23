# model-foundry

> Cast pretrained weights into purpose-built models.

A structured workspace for LoRA fine-tunes, evals, and Hugging Face deployments —
built on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

**Author:** Shabul Abdul, Sr. Data Scientist

---

## Models

| Project | Base Model | Dataset | HF Repo | Notes | Val Loss |
|---|---|---|---|---|
| [qwen2.5-dolly](foundry/qwen2.5-dolly/) | Qwen2.5-3B-Instruct | Dolly-15k | [shabul/qwen2.5-3b-dolly-finetuned](https://huggingface.co/shabul/qwen2.5-3b-dolly-finetuned) | General instruction tuning | 1.446 |
| [feynman-explainer](foundry/feynman-explainer/) | Qwen2.5-3B-Instruct | [shabul/feynman-explainer-dataset](https://huggingface.co/datasets/shabul/feynman-explainer-dataset) | [shabul/qwen2.5-3b-feynman-explainer](https://huggingface.co/shabul/qwen2.5-3b-feynman-explainer) | [Space](https://huggingface.co/spaces/shabul/feynman-explainer) for analogy-first explanations | n/a |

---

## Structure

```
model-foundry/
├── shared/                     # Reusable utilities across all projects
│   ├── data_utils.py           # Dataset formatting, JSONL writing, train/val split
│   ├── hub_utils.py            # LoRA fusion + HF Hub upload
│   └── eval.py                 # Quick inference and evaluation
│
├── foundry/
│   ├── qwen2.5-dolly/          # Project: Qwen2.5-3B on Dolly-15k
│   │   ├── config/
│   │   │   └── lora_config.yaml
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   ├── push_to_hub.py
│   │   └── MODEL_CARD.md
│   └── feynman-explainer/      # Project: Qwen2.5-3B teaching-style explainer
│       ├── config/
│       │   └── lora_config.yaml
│       ├── data/
│       ├── eval/
│       ├── space/
│       ├── generate_dataset.py
│       ├── prepare_data.py
│       ├── train.py
│       ├── push_to_hub.py
│       ├── push_dataset_to_hub.py
│       ├── MODEL_CARD.md
│       └── DATASET_CARD.md
│
├── test.py                     # Minimal local inference smoke test
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
git clone https://github.com/shabul/model-foundry
cd model-foundry
pip install -r requirements.txt
```

Run a project end-to-end (always from repo root):

```bash
python foundry/qwen2.5-dolly/prepare_data.py
python foundry/qwen2.5-dolly/train.py
python foundry/qwen2.5-dolly/push_to_hub.py --repo shabul/<model-name>
```

Feynman explainer flow:

```bash
python foundry/feynman-explainer/generate_dataset.py --workers 8
python foundry/feynman-explainer/prepare_data.py
python foundry/feynman-explainer/train.py
python foundry/feynman-explainer/push_to_hub.py --repo shabul/qwen2.5-3b-feynman-explainer
python foundry/feynman-explainer/push_dataset_to_hub.py --repo shabul/feynman-explainer-dataset
```

Quick inference on any model + adapter:

```bash
python -m shared.eval \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter foundry/qwen2.5-dolly/adapters \
  --prompt "Explain transformers in simple terms."
```

Quick local smoke test for the published Feynman model:

```bash
python test.py
```

---

## Adding a new project

1. `cp -r foundry/qwen2.5-dolly foundry/<new-project>`
2. Edit `config/lora_config.yaml` — swap model + dataset
3. Update `prepare_data.py` if the dataset schema differs (field names, formatting)
4. Run, train, push
5. Add a row to the Models table above

---

## Hardware

All models trained locally on Apple M5 MacBook Pro (24 GB unified memory) using
[mlx-lm](https://github.com/ml-explore/mlx-lm). No cloud GPUs required.
