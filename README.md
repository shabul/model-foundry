<div align="center">

```
 ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
 ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
 ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
 ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
 ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
 ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
                                              
 ███████╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗ ██████╗ ██╗   ██╗
 ██╔════╝██╔═══██╗██║   ██║████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝
 █████╗  ██║   ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝ ╚████╔╝ 
 ██╔══╝  ██║   ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗  ╚██╔╝  
 ██║     ╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║   ██║   
 ╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝   ╚═╝  
```

**Cast pretrained weights into purpose-built models.**

[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M_Series-000000?style=flat-square&logo=apple&logoColor=white)](https://www.apple.com/mac/)
[![MLX](https://img.shields.io/badge/MLX-LoRA_Fine--tuning-1E40AF?style=flat-square)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Hub-shabul-FFD21E?style=flat-square)](https://huggingface.co/shabul)
[![GitHub](https://img.shields.io/badge/GitHub-model--foundry-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/shabul/model-foundry)

*Shabul Abdul · Sr. Data Scientist*

</div>

---

## Models

| # | Project | Persona | Base Model | Dataset | HF Model | Space | Val Loss |
|---|---------|---------|------------|---------|----------|-------|:--------:|
| 1 | [qwen2.5-dolly](foundry/qwen2.5-dolly/) | General assistant | `Qwen2.5-3B-Instruct` | [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | [shabul/qwen2.5-3b-dolly-finetuned](https://huggingface.co/shabul/qwen2.5-3b-dolly-finetuned) | — | `1.446` |
| 2 | [feynman-explainer](foundry/feynman-explainer/) | Analogy-first teacher | `Qwen2.5-3B-Instruct` | [shabul/feynman-explainer-dataset](https://huggingface.co/datasets/shabul/feynman-explainer-dataset) | [shabul/qwen2.5-3b-feynman-explainer](https://huggingface.co/shabul/qwen2.5-3b-feynman-explainer) | [▶ Space](https://huggingface.co/spaces/shabul/feynman-explainer) | — |
| 3 | [devils-advocate](foundry/devils-advocate/) | Logical counter-arguer | `Gemma-2-9B-IT` | synthetic · 20 concepts | [shabul/gemma-2-9b-devils-advocate](https://huggingface.co/shabul/gemma-2-9b-devils-advocate) | — | — |
| 4 | [sherlock-debugger](foundry/sherlock-debugger/) | Deductive bug detective | `Gemma-2-9B-IT` | synthetic · 20 cases | [shabul/gemma-2-9b-sherlock-debugger](https://huggingface.co/shabul/gemma-2-9b-sherlock-debugger) | — | — |
| 5 | [desi-finance-advisor](foundry/desi-finance-advisor/) | Indian personal finance bhai | `Mistral-7B-Instruct-v0.2` | Zerodha Varsity + synthetic | `shabul/mistral-7b-desi-finance-advisor` *(post-training)* | — | — |

---

## Projects

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚒  FORGE #1 — qwen2.5-dolly                                    ║
║     General instruction following on Dolly-15k                   ║
║     Qwen2.5-3B-Instruct · LoRA rank-8 · 600 iters               ║
╚══════════════════════════════════════════════════════════════════╝
```

Standard LoRA fine-tune on the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) instruction dataset. Serves as the baseline template for all other foundry projects.

- **Model:** [`shabul/qwen2.5-3b-dolly-finetuned`](https://huggingface.co/shabul/qwen2.5-3b-dolly-finetuned)
- **Val loss:** `1.446`

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚒  FORGE #2 — feynman-explainer                                ║
║     Explains anything via analogy, built-up intuition, examples  ║
║     Qwen2.5-3B-Instruct · LoRA rank-8 · synthetic dataset       ║
╚══════════════════════════════════════════════════════════════════╝
```

Fine-tuned to explain complex topics the way Richard Feynman would — starting with a simple analogy, building intuition layer by layer, using everyday language. Evaluated against a Feynman composite score (analogy density, reading ease, sentence length).

- **Model:** [`shabul/qwen2.5-3b-feynman-explainer`](https://huggingface.co/shabul/qwen2.5-3b-feynman-explainer)
- **Dataset:** [`shabul/feynman-explainer-dataset`](https://huggingface.co/datasets/shabul/feynman-explainer-dataset)
- **Live demo:** [huggingface.co/spaces/shabul/feynman-explainer](https://huggingface.co/spaces/shabul/feynman-explainer)
- **Eval:** Feynman composite `47.9 → 82.4` (+34.5) after fine-tuning · see [`eval/report_v1.md`](foundry/feynman-explainer/eval/report_v1.md)

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚒  FORGE #3 — devils-advocate                                  ║
║     Challenges every premise with piercing logical counter-args  ║
║     Gemma-2-9B-IT · LoRA rank-8 · 20 synthetic concepts         ║
╚══════════════════════════════════════════════════════════════════╝
```

Fine-tuned to argue *against* the user's premise — exposing blind spots, edge cases, and unintended consequences in flowing intellectual prose. Dataset generated with Gemini 2.5 Flash Lite.

- **Model:** [`shabul/gemma-2-9b-devils-advocate`](https://huggingface.co/shabul/gemma-2-9b-devils-advocate)
- **Persona phrases:** *"The flaw in that logic is…"*, *"Consider the counter-factual…"*, *"From a purely pragmatic standpoint…"*

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚒  FORGE #4 — sherlock-debugger                               ║
║     Treats every bug as a crime scene — deductive, theatrical    ║
║     Gemma-2-9B-IT · LoRA rank-8 · 20 synthetic bug cases        ║
╚══════════════════════════════════════════════════════════════════╝
```

Fine-tuned to debug code in the voice of Sherlock Holmes — deductive reasoning, dramatic reveals, and a solution hidden inside the narrative. Dataset covers Python, JS, C++, Rust, Go, SQL, CSS, Docker, Git, and more.

- **Model:** [`shabul/gemma-2-9b-sherlock-debugger`](https://huggingface.co/shabul/gemma-2-9b-sherlock-debugger)
- **Persona phrases:** *"Elementary, my dear coder…"*, *"The game is afoot!"*, *"You see, but you do not observe…"*

---

## Structure

```
model-foundry/
│
├── shared/                         # Shared utilities across all projects
│   ├── data_utils.py               # JSONL writing, train/val split
│   ├── hub_utils.py                # LoRA fusion + HF Hub upload
│   └── eval.py                     # Local inference & evaluation
│
├── foundry/
│   ├── qwen2.5-dolly/              # Forge #1 — General instruction tuning
│   │   ├── config/lora_config.yaml
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   ├── push_to_hub.py
│   │   └── MODEL_CARD.md
│   │
│   ├── feynman-explainer/          # Forge #2 — Analogy-first explainer
│   │   ├── config/lora_config.yaml
│   │   ├── data/
│   │   ├── eval/
│   │   │   └── report_v1.md
│   │   ├── space/
│   │   ├── generate_dataset.py
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   ├── push_to_hub.py
│   │   ├── push_dataset_to_hub.py
│   │   ├── MODEL_CARD.md
│   │   └── DATASET_CARD.md
│   │
│   ├── devils-advocate/            # Forge #3 — Logical counter-arguer
│   │   ├── config/lora_config.yaml
│   │   ├── data/
│   │   ├── generate_dataset.py
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   ├── push_to_hub.py
│   │   └── push_dataset_to_hub.py
│   │
│   ├── sherlock-debugger/          # Forge #4 — Deductive bug detective
│   │
│   └── desi-finance-advisor/       # Forge #5 — Indian personal finance (SFT + DPO)
│       ├── config/lora_config.yaml
│       ├── data/
│       ├── generate_dataset.py
│       ├── prepare_data.py
│       ├── train.py
│       ├── push_to_hub.py
│       └── push_dataset_to_hub.py
│
├── test.py                         # Smoke test for published models
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

### Forge #1 · Dolly

```bash
python foundry/qwen2.5-dolly/prepare_data.py
python foundry/qwen2.5-dolly/train.py
python foundry/qwen2.5-dolly/push_to_hub.py --repo shabul/qwen2.5-3b-dolly-finetuned
```

### Forge #2 · Feynman Explainer

```bash
export GOOGLE_API_KEY=...
python foundry/feynman-explainer/generate_dataset.py --workers 8
python foundry/feynman-explainer/prepare_data.py
python foundry/feynman-explainer/train.py
python foundry/feynman-explainer/push_to_hub.py        --repo shabul/qwen2.5-3b-feynman-explainer
python foundry/feynman-explainer/push_dataset_to_hub.py --repo shabul/feynman-explainer-dataset
```

### Forge #3 · Devil's Advocate

```bash
export GOOGLE_API_KEY=...
python foundry/devils-advocate/generate_dataset.py
python foundry/devils-advocate/prepare_data.py
python foundry/devils-advocate/train.py
python foundry/devils-advocate/push_to_hub.py        --repo shabul/gemma-2-9b-devils-advocate
python foundry/devils-advocate/push_dataset_to_hub.py --repo shabul/devils-advocate-dataset
```

### Forge #4 · Sherlock Debugger

```bash
export GOOGLE_API_KEY=...
python foundry/sherlock-debugger/generate_dataset.py
python foundry/sherlock-debugger/prepare_data.py
python foundry/sherlock-debugger/train.py
python foundry/sherlock-debugger/push_to_hub.py        --repo shabul/gemma-2-9b-sherlock-debugger
python foundry/sherlock-debugger/push_dataset_to_hub.py --repo shabul/sherlock-debugger-dataset
```

### Local inference

```bash
python -m shared.eval \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter foundry/feynman-explainer/adapters \
  --prompt "Why does a neural network need non-linear activation functions?"

python test.py   # smoke test for the published Feynman model
```

---

## Adding a new project

```
1.  cp -r foundry/qwen2.5-dolly foundry/<new-project>
2.  Edit config/lora_config.yaml   — swap model + data path
3.  Edit generate_dataset.py       — define your CONCEPTS / CASES + persona prompt
4.  Edit prepare_data.py           — update SYSTEM_PROMPT for the persona
5.  Run the pipeline end-to-end
6.  Add a row to the Models table above
```

---

## Hardware

```
┌─────────────────────────────────────────┐
│  MacBook Pro  ·  Apple M5               │
│  24 GB unified memory                   │
│  mlx-lm  ·  no cloud GPUs required     │
└─────────────────────────────────────────┘
```

All fine-tuning runs locally on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm).
Gemma-2-9B models use the 4-bit quantised `mlx-community/gemma-2-9b-it-4bit` checkpoint to fit in 24 GB.

---

## References

| Resource | Link |
|----------|------|
| MLX — Apple's ML framework | [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) |
| mlx-lm — LoRA fine-tuning | [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) |
| Qwen2.5-3B-Instruct | [huggingface.co/Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| Gemma-2-9B-IT (4-bit) | [huggingface.co/mlx-community/gemma-2-9b-it-4bit](https://huggingface.co/mlx-community/gemma-2-9b-it-4bit) |
| Databricks Dolly-15k | [huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| Google Gemini API | [ai.google.dev](https://ai.google.dev/) |
| HuggingFace Hub — shabul | [huggingface.co/shabul](https://huggingface.co/shabul) |
