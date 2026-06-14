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

| # | Project | Persona | Base Model | Method | HF Model | Val Loss | Win Rate |
|---|---------|---------|------------|--------|----------|:--------:|:--------:|
| 1 | [qwen2.5-dolly](foundry/qwen2.5-dolly/) | General assistant | `Qwen2.5-3B` | SFT | [↗](https://huggingface.co/shabul/qwen2.5-3b-dolly-finetuned) | `1.446` | — |
| 2 | [feynman-explainer](foundry/feynman-explainer/) | Analogy-first teacher | `Qwen2.5-3B` | SFT + Eval loop | [↗](https://huggingface.co/shabul/qwen2.5-3b-feynman-explainer) | — | +34.5 composite |
| 3 | [devils-advocate](foundry/devils-advocate/) | Logical counter-arguer | `Gemma-2-9B` | SFT | [↗](https://huggingface.co/shabul/gemma-2-9b-devils-advocate) | — | — |
| 4 | [sherlock-debugger](foundry/sherlock-debugger/) | Deductive bug detective | `Gemma-2-9B` | SFT | [↗](https://huggingface.co/shabul/gemma-2-9b-sherlock-debugger) | — | — |
| 5 | [desi-finance-advisor](foundry/desi-finance-advisor/) | Indian finance bhai | `Mistral-7B-Instruct-v0.2` | SFT → RSFT | [↗](https://huggingface.co/shabul/mistral-7b-desi-finance-advisor) | `0.269` | **66.7%** vs SFT |

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

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚒  FORGE #5 — desi-finance-advisor                             ║
║     Indian personal finance advisor with SFT + RSFT alignment   ║
║     Mistral-7B-Instruct-v0.2 · LoRA rank-16 · 2-stage pipeline  ║
╚══════════════════════════════════════════════════════════════════╝
```

The most technically complex forge project. Fine-tunes Mistral-7B into **Desi Finance Bhai** — a warm, factually grounded Indian personal finance advisor who speaks in simple English with light Hindi (*bhai, yaar, seedha, matlab*). Covers SIP/mutual funds, home loans, tax planning (80C/80D), stocks, PPF/NPS/EPF, gold, and real estate.

**Two-stage training pipeline:**

| Stage | Technique | Data | Iters | Val Loss |
|-------|-----------|------|-------|----------|
| Stage 1 — SFT | Supervised Fine-Tuning | 77 Gemini-generated Q&A pairs | 300 | 1.471 |
| Stage 2 — RSFT | Rejection-Sampling Fine-Tuning | 48 Gemini-judged best responses | 300 | 0.269 |

**RSFT** (Rejection-Sampling SFT) is a preference-alignment technique analogous to DPO: the SFT model generates 5 candidate responses per prompt, a Gemini 2.5 judge selects the best, and the model is retrained exclusively on those chosen outputs. This steers responses toward factual accuracy and appropriate financial caveats without requiring a separate reward model.

> *Why RSFT instead of DPO?* — mlx-lm 0.31.3 does not implement `DPODataset`. RSFT with best-of-N filtering achieves ~60–67% of DPO's alignment gain (per AlpacaFarm 2023) at no additional infrastructure cost.

**Evaluation results:**

| Metric | SFT | RSFT | Delta |
|--------|-----|------|-------|
| Perplexity (↓ better) | 4.29 | 6.39 | +2.1 (expected distribution shift) |
| Gemini win rate (↑ better) | 33.3% | **66.7%** | +33.4 pp |

- **Model:** [`shabul/mistral-7b-desi-finance-advisor`](https://huggingface.co/shabul/mistral-7b-desi-finance-advisor)
- **Base:** `mlx-community/Mistral-7B-Instruct-v0.2-4bit` (4-bit NF4, ~7 GB)
- **Training hardware:** Apple M5 · 24 GB unified memory · no cloud GPUs
- **Full pipeline log:** [`foundry/desi-finance-advisor/PROJECT_LOG.md`](foundry/desi-finance-advisor/PROJECT_LOG.md)
- **Demo:** `python foundry/desi-finance-advisor/demo/gradio_app.py`

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
│   ├── feynman-explainer/          # Forge #2 — Analogy-first explainer
│   ├── devils-advocate/            # Forge #3 — Logical counter-arguer
│   ├── sherlock-debugger/          # Forge #4 — Deductive bug detective
│   └── desi-finance-advisor/       # Forge #5 — Indian personal finance (SFT + RSFT)
│       ├── phase1_data_prep/       #   Data: Gemini synthetic generation
│       ├── phase2_sft/             #   Stage 1: SFT training
│       ├── phase3_dpo_dataset/     #   Best-of-5 candidate generation + Gemini judging
│       ├── phase4_dpo/             #   Stage 2: RSFT alignment training
│       ├── phase5_eval/            #   Perplexity + Gemini win-rate evaluation
│       ├── demo/                   #   Gradio streaming chat UI
│       ├── adapters/               #   LoRA adapter checkpoints
│       └── PROJECT_LOG.md          #   Full academic log of every attempt + failure
│
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

### Forge #5 · Desi Finance Advisor (full pipeline)

```bash
export GOOGLE_API_KEY=...
export HF_TOKEN=...

# Stage 1 — SFT
python foundry/desi-finance-advisor/phase1_data_prep/generate_synthetic.py
python foundry/desi-finance-advisor/phase2_sft/prepare_sft_data.py
python foundry/desi-finance-advisor/phase2_sft/train_sft.py

# Preference dataset via best-of-5 candidate generation + Gemini judge
python foundry/desi-finance-advisor/phase3_dpo_dataset/generate_responses.py
python foundry/desi-finance-advisor/phase3_dpo_dataset/label_responses.py

# Stage 2 — RSFT alignment
python foundry/desi-finance-advisor/phase4_dpo/prepare_rsft_data.py
python foundry/desi-finance-advisor/phase4_dpo/train_rsft.py

# Evaluation
python foundry/desi-finance-advisor/phase5_eval/perplexity.py
python foundry/desi-finance-advisor/phase5_eval/win_rate.py

# Demo
python foundry/desi-finance-advisor/demo/gradio_app.py

# Push to Hub
python foundry/desi-finance-advisor/push_to_hub.py --repo shabul/mistral-7b-desi-finance-advisor
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
python foundry/feynman-explainer/push_to_hub.py --repo shabul/qwen2.5-3b-feynman-explainer
```

### Forge #3 · Devil's Advocate

```bash
export GOOGLE_API_KEY=...
python foundry/devils-advocate/generate_dataset.py
python foundry/devils-advocate/prepare_data.py
python foundry/devils-advocate/train.py
python foundry/devils-advocate/push_to_hub.py --repo shabul/gemma-2-9b-devils-advocate
```

### Forge #4 · Sherlock Debugger

```bash
export GOOGLE_API_KEY=...
python foundry/sherlock-debugger/generate_dataset.py
python foundry/sherlock-debugger/prepare_data.py
python foundry/sherlock-debugger/train.py
python foundry/sherlock-debugger/push_to_hub.py --repo shabul/gemma-2-9b-sherlock-debugger
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

All fine-tuning runs locally on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm). 7B models use 4-bit NF4 quantisation to fit within 24 GB; peak training memory for Mistral-7B was 7.2 GB. No cloud GPU cost was incurred across any forge project.

---

## References

| Resource | Link |
|----------|------|
| MLX — Apple's ML framework | [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) |
| mlx-lm — LoRA fine-tuning | [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) |
| LoRA: Low-Rank Adaptation (Hu et al., 2021) | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| DPO: Direct Preference Optimisation (Rafailov et al., 2023) | [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) |
| AlpacaFarm: Rejection Sampling SFT (Dubois et al., 2023) | [arxiv.org/abs/2305.14387](https://arxiv.org/abs/2305.14387) |
| Mistral-7B-Instruct-v0.2 | [huggingface.co/mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| Qwen2.5-3B-Instruct | [huggingface.co/Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| Gemma-2-9B-IT (4-bit) | [huggingface.co/mlx-community/gemma-2-9b-it-4bit](https://huggingface.co/mlx-community/gemma-2-9b-it-4bit) |
| Databricks Dolly-15k | [huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| Google Gemini API | [ai.google.dev](https://ai.google.dev/) |
| HuggingFace Hub — shabul | [huggingface.co/shabul](https://huggingface.co/shabul) |
