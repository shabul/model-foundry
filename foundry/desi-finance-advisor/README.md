<div align="center">

```
██████╗ ███████╗███████╗██╗    ███████╗██╗███╗   ██╗ █████╗ ███╗   ██╗ ██████╗███████╗
██╔══██╗██╔════╝██╔════╝██║    ██╔════╝██║████╗  ██║██╔══██╗████╗  ██║██╔════╝██╔════╝
██║  ██║█████╗  ███████╗██║    █████╗  ██║██╔██╗ ██║███████║██╔██╗ ██║██║     █████╗  
██║  ██║██╔══╝  ╚════██║██║    ██╔══╝  ██║██║╚██╗██║██╔══██║██║╚██╗██║██║     ██╔══╝  
██████╔╝███████╗███████║██║    ██║     ██║██║ ╚████║██║  ██║██║ ╚████║╚██████╗███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝    ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝

                █████╗ ██████╗ ██╗   ██╗██╗███████╗ ██████╗ ██████╗ 
               ██╔══██╗██╔══██╗██║   ██║██║██╔════╝██╔═══██╗██╔══██╗
               ███████║██║  ██║██║   ██║██║███████╗██║   ██║██████╔╝
               ██╔══██║██║  ██║╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██╗
               ██║  ██║██████╔╝ ╚████╔╝ ██║███████║╚██████╔╝██║  ██║
               ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
```

**Your personal finance *bhai* — SFT + DPO fine-tuned on Indian money topics.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97_Model-Mistral--7B--Instruct--v0.2-blue?style=flat-square)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
[![MLX](https://img.shields.io/badge/MLX-Apple_Silicon-000000?style=flat-square&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Training](https://img.shields.io/badge/Training-SFT_+_DPO-orange?style=flat-square)](https://github.com/huggingface/trl)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Hub-shabul-FFD21E?style=flat-square)](https://huggingface.co/shabul)
[![Data](https://img.shields.io/badge/Data-Zerodha_Varsity_+_Synthetic-green?style=flat-square)](https://zerodha.com/varsity/)

*Shabul Abdul · Sr. Data Scientist*

</div>

---

## What is this?

A Mistral-7B-Instruct model fine-tuned in two stages to be a warm, accurate Indian personal finance advisor:

1. **SFT** (Supervised Fine-Tuning) — teaches the Desi Finance Bhai persona and covers 6 finance topics
2. **DPO** (Direct Preference Optimisation) — steers the model to prefer factually accurate, caveat-aware responses over misleading or jargon-heavy ones

**Persona:** English-first with light Hindi (*bhai, yaar, seedha, matlab*) · warm, conversational, never promises guaranteed returns

**Topics covered:** SIPs & Mutual Funds · Home Loans & EMIs · Tax (Old vs New Regime) · Stocks & Zerodha · PPF / NPS / EPF · Gold & Real Estate

---

## Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       DESI FINANCE ADVISOR — PIPELINE                        │
└──────────────────────────────────────────────────────────────────────────────┘

   PHASE 1                    PHASE 2         PHASE 3              PHASE 4
   Data Prep                   SFT            DPO Data              DPO
                                                                         
 ┌─────────────┐           ┌──────────┐    ┌───────────┐         ┌──────────┐
 │ collect_    │           │ prepare_ │    │ generate_ │         │ prepare_ │
 │ data.py     │           │ sft_data │    │ responses │         │ dpo_data │
 │ (Varsity)   │           └────┬─────┘    └─────┬─────┘         └────┬─────┘
 └──────┬──────┘                │                │                    │
        │                  ┌────▼─────┐    ┌─────▼─────┐        ┌────▼─────┐
 ┌──────▼──────┐           │ train_   │    │ label_    │        │ train_   │
 │ clean_and_  │──────────▶│ sft.py   │───▶│ responses │───────▶│ dpo.py   │
 │ format.py   │           │ (MLX)    │    │ (Gemini)  │        │ (MLX)    │
 └──────┬──────┘           └──────────┘    └───────────┘        └──────────┘
        │                       │                                     │
 ┌──────▼──────┐           adapters/sft/                         adapters/dpo/
 │ generate_   │                                                      │
 │ synthetic   │                                               ┌──────▼──────┐
 │ (Gemini)    │                                               │  PHASE 5    │
 └─────────────┘                                               │  win_rate   │
                                                               │  perplexity │
                                                               └──────┬──────┘
                                                                      │
                                                               ┌──────▼──────┐
                                                               │  demo/      │
                                                               │  gradio_app │
                                                               └─────────────┘
```

---

## Data Sources

| Topic | Source |
|-------|--------|
| SIPs & Mutual Funds | [Zerodha Varsity](https://zerodha.com/varsity/module/mutual-funds/) · synthetic |
| Home Loans & EMIs | [Zerodha Varsity](https://zerodha.com/varsity/) · synthetic |
| Tax Old vs New Regime | [Zerodha Varsity](https://zerodha.com/varsity/module/markets-and-taxation/) · synthetic |
| Stocks & Zerodha | [Zerodha Varsity](https://zerodha.com/varsity/module/introduction-to-stock-markets/) · synthetic |
| PPF / NPS / EPF | Zerodha Varsity · synthetic |
| Gold & Real Estate | Synthetic |
| **SFT total** | ~800–1200 instruction–response pairs |
| **DPO triplets** | ~180–200 (prompt, chosen, rejected) |

---

## Before vs After DPO

### Example 1 — SIP vs Lump Sum

**Prompt:** *SIP vs lump sum — which is better right now?*

**SFT response (before DPO):**
> SIP is generally better for most investors because it reduces risk through rupee cost averaging. When markets go down, you buy more units, and when they go up, fewer. This gives you guaranteed protection against market volatility and ensures steady growth over time. Most financial experts recommend SIP for salaried individuals.

**DPO response (after DPO):**
> Yaar, it depends on where you are and where the market is! If you're unsure about market timing — and honestly, nobody really knows — SIP is the smarter move. Rupee cost averaging means you buy more units when markets dip and fewer when they're high, so your average cost stays reasonable over time. Lump sum can work better if you're confident markets are near a bottom, but that's market timing and even pros get it wrong. For a salaried person with regular income, a monthly SIP of ₹5,000–₹10,000 in a Nifty 50 index fund is a solid, boring, and effective starting point. Note: *past performance doesn't guarantee future returns* — SIP averages your cost but doesn't eliminate risk. Consult a SEBI-registered advisor before making big calls.

> **Why DPO wins:** Removed "guaranteed protection" (factually wrong) · added appropriate caveat · more specific actionable advice · warmer tone with Hindi

---

### Example 2 — Tax Regime

**Prompt:** *Should I choose old or new tax regime? I earn 12 LPA.*

**SFT response (before DPO):**
> At 12 LPA, the new tax regime is almost always better because it has lower slab rates. The old regime is complicated with too many deductions and most people don't utilise them fully anyway. Just go with the new regime.

**DPO response (after DPO):**
> Bhai, at 12 LPA it's not automatic — you need to do the maths! The new regime gives you lower slab rates with zero deductions. The old regime lets you claim 80C (₹1.5L), HRA, home loan interest, 80D, and more. Quick rule of thumb: if your total deductions cross ₹3.75 lakh, the old regime saves more tax; below that, new regime wins. At 12 LPA without HRA or home loan interest, the new regime likely wins. But if you're paying rent in a metro or have a home loan, run the numbers — it can flip either way. Use the IT department's online tax calculator or ask your CA before locking in a choice for the year.

> **Why DPO wins:** Removed overconfident "almost always better" · gave the ₹3.75L rule of thumb · mentioned the CA caveat · personalised by situation

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/shabul/model-foundry
cd model-foundry
pip install -r foundry/desi-finance-advisor/requirements.txt
export GOOGLE_API_KEY=your_key_here
```

### Phase 1 — Data Collection

```bash
# Scrape Zerodha Varsity
python foundry/desi-finance-advisor/phase1_data_prep/collect_data.py

# Convert scraped chunks to SFT pairs (Gemini)
python foundry/desi-finance-advisor/phase1_data_prep/clean_and_format.py

# Generate 800+ synthetic QA pairs (Gemini)
python foundry/desi-finance-advisor/phase1_data_prep/generate_synthetic.py
```

### Phase 2 — SFT Training

```bash
# Format data for mlx-lm
python foundry/desi-finance-advisor/phase2_sft/prepare_sft_data.py

# Train (~1200 iterations, ~45 min on M5 24GB)
python foundry/desi-finance-advisor/phase2_sft/train_sft.py
```

### Phase 3 — Build DPO Dataset

```bash
# Generate 5 candidate responses per prompt from the SFT model
python foundry/desi-finance-advisor/phase3_dpo_dataset/generate_responses.py

# Gemini labels chosen vs rejected
python foundry/desi-finance-advisor/phase3_dpo_dataset/label_responses.py
```

### Phase 4 — DPO Training

```bash
# Format triplets for mlx-lm
python foundry/desi-finance-advisor/phase4_dpo/prepare_dpo_data.py

# DPO fine-tune from SFT adapter (~400 iterations, ~20 min on M5 24GB)
python foundry/desi-finance-advisor/phase4_dpo/train_dpo.py
```

### Phase 5 — Evaluate

```bash
# Win rate: DPO vs SFT (Gemini judge)
export GOOGLE_API_KEY=your_key_here
python foundry/desi-finance-advisor/phase5_eval/win_rate.py

# Perplexity comparison
python foundry/desi-finance-advisor/phase5_eval/perplexity.py
```

### Demo

```bash
python foundry/desi-finance-advisor/demo/gradio_app.py
# Open http://localhost:7860
```

### Push to HuggingFace

```bash
export HF_TOKEN=your_token_here
python foundry/desi-finance-advisor/push_to_hub.py --repo shabul/mistral-7b-desi-finance-advisor
```

---

## DPO Preference Rules

When Gemini labels chosen vs rejected, it prefers:

| ✅ Chosen | ❌ Rejected |
|-----------|------------|
| Factually accurate, no misleading claims | Says "guaranteed returns" or overstates certainty |
| Mentions caveats (*"consult a CA"*, *"past returns..."*) | Overconfident financial claims |
| Simple English + warm Hindi-sprinkled tone | Pure cold jargon |
| Actionable and specific to the question | Vague, generic, unhelpful |
| Appropriate length — useful, not rambling | Too terse or walls of text |

---

## Hardware

```
┌─────────────────────────────────────────┐
│  MacBook Pro  ·  Apple M5               │
│  24 GB unified memory                   │
│  mlx-lm  ·  no cloud GPUs required     │
│                                         │
│  SFT  : ~45 min  (1200 iters)          │
│  DPO  : ~20 min  (400 iters)           │
└─────────────────────────────────────────┘
```

Training runs entirely on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm).
The 4-bit quantised Mistral-7B fits comfortably in 24 GB unified memory.

---

## References

| Resource | Link |
|----------|------|
| Mistral-7B-Instruct-v0.2 (4-bit) | [mlx-community/Mistral-7B-Instruct-v0.2-4bit](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit) |
| mlx-lm — LoRA + DPO | [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) |
| Zerodha Varsity | [zerodha.com/varsity](https://zerodha.com/varsity/) |
| Google Gemini API | [ai.google.dev](https://ai.google.dev/) |
| DPO paper | [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) |
| HuggingFace Hub — shabul | [huggingface.co/shabul](https://huggingface.co/shabul) |
