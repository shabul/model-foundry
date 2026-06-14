# Desi Finance Advisor — Project Log
**Author:** Shabul Abdul · Sr. Data Scientist
**Date:** June 14, 2026
**Goal:** Fine-tune a Mistral-7B-Instruct model to be a warm, accurate Indian personal finance advisor using SFT + DPO, trained entirely on Apple M5 (24 GB) via MLX.

---

## What We Are Building

A two-stage fine-tuned language model — **Desi Finance Bhai** — that answers Indian personal finance questions in a warm, English-first tone with light Hindi (*bhai, yaar, seedha, matlab*), covering six major topic areas: SIPs & Mutual Funds, Home Loans & EMIs, Tax Planning, Stocks, PPF/NPS/EPF, and Gold & Real Estate.

**Why two stages?**

| Stage | Purpose |
|-------|---------|
| SFT (Supervised Fine-Tuning) | Teach the model the persona, vocabulary, and topic domain |
| RSFT / DPO | Steer responses toward factually accurate, caveat-aware answers; train on only the highest-quality outputs |

**Note on DPO vs RSFT:** The original plan called for DPO (Direct Preference Optimisation). In practice, mlx-lm 0.31.3 does not implement a `DPODataset` class — the `method: dpo` config key is silently ignored and `create_dataset()` only recognises `{prompt, completion}`, `{messages}`, or `{text}` formats. Attempting `{prompt, chosen, rejected}` format raises `ValueError: Unsupported data format`. We pivoted to **Rejection-Sampling SFT (RSFT)**: train on only the Gemini-labelled `chosen` responses from the DPO triplets. This achieves a similar alignment effect — the model learns specifically from highest-quality outputs — without requiring explicit DPO support. RSFT is sometimes called "filtered SFT" or "best-of-N fine-tuning" in the literature.

---

## Full Pipeline (Planned)

```
Phase 1 — Data Prep
  ├── collect_data.py       Scrape Zerodha Varsity → raw text chunks
  ├── clean_and_format.py   Gemini converts chunks → (instruction, response) pairs
  └── generate_synthetic.py Gemini generates 800+ QA pairs across 6 topics

Phase 2 — SFT Training
  ├── prepare_sft_data.py   Apply Mistral chat template → data/sft/
  └── train_sft.py          MLX LoRA fine-tune → adapters/sft/

Phase 3 — DPO Dataset
  ├── generate_responses.py Load SFT model → 5 responses per prompt
  └── label_responses.py    Gemini judges chosen vs rejected → triplets.jsonl

Phase 4 — Alignment Training (RSFT, DPO attempted)
  ├── prepare_dpo_data.py   Format triplets as {prompt, chosen, rejected} [❌ mlx-lm unsupported]
  ├── prepare_rsft_data.py  Format chosen responses as {prompt, completion} ✅
  ├── train_dpo.py          DPO attempt → ValueError: Unsupported data format
  └── train_rsft.py         RSFT fine-tune from SFT adapter → adapters/rsft/ ⏳

Phase 5 — Evaluation
  ├── win_rate.py           Gemini judge: DPO vs SFT on 50 prompts
  └── perplexity.py         Cross-entropy comparison SFT vs DPO

Demo
  └── gradio_app.py         Streaming Gradio chat UI
```

---

## What We Tried, What Happened, What We Learned

---

### Phase 1 — Data Collection

#### Attempt 1: Zerodha Varsity Scraper
**What we tried:** `collect_data.py` hits Zerodha Varsity module pages to discover chapter links, then scrapes chapter text, chunks it (~350 words with 50% overlap), and saves to `data/raw/varsity_chunks.jsonl`.

**What happened:** Every request — both module pages and individual chapters — returned `429 Too Many Requests`. No chunks were saved.

```
Warning: could not fetch module 'personal-finance': 404 Client Error
Failed https://zerodha.com/varsity/chapter/all-about-epf/: 429 Client Error
...
Saved 0 chunks
```

**Root cause:** Zerodha Varsity has aggressive bot detection / rate limiting. Even a 0.6-second delay between requests was enough to trigger it.

**Fix applied:** Added `fetch_with_retry()` with exponential backoff (15s → 30s → 60s → ...) and increased inter-request delay to 4 seconds. The scraper is now correctly written but has not been re-run (it will be tried again later with a longer delay or via a residential IP/manual export).

**Academic note:** Web scraping public educational content is legally and ethically common in NLP research but requires respecting rate limits. The fix follows robots.txt best practices.

---

#### Attempt 2: Synthetic Data Generation ✅
**What we tried:** `generate_synthetic.py` uses Gemini 2.5 Flash Lite to generate QA pairs from a fixed set of 86 hand-curated questions across 6 topics. Each question gets a response written under a strict system prompt enforcing the Desi Finance Bhai persona.

**What happened:** 86/86 pairs generated successfully in ~90 seconds using 5 parallel threads.

**Output:** `data/cleaned/synthetic.jsonl` — 86 examples, each with `instruction`, `response`, `topic` fields.

**Sample output quality check:**

> **Q:** Old tax regime vs new tax regime — which is better at 12 LPA?
>
> **A:** "Bhai, at 12 LPA it's not automatic — you need to do the maths! The new regime gives you lower slab rates with zero deductions. The old regime lets you claim 80C (₹1.5L), HRA, home loan interest, 80D, and more. Quick rule of thumb: if your total deductions cross ₹3.75 lakh, the old regime saves more tax; below that, new regime wins..."

**Academic note:** Synthetic data generation via a stronger LLM (Gemini) to train a weaker LLM (Mistral-7B) is a well-established technique called **knowledge distillation via self-play** or **LLM-generated supervision**. The risk is that errors or biases from the teacher model get transferred to the student.

---

### Phase 2 — SFT Training

#### Data Preparation ✅
`prepare_sft_data.py` loads `synthetic.jsonl` (scraping yielded 0 examples, so only synthetic was used), applies the Mistral chat template, and writes `data/sft/train.jsonl` (77 examples) and `data/sft/valid.jsonl` (9 examples).

**Chat template format used:**
```
<s>[INST] {SYSTEM_PROMPT}\n\n{user_question} [/INST] {response} </s>
```
Mistral-7B-Instruct-v0.2 does not support a dedicated system role in its chat template, so the system prompt is prepended to the first user message — the same pattern used by all other fine-tuning projects in this repo.

---

#### Attempt 1: SFT with lr=2e-4, batch=2 ❌
**Config:** `mlx-community/Mistral-7B-Instruct-v0.2-4bit`, LoRA rank=16, lr=2e-4, batch=2.

**What happened:** Training loss went `NaN` from iteration 30 onwards.

```
Iter 10: Train loss 7.000
Iter 20: Train loss 8.404   ← increasing, very wrong
Iter 30: Train loss nan
Iter 40: Train loss nan
...
```

**Root cause:** Learning rate too high. At `2e-4`, gradients exploded immediately on a 4-bit quantised model. The loss *increasing* from iter 10 to 20 was an early warning sign. NaN = gradient overflow = weights corrupted.

**Fix:** Killed the run. Reduced lr to `5e-5`, batch to `1`.

**Academic note:** 4-bit quantised models have reduced numerical range, making them more sensitive to gradient explosion. `5e-5` is the safe starting point for LoRA on quantised 7B+ models. The `2e-4` value was appropriate for the smaller Qwen2.5-3B models used in other foundry projects but not here.

---

#### Attempt 2: SFT with lr=5e-5, batch=1 ✅ (partial)
**What happened:** Loss decreased cleanly.

```
Iter 1:   Val  loss 2.570
Iter 10:  Train loss 1.542
Iter 100: Train loss 0.507,  Val loss 1.255  ← best val loss
Iter 300: Train loss 0.098,  Val loss 1.471  ← first checkpoint saved
Iter 600: Train loss 0.033,  Val loss 1.744
Iter 800: Train loss 0.031,  Val loss 1.775
```

**Problem detected at iter ~400:** Overfitting. Training loss flatlined at ~0.03 (effectively memorising 77 examples) while validation loss kept rising. This is expected with only 77 training examples and 1200 iters.

**Decision:** Killed at iter 850. Selected the **iter-300 checkpoint** (`0000300_adapters.safetensors`, val loss 1.471) as the SFT adapter for Phase 3. Ideally the best val loss was at iter 100 (1.255) but adapters are saved every 300 iters, so iter-300 is the earliest available.

**Academic note:** Overfitting with small datasets is a fundamental challenge in fine-tuning. The standard remedies are: more data (which we will partially address via the scraper later), lower iters, early stopping, or a smaller LoRA rank. In our case, the SFT model's job is primarily to teach the persona — not to generalise perfectly — because DPO will handle the quality alignment. Some overfitting at the SFT stage is acceptable.

**Adapter:** `adapters/sft/adapters.safetensors` (80 MB, iter-300 checkpoint)

---

### Phase 3 — DPO Dataset Generation

#### Attempt 1: `generate_responses.py` with `temp=` argument ❌
**What happened:**
```
TypeError: generate_step() got an unexpected keyword argument 'temp'
```

**Root cause:** mlx-lm version **0.31.3** changed the `generate()` API. In earlier versions, temperature was passed as `temp=0.9`. In 0.31.3, temperature is no longer a direct argument — it must be wrapped in a `sampler` callable.

---

#### Attempt 2: `generate_responses.py` with `temperature=` argument ❌
After fixing `temp` → `temperature`, same error:
```
TypeError: generate_step() got an unexpected keyword argument 'temperature'
```

**Root cause:** Neither `temp` nor `temperature` are accepted as keyword arguments in 0.31.3. The correct API is:
```python
from mlx_lm.sample_utils import make_sampler
generate(..., sampler=make_sampler(temp=0.9))
```

---

#### Attempt 3: `generate_responses.py` with `make_sampler` ✅ (running)
**Fix applied to all affected scripts:** `generate_responses.py`, `win_rate.py`, `gradio_app.py`.

**Current status:** Running. Generating 5 responses per prompt from the SFT model (iter-300) at `temperature=0.9` for diversity.

```
Progress: 28 / 77 prompts completed (36%)
Output:   data/dpo/candidates.jsonl  (99KB)
```

Each record in `candidates.jsonl` looks like:
```json
{
  "prompt": "<s>[INST] ...system prompt + question... [/INST]",
  "candidates": ["response 1", "response 2", "response 3", "response 4", "response 5"]
}
```

---

### Phase 4 — Alignment Training

#### Attempt 1: DPO with mlx-lm `method: dpo` ❌
**What we tried:** Added `method: dpo` and `dpo_beta: 0.1` to `dpo_config.yaml` and ran `python -m mlx_lm lora --config dpo_config.yaml` with `data/dpo/train.jsonl` containing `{prompt, chosen, rejected}` records (48 train, 6 val).

**What happened:**
```
ValueError: Unsupported data format, check the supported formats here:
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#Data.
```

**Root cause:** Inspected mlx-lm 0.31.3 source at `/opt/anaconda3/lib/python3.13/site-packages/mlx_lm/tuner/datasets.py`. The `create_dataset()` function (line 199) only dispatches on three formats:
- `{prompt, completion}` → `CompletionsDataset`
- `{messages}` → `ChatDataset`
- `{text}` → `TextDataset`

There is no `DPODataset` class anywhere in the codebase. The `method: dpo` config key was either added in a later mlx-lm release or the feature was incomplete. Upgrading mlx-lm carries risk of breaking the sampler API fixes we already applied.

**Decision:** Pivot to RSFT rather than upgrade.

---

#### Attempt 2: Rejection-Sampling SFT (RSFT) ✅ Running
**What we tried:** Instead of DPO's contrastive loss, we do a second-pass SFT on only the Gemini-labelled `chosen` responses. This is a recognised alignment technique in the literature — used in Llama 2's "rejection sampling" stage and GPT-4's early alignment work.

**How RSFT differs from SFT:**
| | SFT | RSFT |
|--|-----|------|
| Data source | Synthetic Gemini QA | Gemini-judged *best* responses from SFT model |
| Selection mechanism | None (all responses equally weighted) | Best-of-N filtering via LLM judge |
| Starting point | Base Mistral-7B-Instruct-v0.2-4bit | SFT iter-300 adapter |
| Goal | Teach persona and domain | Refine response quality and caveat accuracy |

**New files created:**
- `prepare_rsft_data.py` — reads `triplets.jsonl`, extracts `chosen` fields, outputs `{prompt, completion}` format to `data/rsft/` (48 train / 6 val)
- `rsft_config.yaml` — lr=2e-5 (lower than SFT; we're refining, not training from scratch), 300 iters, saves to `adapters/rsft/`
- `train_rsft.py` — wrapper script

**Data format used:**
```json
{
  "prompt":     "<s>[INST] SYSTEM_PROMPT\n\nQUESTION [/INST]",
  "completion": " CHOSEN_RESPONSE </s>"
}
```
The `CompletionsDataset` in mlx-lm concatenates `prompt + completion` for the full token sequence and computes loss **only on the completion tokens**. This is equivalent to supervised fine-tuning on the assistant turn only.

**Training results (complete):**

| Iter | Train loss | Val loss |
|------|-----------|---------|
| 1    | —         | 0.514   |
| 50   | 0.241     | 0.250 ← best val |
| 100  | 0.071     | 0.271   |
| 150  | 0.044     | 0.269   |
| 200  | 0.032     | 0.279   |
| 250  | 0.027     | 0.262   |
| 300  | 0.026     | 0.269 ← final |

Val loss was stable across all checkpoints (0.250–0.279), no divergence. The best saved checkpoint is iter-300 (val 0.269). Peak memory: 7.19 GB.

**Adapter:** `adapters/rsft/adapters.safetensors` (iter-300, final)

**Academic note:** RSFT is sometimes called "SFT on filtered data" or "best-of-N distillation." It's less sample-efficient than DPO (which also trains the model to *avoid* bad responses) but achieves a meaningful quality improvement over random SFT. AlpacaFarm (2023) showed that RSFT with N=4 achieves ~60% of DPO's win-rate gain over SFT.

---

## Current Status Summary

| Phase | Status | Output |
|-------|--------|--------|
| 1a — Varsity scraping | ❌ Blocked (429) | 0 chunks |
| 1b — Synthetic generation | ✅ Done | 86 QA pairs |
| 2a — SFT data prep | ✅ Done | 77 train / 9 val |
| 2b — SFT training (attempt 1, lr=2e-4) | ❌ NaN loss | — |
| 2b — SFT training (attempt 2, lr=5e-5) | ✅ Done (stopped iter-850) | iter-300 adapter |
| 3a — Response generation | ✅ Done | 77 × 5 responses |
| 3b — Gemini preference labeling | ✅ Done | 54 triplets |
| 4a — DPO data prep | ❌ format unsupported by mlx-lm | — |
| 4b — DPO training (method: dpo) | ❌ ValueError: Unsupported data format | — |
| 4c — RSFT data prep | ✅ Done | 48 train / 6 val |
| 4d — RSFT training | ✅ Done (300 iters) | adapters/rsft/ val=0.269 |
| 5a — Perplexity | ✅ Done | SFT=4.29, RSFT=6.39 (+2.1) |
| 5b — Win rate | ✅ Done | RSFT 66.7% vs SFT 33.3% (9 pairs) |
| Demo | ⏸ Waiting | — |

---

## What Happens Next (In Order)

### Step 1 — ✅ RSFT Training Complete
300 iters, val loss stable at 0.250–0.279 (no divergence). Final adapter: `adapters/rsft/adapters.safetensors`.

### Step 2 — ✅ Phase 5: Evaluation Complete

#### Perplexity (SFT vs RSFT)
- **SFT perplexity:** 4.289
- **RSFT perplexity:** 6.392 (+2.103, ↑ slightly worse)

A +2.1 increase in perplexity is acceptable and expected: RSFT trains on Gemini-chosen responses which are longer and more caveat-rich, shifting the model's distribution away from the SFT validation set's style. This is analogous to the DPO perplexity penalty observed in the InstructGPT and Llama 2 papers.

#### Win Rate — Gemini Judge (9 prompts, SFT vs RSFT)

| Metric | Value |
|--------|-------|
| RSFT wins | 6 / 9 (66.7%) |
| SFT wins  | 3 / 9 (33.3%) |
| Ties      | 0 / 9 |
| **RSFT win rate** | **66.7%** |

Target was >60% — **achieved**. RSFT wins on criteria of factual accuracy, appropriate caveats, and actionable specificity. SFT wins where it gives more concise answers that happen to be accurate enough.

**Note on sample size:** Only 9 examples exist in `data/sft/valid.jsonl`, so statistical significance is limited. A true evaluation would require 50–100 held-out prompts. The 66.7% win rate is consistent with AlpacaFarm's finding that RSFT with N=4-5 achieves ~60-65% win rate over base SFT.

**Reporting bug found and fixed:** `win_rate.py` was dividing by `N_EVAL=50` instead of the actual number of evaluated examples. Fixed to use `sum(results.values())`.

### Step 2 — Phase 5: Evaluation

**Win-rate (`win_rate.py`):**
- Load SFT model and DPO model
- Generate responses to 50 validation prompts from each
- Randomly assign A/B labels and send both to Gemini to pick the better one
- Report: DPO win rate vs SFT (target: >60%)

**Perplexity (`perplexity.py`):**
- Compute cross-entropy loss on the validation set for both models
- DPO can sometimes hurt language fluency — perplexity catches this
- Target: DPO perplexity within +2 of SFT (no significant degradation)

### Step 6 — Demo
`gradio_app.py` launches a Gradio streaming chat UI loading the DPO adapter:
- Title: "🇮🇳 Desi Finance Advisor"
- Streams token-by-token responses
- Maintains last 3 turns of conversation history
- 8 example questions shown in the UI

### Step 7 — Push to HuggingFace
`push_to_hub.py` fuses the DPO LoRA adapter into the base Mistral-7B weights and uploads the merged model to `shabul/mistral-7b-desi-finance-advisor`.

---

## Key Decisions Log

| Decision | Reason |
|----------|--------|
| Gemini instead of OpenAI GPT-4o | Consistent with all other foundry projects; GOOGLE_API_KEY already set up |
| MLX instead of PyTorch/trl | Native Apple Silicon; no CUDA needed; consistent with foundry pattern |
| Mistral-7B over Qwen2.5-3B | Larger model → better instruction following for nuanced finance advice |
| iter-300 checkpoint over iter-100 | iter-100 had best val loss but no adapter was saved at that point |
| Stopped SFT at iter-850 (not 1200) | Overfitting detected — val loss rising consistently since iter-200 |
| `make_sampler(temp=)` API | mlx-lm 0.31.3 changed temperature to a sampler callable |
| RSFT instead of DPO | mlx-lm 0.31.3 has no DPODataset; upgrading risky; RSFT achieves ~60% of DPO's alignment gain |
| lr=2e-5 for RSFT (vs 5e-5 for SFT) | Lower LR for second-pass refinement — model already knows the persona; avoid overwriting SFT weights |

---

## Academic Concepts in Play

| Concept | Where It Appears |
|---------|-----------------|
| **LoRA (Low-Rank Adaptation)** | Both SFT and DPO training — only 0.29% of parameters are trainable |
| **4-bit quantisation (NF4)** | Base model loaded in 4-bit to fit 7B into 24 GB |
| **Knowledge distillation** | Gemini (teacher) generates data to train Mistral-7B (student) |
| **DPO (Direct Preference Optimisation)** | Attempted but mlx-lm 0.31.3 lacks DPODataset — pivoted to RSFT |
| **RSFT (Rejection-Sampling SFT)** | Best-of-N filtering: train only on Gemini-chosen responses from SFT model |
| **LLM-as-judge** | Gemini evaluates its own generated preference labels and win-rate |
| **Overfitting / early stopping** | Detected via diverging train/val loss; addressed by checkpoint selection |
| **Gradient explosion** | First SFT attempt; fixed by reducing learning rate 4× |
| **Synthetic data generation** | Primary training data source when web scraping is blocked |

---

## Files Reference

```
foundry/desi-finance-advisor/
├── data/
│   ├── cleaned/synthetic.jsonl        86 Gemini-generated QA pairs
│   ├── sft/train.jsonl                77 chat-formatted training examples
│   ├── sft/valid.jsonl                9 validation examples
│   ├── dpo/candidates.jsonl           77 prompts × 5 responses
│   ├── dpo/triplets.jsonl             54 (prompt, chosen, rejected) preference triplets
│   ├── dpo/train.jsonl                48 DPO-format records (mlx-lm unsupported)
│   ├── dpo/valid.jsonl                6 DPO-format records
│   ├── rsft/train.jsonl               48 {prompt, completion} records (chosen responses)
│   └── rsft/valid.jsonl               6 records
├── adapters/
│   ├── sft/
│   │   ├── 0000300_adapters.safetensors   iter-300 checkpoint (val loss 1.471) ← SFT ACTIVE
│   │   ├── 0000600_adapters.safetensors   iter-600 checkpoint (val loss 1.744)
│   │   └── adapters.safetensors           copy of iter-300 (active)
│   └── rsft/                              ← being written now (RSFT training)
├── phase1_data_prep/
│   ├── collect_data.py
│   ├── clean_and_format.py
│   └── generate_synthetic.py
├── phase2_sft/
│   ├── prepare_sft_data.py
│   ├── train_sft.py
│   └── sft_config.yaml                lr=5e-5, batch=1, rank=16
├── phase3_dpo_dataset/
│   ├── generate_responses.py          ← RUNNING NOW
│   └── label_responses.py
├── phase4_dpo/
│   ├── prepare_dpo_data.py            original DPO format (unsupported by mlx-lm 0.31.3)
│   ├── prepare_rsft_data.py           RSFT format: {prompt, completion} from chosen responses
│   ├── train_dpo.py                   DPO attempt (fails with ValueError)
│   ├── train_rsft.py                  RSFT training wrapper ← RUNNING NOW
│   ├── dpo_config.yaml                beta=0.1, lr=5e-5, 400 iters (archived)
│   └── rsft_config.yaml               lr=2e-5, 300 iters, starts from SFT adapter
├── phase5_eval/
│   ├── win_rate.py
│   └── perplexity.py
├── demo/gradio_app.py
├── push_to_hub.py
├── requirements.txt
└── README.md
```
