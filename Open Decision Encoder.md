# Open Decision Encoder

## 1. Mission

Build and release an open-source, encoder-only probabilistic decision model that accepts:

```python
model.decide(
    state="...",
    question="...",
    options=[
        {"label": "...", "description": "..."},
        {"label": "...", "description": "..."},
        ...
    ],
)
```

and returns:

```python
{
    "probabilities": {
        "option_a": 0.73,
        "option_b": 0.19,
        "option_c": 0.08
    },
    "selected": "option_a",
    "entropy": ...,
    "margin": ...
}
```

The model must:

- use a bidirectional encoder rather than autoregressive generation;
- support options defined dynamically at inference time;
- output a full probability distribution;
- support explicit abstention when none of the provided choices is justified;
- be robust to candidate ordering;
- run and train locally on an Apple Silicon Mac with 32 GB unified memory;
- be publishable as weights, code, dataset-building recipe, evaluation code, and Hugging Face model card.

This is NOT an attempt to reproduce proprietary Jev internals.

The research objective is:

> Can a relatively small bidirectional encoder learn a general-purpose, runtime-option-conditioned decision function with useful probability calibration?

---

# 2. Hardware constraint

Target machine:

```text
Apple Silicon MacBook Pro M5
32 GB unified memory
single MPS GPU
```

All core experiments must work locally.

Do not design around:

```text
CUDA
FlashAttention 2
DeepSpeed
FSDP
multi-GPU
H100/A100-specific kernels
```

Use:

```text
PyTorch
Transformers
MPS
bf16 where supported
gradient checkpointing when necessary
dynamic padding
gradient accumulation
```

Set:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

No two agents may launch GPU training jobs simultaneously.

Coding, dataset preparation, tests and analysis may happen concurrently.

GPU training runs must be serialized.

---

# 3. Base model decision

## Primary model

Use:

```text
answerdotai/ModernBERT-base
```

Do NOT start with ModernBERT-large.

Characteristics relevant to us:

```text
~149M parameters
22 transformer layers
hidden dimension = 768
bidirectional encoder
long-context capable
```

Large models are experiments for later, not V1.

## Fallback encoder

Keep:

```text
microsoft/deberta-v3-base
```

as a control/fallback.

Only switch from ModernBERT if the hardware agent demonstrates that ModernBERT causes serious MPS incompatibility or CPU fallback that makes training impractical.

---

# 4. Model architecture

Do NOT implement:

```text
Encoder -> fixed 77-class classifier
```

That would make the model dependent on classes seen during training.

Instead candidates themselves must be input text.

## Input

Use the encoder's existing `[MASK]` token as the option marker.

Example:

```text
State:
The customer says their card was charged twice.

Question:
Which issue best describes this request?

Options:

[MASK]
Label: duplicated_card_transaction
Description: The same card transaction appears more than once.

[MASK]
Label: cash_withdrawal_problem
Description: A problem involving an ATM cash withdrawal.

[MASK]
Label: card_delivery
Description: A question concerning card delivery.

[MASK]
Label: transfer_pending
Description: A bank transfer has not completed yet.
```

Do not create a randomly initialized `[OPT]` token for V1.

Using the pretrained `[MASK]` representation avoids introducing an unnecessary randomly initialized embedding.

## Encoder

Run the full serialized input once:

```text
input
  ↓
ModernBERT
  ↓
H = [h1, h2, ... hL]
```

For every `[MASK]`, obtain its contextual representation:

```text
h_option_1
h_option_2
...
h_option_n
```

## Decision head

Use one shared scoring network:

```text
LayerNorm(768)
    ↓
Linear(768, 256)
    ↓
GELU
    ↓
Dropout(0.1)
    ↓
Linear(256, 1)
```

Same head for every candidate.

Therefore:

```text
z_i = DecisionHead(h_option_i)
```

Then:

```text
p_i = softmax(z)_i
```

The model architecture must support a variable number of candidates per example.

Invalid padded candidate positions must receive `-inf` before softmax.

---

# 5. Unified decision representation

V1 should NOT build separate architectures for:

```text
Boolean
Choice
Score
Routing
Intent
```

Convert everything into candidate scoring.

### Boolean

```text
Question:
Was the operation successful?

Options:
[MASK] Yes
[MASK] No
```

### Choice

```text
Question:
Which tool should run next?

Options:
[MASK] Search
[MASK] Database
[MASK] Calculator
[MASK] Escalate
```

### Ordered score

```text
Question:
How severe is this incident?

Options:
[MASK] Low — minimal operational impact
[MASK] Medium — partial operational degradation
[MASK] High — major operational degradation
[MASK] Critical — immediate containment required
```

The same architecture handles everything.

An ordinal auxiliary loss can be investigated later.

---

# 6. Canonical dataset schema

Every source dataset must be converted to ONE internal format.

Use JSONL or Parquet containing approximately:

```json
{
  "id": "banking77-000001-v1",
  "source": "banking77",
  "source_split": "train",
  "decision_type": "choice",

  "state": "My card got charged twice for the same thing.",

  "question": "Which banking issue best describes this request?",

  "options": [
    {
      "id": "transaction_charged_twice",
      "label": "Transaction charged twice",
      "description": "The same card transaction was charged more than once."
    },
    {
      "id": "card_payment_not_recognised",
      "label": "Unrecognized card payment",
      "description": "The customer does not recognize a card payment."
    }
  ],

  "target_probabilities": [1.0, 0.0],

  "metadata": {
    "augmentation": "hard_negative",
    "original_label": "transaction_charged_twice",
    "candidate_count": 2,
    "abstain_case": false
  }
}
```

Requirements:

```text
len(options) == len(target_probabilities)

sum(target_probabilities) ~= 1

candidate IDs unique

gold candidate exists unless abstain_case=true
```

Validate these automatically.

---

# 7. Dataset strategy

Use four layers of training data.

## Layer A — BANKING77

Source:

```text
PolyAI/banking77
```

Purpose:

```text
fine-grained option discrimination
hard negative learning
runtime-defined choice learning
```

Only source training examples may enter our training corpus.

Preserve the original test set for evaluation.

Transformation:

```text
state = customer utterance

question =
"Which banking issue best describes this request?"

gold option =
original intent
```

Create readable descriptions for all 77 intents.

For example:

```text
card_payment_wrong_exchange_rate

→

label:
Wrong card-payment exchange rate

description:
An incorrect currency exchange rate was applied to a card payment.
```

Create a checked taxonomy file:

```text
data/taxonomies/banking77.json
```

Do not regenerate descriptions during training.

---

# 8. BANKING77 candidate generation

Do NOT always provide all 77 options.

Train the model to handle variable candidate sets.

Generate K where:

```text
2 <= K <= 8
```

most of the time.

Correct option must normally be included.

Negative sampling:

```text
~70% hard negatives
~30% random negatives
```

Hard negatives should come from semantically nearby categories.

Examples:

```text
card-related intent
    ↓
other card-related options

transfer-related intent
    ↓
other transfer options

cash withdrawal
    ↓
other ATM/cash options
```

Produce approximately two differently sampled candidate sets per raw training example.

IMPORTANT:

Perform source train/validation/test splitting BEFORE augmentation.

Never allow differently augmented copies of the same source utterance into different splits.

---

# 9. Layer B — MASSIVE

Source:

```text
AmazonScience/massive
config = en-US
```

Use English for V1.

Purpose:

```text
broader intent understanding
cross-domain decisions
generalization beyond banking
```

MASSIVE provides both scenario and intent.

Use scenario information to generate hard negatives.

Example:

```text
state:
Turn the living room lights down.

question:
Which action best describes the user's request?

options:

dim smart light
turn smart light off
change smart-light color
increase audio volume
```

Candidate negatives should preferentially come from the same scenario.

Create:

```text
data/taxonomies/massive.json
```

with readable descriptions of all 60 intents.

Again:

```text
70% semantically related negatives
30% cross-domain/random negatives
```

---

# 10. Layer C — Typed Decisions training split

Source:

```text
LocalLLaMA/typed-decisions
```

Use ONLY training splits.

Do not touch test examples during model development except during final evaluation.

Typed Decisions contains four workflows:

```text
agent_trace_observability
customer_service
invoice_processing
security_incidents
```

Convert each individual question into one training example.

If one state contains five questions:

```text
state + Q1 → sample 1
state + Q2 → sample 2
state + Q3 → sample 3
state + Q4 → sample 4
state + Q5 → sample 5
```

Unlike BANKING77 and MASSIVE, preserve its full gold probability distribution.

Example:

```text
[0.05, 0.20, 0.70, 0.05]
```

must NOT become:

```text
[0, 0, 1, 0]
```

This dataset will teach probability-distribution behavior directly.

---

# 11. Layer D — our synthetic decision corpus

Build a reproducible synthetic generator.

Do NOT start by asking an LLM to create 100,000 arbitrary examples.

Ground truth should initially come from deterministic rules.

Target approximately:

```text
30K–50K examples
```

for the first meaningful run.

Domains:

```text
tool routing
agent action selection
incident triage
invoice processing
workflow routing
model routing
support-ticket routing
risk severity
retry/fallback decisions
verification decisions
```

Example:

```text
State:
model_response_status = timeout
retry_count = 2
fallback_available = true
request_priority = normal

Question:
What should the gateway do next?

Options:
Retry same model
Use fallback
Abort request
Escalate to human
```

Rule:

```python
if status == "timeout" and retry_count >= 2 and fallback_available:
    answer = "Use fallback"
```

Introduce:

```text
irrelevant fields
distractors
different field ordering
natural-language paraphrases
missing evidence
conflicting evidence
longer contexts
```

The generator itself must be checked into the repository.

We should be able to regenerate the entire dataset from:

```bash
python scripts/generate_synthetic.py --seed 42
```

---

# 12. Abstention training

Approximately 10–15% of transformed training cases should be abstention cases.

Create them deliberately.

Take a valid example:

```text
Gold:
Payments
```

Remove Payments from candidates.

Add:

```text
None of these options is sufficiently supported.
```

as a candidate.

That candidate becomes gold.

Example:

```text
Fraud             0
Loans             0
Card delivery     0
None of these     1
```

Do NOT simply add `None` to every example.

The model needs both:

```text
correct option exists
correct option absent
```

cases.

---

# 13. Option-order robustness

Every time a transformed training example is created:

```text
shuffle candidate order
```

Gold probabilities must be shuffled identically.

Never allow:

```text
correct option usually first
```

or:

```text
abstain usually last
```

as a shortcut.

Later add a dedicated consistency experiment.

Given:

```text
[A, B, C]
→
[0.7, 0.2, 0.1]
```

and permutation:

```text
[C, A, B]
```

the aligned probabilities should remain approximately:

```text
[0.1, 0.7, 0.2]
```

Permutation flip rate becomes an explicit evaluation metric.

---

# 14. Training targets and loss

Support both hard and soft targets.

Hard example:

```text
[0, 1, 0]
```

Soft example:

```text
[0.08, 0.77, 0.15]
```

Primary loss:

```text
soft cross entropy

L = -sum_i y_i log(p_i)
```

This works for both.

Do NOT require reinforcement learning for V1.

Calibration should first come from:

```text
proper probabilistic loss
clean validation data
temperature scaling
```

Optional later experiments:

```text
Brier auxiliary loss
permutation-consistency loss
ordinal loss
```

Do not add them all before obtaining the simple baseline.

---

# 15. Training phases

## Phase 0 — pipeline sanity test

Use 128–500 examples.

Goal:

```text
overfit tiny dataset
```

Expected behavior:

```text
very high training accuracy
training loss approaches near zero
```

If the model cannot deliberately overfit a tiny sample, DO NOT launch larger training.

Debug first.

---

## Phase 1 — frozen encoder

Freeze ModernBERT.

Train only:

```text
decision head
```

Purpose:

Determine how useful the pretrained representations already are.

Initial settings:

```text
max_length = 512
bf16 = true
dynamic padding
head LR ≈ 1e-3
AdamW
```

Tune if required.

---

## Phase 2 — partially unfrozen

Unfreeze approximately the final four transformer layers plus decision head.

Use separate learning rates:

```text
encoder ≈ 1e-5 to 2e-5
decision head ≈ 1e-4 to 3e-4
```

Initial physical batch:

```text
4
```

and use gradient accumulation to obtain an effective batch around 16–32.

---

## Phase 3 — full encoder fine-tuning

Only run after Phase 2 demonstrates meaningful gains.

Start around:

```text
max_length = 512
physical batch = 2–4
gradient accumulation
bf16
gradient checkpointing
```

Start encoder LR around:

```text
1e-5
```

Do NOT jump immediately to 8K sequences.

---

# 16. Context-length curriculum

Mac memory is a constraint.

Use:

```text
Stage A → 256/512 tokens
Stage B → 1024 tokens
Stage C → optionally 2048
```

Only investigate:

```text
4096/8192
```

after everything else works.

Decision datasets generally do not require 8K tokens for initial learning.

Do not waste local compute proving that the architecture can OOM.

---

# 17. Calibration

After training finishes, freeze model weights.

Use validation predictions to learn ONE temperature:

```text
p = softmax(z / T)
```

Choose T by minimizing validation NLL.

Save:

```text
temperature.json
```

with the model.

Inference must support:

```text
raw probabilities
calibrated probabilities
```

Never fit temperature on test benchmarks.

---

# 18. Evaluation suites

## In-domain

Evaluate:

```text
BANKING77 test
MASSIVE en-US test
```

using deterministic candidate-generation seeds.

## Specialized

Evaluate:

```text
Typed Decisions test
```

Only after primary development.

## Zero-shot external benchmark

Evaluate:

```text
DecisionBench
```

DecisionBench is benchmark-only.

Nothing derived from its evaluation examples, labels, state text or candidate combinations may enter training or synthetic generation.

Maintain a contamination test asserting that DecisionBench is absent from every training manifest.

---

# 19. Metrics

Record at minimum:

```text
accuracy
negative log likelihood
Brier score
ECE
mean entropy
candidate count
```

Add:

```text
permutation flip rate
abstain precision
abstain recall
abstain F1
risk-coverage curve
```

For ordered score tasks also report:

```text
expected-score MAE
```

Performance metrics on the Mac:

```text
p50 latency
p95 latency
examples/sec
peak unified-memory use
latency vs candidate count
latency vs input length
```

---

# 20. Critical ablations

Do not run all of these until a strong base model exists.

Eventually compare:

```text
Frozen encoder
vs
last-four-layer fine-tuning
vs
full fine-tuning
```

Then:

```text
random negatives
vs
hard negatives
```

Then:

```text
without abstention
vs
with abstention
```

Then:

```text
without candidate-order augmentation
vs
with candidate-order augmentation
```

Then:

```text
ModernBERT-base
vs
DeBERTa-v3-base
```

The result should tell us WHERE performance actually comes from.

---

# 21. Repository structure

Create:

```text
open-decision-encoder/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
│
├── configs/
│   ├── smoke.yaml
│   ├── frozen.yaml
│   ├── last4.yaml
│   └── full.yaml
│
├── src/
│   └── decision_encoder/
│       ├── __init__.py
│       ├── modeling.py
│       ├── serialization.py
│       ├── collator.py
│       ├── inference.py
│       ├── calibration.py
│       ├── metrics.py
│       └── data/
│           ├── schema.py
│           ├── banking77.py
│           ├── massive.py
│           ├── typed_decisions.py
│           └── synthetic.py
│
├── scripts/
│   ├── check_environment.py
│   ├── download_datasets.py
│   ├── build_dataset.py
│   ├── generate_synthetic.py
│   ├── train.py
│   ├── calibrate.py
│   ├── evaluate.py
│   └── benchmark_mps.py
│
├── data/
│   ├── manifests/
│   └── taxonomies/
│
├── tests/
│   ├── test_schema.py
│   ├── test_serialization.py
│   ├── test_model.py
│   ├── test_collator.py
│   ├── test_permutation.py
│   └── test_no_benchmark_leakage.py
│
└── reports/
    ├── hardware.md
    ├── dataset.md
    ├── training.md
    └── evaluation.md
```

Do not commit downloaded raw datasets or checkpoints unless specifically intended for publication.

---

# 22. Multi-agent organization

The Lead Agent owns integration.

It should delegate work rather than letting multiple agents modify the same files simultaneously.

## Agent A — Hardware / MPS Engineer

Mission:

Verify that ModernBERT-base can train reliably on this Mac.

Tasks:

```text
install current compatible PyTorch + Transformers
verify torch.backends.mps.is_available()
run ModernBERT forward
run backward
run 100 optimization steps
measure memory
measure throughput at lengths 256/512/1024
test bf16
test gradient checkpointing
record CPU-fallback warnings
```

Also test DeBERTa-v3-base as fallback.

Deliver:

```text
reports/hardware.md
scripts/check_environment.py
scripts/benchmark_mps.py
recommended initial batch sizes
pinned working dependency versions
```

This agent MUST NOT begin real training.

---

# 23. Agent B — Dataset and Licensing Scout

Mission:

Own data provenance.

Tasks:

Verify:

```text
dataset source
license
splits
number of rows
fields
original task
allowed redistribution conditions
```

For:

```text
BANKING77
MASSIVE
Typed Decisions
DecisionBench
```

Find 1–3 additional permissively licensed datasets that could improve:

```text
NLI
binary verification
ordinal decisions
```

but DO NOT automatically add them.

Produce a recommendation for Lead review.

Deliver:

```text
data/manifests/sources.yaml
reports/dataset.md
```

Every training example must remain traceable to its original source.

---

# 24. Agent C — Public Dataset Transformation Engineer

Mission:

Convert public datasets into canonical decision examples.

Own:

```text
banking77.py
massive.py
typed_decisions.py
schema.py
```

Tasks:

Implement:

```text
taxonomy descriptions
hard-negative selection
random-negative selection
candidate subsampling
abstention transformations
candidate shuffling
seeded deterministic transforms
```

Critical requirement:

Split first.

Augment second.

Deliver:

```text
working dataset builder
unit tests
dataset statistics
```

No model changes.

---

# 25. Agent D — Synthetic Dataset Engineer

Mission:

Build deterministic decision data.

Implement domains listed earlier.

Every synthetic case must have:

```text
generator seed
underlying structured state
rendered natural-language state
rule that produced target
difficulty level
```

Support:

```text
choice
boolean
ordinal
abstention
```

Deliver:

```text
generate_synthetic.py
synthetic.py
tests
generation report
```

Do not use DecisionBench examples as inspiration at record level.

---

# 26. Agent E — Model Architecture Engineer

Mission:

Implement the encoder decision model.

Own:

```text
modeling.py
serialization.py
collator.py
inference.py
```

Implement:

```text
[MASK] position extraction
variable candidate batching
shared decision head
masked softmax
soft-target loss
hard-target loss
save/load
decide() API
```

Unit tests must verify:

```text
candidate counts vary within same batch
probabilities sum to one
padding never receives probability
gradients reach decision head
encoder freezing actually works
soft targets train correctly
```

This agent must initially work using toy data.

No dependency on finished real datasets.

---

# 27. Agent F — Training Engineer

Mission:

Own reproducible experiments.

Wait until:

```text
model unit tests pass
dataset validators pass
hardware report passes
```

Then execute sequentially:

```text
tiny-overfit
frozen
last-four-layers
full fine-tune
```

Produce:

```text
configs/*.yaml
checkpoints
training logs
reports/training.md
```

Track every experiment with:

```text
git commit
model config
dataset manifest hash
random seed
training duration
metrics
```

Only ONE MPS training process at a time.

---

# 28. Agent G — Evaluation and Calibration Scientist

Mission:

Treat probabilities seriously.

Implement:

```text
accuracy
NLL
Brier
ECE
risk coverage
abstention metrics
permutation robustness
temperature scaling
```

Build candidate-order stress tests.

Example:

Run each evaluation item using 10 candidate permutations.

Measure:

```text
argmax flip rate
mean KL divergence between aligned distributions
maximum probability deviation
```

Deliver:

```text
metrics.py
calibration.py
evaluate.py
reports/evaluation.md
```

This agent MUST NOT modify model training based on test benchmark results without documenting that iteration as benchmark-driven development.

---

# 29. Agent H — Release / Hugging Face Engineer

Start only after model selection.

Deliver:

```text
README
model card
dataset recipe
inference example
license notices
evaluation table
limitations
reproducibility instructions
Hugging Face upload scripts
```

Model card must clearly state:

```text
training sources
evaluation sources
calibration procedure
Mac training hardware
limitations
abstention behavior
known failure modes
```

Release:

```text
weights
tokenizer reference
decision head
temperature
config
training code
evaluation code
```

---

# 30. Execution waves

## Wave 0 — parallel

Run:

```text
Agent A: hardware
Agent B: dataset/licensing
Lead: repository scaffolding + architecture contract
```

Gate:

ModernBERT MPS forward/backward works.

---

## Wave 1 — parallel

After canonical schema is frozen:

```text
Agent C: public dataset transformations
Agent D: synthetic corpus
Agent E: model implementation
```

These agents should communicate through the schema rather than touching one another's code.

Gate:

```text
all unit tests pass
```

---

## Wave 2 — integration

Lead integrates:

```text
model
collator
datasets
training script
```

Run:

```text
128-example deliberate overfit
```

Gate:

Model can nearly memorize the tiny sample.

If not, stop and debug.

---

## Wave 3 — sequential training

Agent F runs:

```text
frozen
→
last-four-layers
→
full fine-tune if justified
```

Choose checkpoints using validation NLL, not test benchmarks.

---

## Wave 4

Agent G performs:

```text
temperature calibration
in-domain evaluation
abstention tests
permutation tests
Typed Decisions test
DecisionBench zero-shot evaluation
```

---

## Wave 5

Run ablations only if they answer meaningful questions.

Do not perform dozens of random hyperparameter searches on the Mac.

---

## Wave 6

Agent H packages selected checkpoint and artifacts for Hugging Face.

---

# 31. Hard project rules

1. Never train on DecisionBench.

2. Never train on Typed Decisions test.

3. Never allow test-set feedback to silently become training data.

4. Split before augmentation.

5. Candidate order must always be randomized.

6. Dataset provenance must remain recoverable.

7. Every experiment must be reproducible from config + seed.

8. Start with ModernBERT-base.

9. Start with 512-token sequences.

10. Do not attempt RL before supervised learning works.

11. Do not attempt ModernBERT-large before base experiments are complete.

12. Do not run simultaneous MPS training jobs.

13. Do not optimize solely for accuracy; calibration matters.

14. Do not introduce proprietary/company data into this repository.

15. Do not claim zero-shot performance on tasks or datasets used during training.

---

# 32. Initial success criteria

V1 is successful when we have:

```text
✓ ModernBERT-base decision model
✓ variable runtime candidates
✓ probability distributions
✓ abstention
✓ option-order augmentation
✓ BANKING77 training
✓ MASSIVE training
✓ Typed Decisions train specialization
✓ synthetic decision training
✓ local M5 training
✓ temperature calibration
✓ permutation stress test
✓ DecisionBench held-out evaluation
✓ reproducible Hugging Face package
```

We do NOT require state reuse/cross-attention architecture for V1.

---

# 33. V2 research direction

Only after V1 exists, investigate:

## Shared-state decision encoder

Instead of repeatedly running:

```text
state + Q1 → encoder
state + Q2 → encoder
state + Q3 → encoder
```

compute:

```text
state
  ↓
encoder
  ↓
H_state
  │
  ├── lightweight Q1 decoder → decision
  ├── lightweight Q2 decoder → decision
  ├── lightweight Q3 decoder → decision
  └── lightweight Q4 decoder → decision
```

Potential architecture:

```text
question/options encoder
        ↓
cross attention

Q = question/options
K = H_state
V = H_state
```

Research question:

> Can one expensive state representation support many calibrated decision queries more efficiently than re-encoding the state for every decision?

This is the likely research contribution after the basic model proves that encoder-based general decision learning works.

---

# 34. FIRST COMMAND TO THE MULTI-AGENT SYSTEM

Begin only with Wave 0.

Do not prematurely implement the entire project.

Create three parallel workstreams:

```text
A — MPS hardware feasibility
B — data/licensing investigation
C — repository/interface specification
```

Return their reports to the Lead.

The Lead must then freeze:

```text
canonical dataset schema
model interface
training constraints
```

before spawning the dataset-transformation and model-implementation agents.

The project should progress through explicit gates rather than having every agent independently build its own interpretation of the system.