# Open Decision Encoder

A bidirectional ModernBERT-base encoder that scores candidates supplied at runtime. One forward pass produces a probability distribution over the current options. The project is separate from the repository's MLX generative-model experiments.

Read [the implementation contract](IMPLEMENTATION.md) for architecture, data policy, decisions and acceptance gates. Measurements and training status live in [reports](reports/).

## Setup

```bash
cd open-decision-encoder
uv sync --locked --python 3.12
uv run python scripts/check_environment.py
uv run pytest -q
```

The tested machine has 24 GiB unified memory. Real MPS forward/backward probes use ModernBERT, SDPA, and reference compilation disabled. `uv.lock` pins dependencies. CUDA and FlashAttention are unnecessary. Set `PYTORCH_ENABLE_MPS_FALLBACK=0` to detect unsupported operations; enable fallback deliberately only when diagnosing a failure.

## Data

```bash
uv run python scripts/download_datasets.py
uv run python scripts/create_taxonomies.py
uv run python scripts/build_dataset.py
uv run python scripts/audit_dataset.py
```

The downloader records immutable source revisions. BANKING77 uses original CSV files pinned to an upstream Git commit because its legacy Hub loader is unsupported. MASSIVE uses pinned converted en-US Parquet files. Typed Decisions preserves complete soft target distributions. Downloaded data/checkpoints are ignored by Git; provenance manifests and taxonomy descriptions are tracked.

Training and development splits are assigned by normalized source-state group before question expansion and candidate augmentation. Original test states are hashed only to exclude development duplicates. Test examples are not transformed until `--include-test` is requested after checkpoint selection. The synthetic generator uses deterministic policy rules; its group identity excludes surface wording and irrelevant tracking IDs.

```bash
uv run python scripts/generate_synthetic.py --seed 42 --count 30000
```

Thirty thousand synthetic renderings are not thirty thousand independent policies. The reports disclose group counts. Arbitrary policy rules are included in state text so their thresholds do not have to be guessed.

## Train, calibrate and evaluate

```bash
uv run python scripts/prepare_smoke.py
uv run python scripts/train.py --config configs/smoke.yaml
# If last-four-layer overfit fails, diagnose and use configs/smoke_full.yaml.
uv run python scripts/train.py --config configs/frozen.yaml
uv run python scripts/train.py --config configs/last4.yaml
# Run full.yaml only if validation evidence justifies it.
uv run python scripts/calibrate.py --checkpoint checkpoints/last4/best --overflow truncate_state
uv run python scripts/build_dataset.py --include-test
uv run python scripts/evaluate.py --checkpoint checkpoints/last4/best \
  --data data/processed/test.jsonl --overflow truncate_state
uv run python scripts/benchmark_mps.py --checkpoint checkpoints/last4/best
```

Use the actual selected checkpoint recorded in the reports, which may differ from the example. Large training refuses to start unless the tiny-overfit gate passes. MPS processes use an exclusive filesystem lock. Every epoch saves the best validation checkpoint and a resumable last checkpoint with optimizer and random state. `--resume checkpoints/<stage>` resumes an identical config at the next epoch; mid-epoch recovery is not implemented.

Model selection uses validation NLL, temperature fitting uses calibration data, and final tests do neither. Metrics include raw/calibrated NLL, Brier, ECE, entropy, abstention, risk/coverage, per-source/type/candidate-count results, and candidate permutation stress tests. Accuracy on soft labels means argmax agreement with the teacher; it is not proof of objective correctness.

## Inference

```python
from decision_encoder.inference import DecisionPredictor

model = DecisionPredictor.from_pretrained("checkpoints/last4/best")
result = model.decide(
    state="Policy: use fallback after two failed attempts. The request timed out twice; fallback is available.",
    question="What should happen next?",
    options=[
        {"id": "retry", "label": "Retry", "description": "Try the same provider again."},
        {"id": "fallback", "label": "Use fallback", "description": "Send the request to the alternate provider."},
    ],
    allow_abstain=True,
)
print(result)
```

The result includes ID-keyed probabilities, selected ID, entropy in nats, top-two margin, abstention flag and applied temperature. Numeric values on every option also produce an expected score. IDs must be unique; omitted IDs default to labels. The reserved `__abstain__` candidate is added only when requested. This is exclusive-choice scoring; independent Boolean propositions need separate calls.

The default overflow behavior raises an error. Explicit `overflow="truncate_state"` preserves all question/option text and reports whether state was shortened. Do not interpret probabilities after truncation as if all evidence was read. Candidate IDs and provenance metadata never enter the encoder. Literal mask tokens in state do not create spurious options.

## Limits

Candidate order augmentation does not make the architecture mathematically invariant. Probabilities depend on the supplied candidate set. One temperature cannot guarantee calibration under domain shift. Synthetic rule families have limited diversity. Runtime-defined options alone do not prove generalization to unseen tasks. Long external benchmark inputs may exceed the supported context; report those exclusions and coverage rather than silently dropping options.

Weights, tokenizer, decision head, temperature and decision config are all needed for inference. Local release packaging should include this code and provenance. No weights are represented as published until an actual upload is verified.

After publication, `DecisionPredictor.from_pretrained("owner/repository", revision="commit-sha")` downloads only the encoder, tokenizer, head and configuration artifacts. Install this package first; no remote Python code is executed by the loader. Pin the Hub revision for reproducible inference.
