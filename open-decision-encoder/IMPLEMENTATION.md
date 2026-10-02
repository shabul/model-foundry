# Open Decision Encoder — implementation contract

Date: 2026-10-02. Parent repository: model-foundry, reviewed at ca31272 plus the user's existing uncommitted edits. This document refines `../Open Decision Encoder.md`. The user requested autonomous decisions and overnight execution. Existing project files are preserved.

## Repository review and lessons

The five existing projects are MLX autoregressive LoRA experiments: Dolly instruction tuning; Feynman explanation style and heuristic evaluation; Gemma Devil's Advocate and Sherlock personas; Mistral finance SFT followed by rejection-sampling SFT. Shared code handles JSONL, row-level splitting, MLX inference, adapter fusion and uploads. These are useful workflow precedents, but not an encoder training implementation.

Recorded finance results are 6/9 judge wins and perplexity 4.2888 versus 6.3922. Nine validation prompts are insufficient for a strong generalization claim. Judge failures currently count as ties. RSFT validation prompts originate in the SFT training pool, so RSFT validation loss is not evidence of end-to-end unseen-prompt generalization. The Feynman report measures style; eight of its fifteen evaluation prompts occur verbatim in the realized local training data and do not establish reasoning correctness. Concurrent synthetic generation writes completion-order records, making later seeded row splits unstable across regeneration. Requirements are unpinned; root test.py performs downloaded-model inference, not unit tests. Finance demo drops the system instruction after turn one. Several documentation statements mix historical DPO with active RSFT. No unrelated repair is included in this project.

## Decisions made

- Build an isolated package at `open-decision-encoder/`, as requested in the plan, with its own Python 3.12 environment and uv lock. Do not change the MLX environment.
- Actual machine has 24 GiB unified memory (25,769,803,776 bytes), not the planned 32 GiB. Probe actual MPS availability and memory. Default to float32 first; enable bf16 only after a finite-loss forward/backward probe. Use SDPA and disable reference compilation for ModernBERT. Never claim fallback or performance results without measurement.
- Primary encoder is answerdotai/ModernBERT-base. Start at 512 tokens; physical batch 2, accumulation 8. ModernBERT-large, RL and shared-state cross-attention remain out of scope.
- An option is identified by a stable unique ID. Labels and descriptions are model input; IDs are output keys and metadata, not input features. Public API may default missing IDs to unique labels for the requested convenience signature.
- A decision is mutually exclusive over supplied candidates. Independent multi-label decisions require separate questions. Ordinal choices carry numeric values; expected score is their probability-weighted sum.
- Abstention is explicit and opt-in at inference via a reserved candidate. Train with abstention available on both answerable and unanswerable examples; otherwise its mere presence leaks the answer. Keep 10–15% positive abstention examples, but include negative abstention examples too. Missing evidence and absent gold are distinct metadata reasons. A low-confidence rejection threshold is a separate policy, not proof that no choice is correct.
- Preserve all option text and question text. Reject overlong option/question blocks. Truncate state only with an explicit policy and report truncation; default inference fails closed on overflow. Never silently drop a candidate. Build option-marker indices explicitly so literal mask tokens in user text cannot be mistaken for candidates.
- The shared head does not guarantee permutation invariance: positional encodings and attention see ordering. Train with shuffled options and measure aligned probabilities over 10 permutations, including stable-ID tie breaking.
- Probabilities are conditional on the question, state and candidate set, not universal confidence estimates. Report raw and temperature-scaled probabilities separately; one global temperature is the baseline.

## Canonical schema

Version 1 JSONL records: `id`, `source`, `source_split`, `group_id`, `decision_type` (choice/boolean/ordinal), `state`, `question`, `options` [{id,label,description,is_abstain?,value?}], `target_probabilities`, `metadata`.

Metadata includes revision, source row ID, augmentation seed, rule/generator version, original structured state for synthetic audit, and abstention reason. Metadata and target probabilities are never serialized into model input. Validate finite, nonnegative targets summing to one; unique IDs; >=2 options; matching lengths; finite ordinal values; at most one abstention option. Padded candidates never enter loss or metrics. Avoid 0 * -inf in soft cross entropy.

## Data and leakage controls

Pin source revisions and save licenses, original splits, download checksums, transform version and generated-file hashes in manifests. BANKING77 and MASSIVE cards declare CC-BY-4.0; Typed Decisions declares Apache-2.0. Keep attribution and recipes with the release. Pin revisions before using records. Read benchmark documentation only until final evaluation.

Split raw source groups before question expansion, negative sampling or paraphrases. Derive separate training, model-selection validation, and calibration groups from permitted source training/development data. Preserve official test splits. Split Typed Decisions by whole state, not individual question. Deduplicate normalized state hashes across internal splits. Synthetic split assignment follows underlying rule/state identity, not surface wording; separately hold out templates/domains to test transfer.

BANKING77: train-derived splits, checked 77-label taxonomy, 2 candidate variants per source record, K=2–8, 70% nearby negatives where available. MASSIVE: en-US only, scenario-based nearby negatives, readable taxonomy. Typed Decisions: preserve soft gold distributions and numeric score values; its gold is teacher agreement, not objective truth. Never label a model trained on its train split as zero-shot on its test split.

Synthetic corpus: 30K deterministic rule-labeled records across ten planned domains. Include structured and varied natural-language renderings, nuisance fields, missing evidence and absent-gold cases. Include the applicable policy in input when the answer depends on an arbitrary policy threshold; the model cannot infer an unstated rule reliably. Counterfactual pairs should change relevant facts while keeping style similar. No external benchmark records inform generation.

DecisionBench name is ambiguous; resolve the intended canonical repository from metadata and document it. Keep it entirely evaluation-only regardless of upstream split names. An allowlist of training sources plus source/group/hash checks is required; searching a manifest for a name alone is inadequate contamination protection. Do not use external test outcomes to choose checkpoints.

## Model and package interfaces

`DecisionEncoder`: pretrained AutoModel, shared LayerNorm -> Linear(hidden,256) -> GELU -> Dropout(.1) -> Linear(256,1). Gather contextual hidden states at inserted tokenizer.mask_token_id markers. Pad candidates with a boolean mask, set invalid logits to -inf, compute stable float32 soft-target cross entropy.

`DecisionCollator`: explicit component tokenization, dynamic sequence padding, candidate-position padding, optional training-only permutation, target alignment, truncation counters.

`DecisionPredictor.decide(state,question,options,allow_abstain=False,calibrated=True)`: returns ID-keyed probabilities, selected ID, entropy (nats), top-two margin, abstained flag, optional expected score, and truncation/calibration metadata. Run eval/inference mode and deterministic ID-based tie resolution. Save encoder, tokenizer, head, architecture config and temperature together; restore with strict compatibility checks.

## Gates and execution

0. Environment: resolve lock, probe ModernBERT forward/backward and 100 optimization steps, sequence lengths 256/512/1024, dtype/checkpointing support, memory and throughput. Only one MPS process holds a filesystem lock. Record honest failures. Fallback DeBERTa is tested if ModernBERT fails materially.
1. Contract: schema, deterministic grouped splitting, serialization, candidate masks, soft targets, freezing, save/load and permutation alignment unit tests pass. Use a small random encoder fixture without network access.
2. Tiny overfit: 128 rule-labeled examples, training accuracy >=98%, finite loss, target CE <=0.10 for hard targets. If insufficient, diagnose and retry; never claim the larger experiment passed.
3. Frozen baseline, then last-four-layer tuning. Choose by held-out validation NLL. Full tuning only if validation improvement justifies the cost. Save best checkpoints at validation boundaries, plus resume state. Log dirty working-tree/source digest as well as git commit.
4. Fit positive scalar temperature on calibration split only, with weights frozen. Evaluate in-domain tests, abstention and permutations; external tests run only after checkpoint selection. Report accuracy, NLL, Brier (sum over candidates), 15-bin ECE, entropy, risk/coverage, abstention precision/recall/F1, per-K/per-source/per-type metrics, ordinal expected-score MAE, and aligned permutation KL/deviation/flip rate. Soft-target accuracy is explicitly argmax agreement; probability metrics preserve soft targets.
5. Benchmark synchronized warm MPS inference, p50/p95 latency, examples/sec and allocated/driver memory. Publish actual measurements with lengths/batches/candidate counts. Package local release artifacts and model card; external publication follows only once results meet release criteria.

## Acceptance and remaining uncertainty

A working implementation is not equivalent to a successful trained research model. Complete code, passing tests and real hardware results are independently reportable. No fabricated scores, calibration, speed claims or release status. Numeric quality gates beyond tiny-overfit are validation NLL improvement over uniform and frozen baselines, plus disclosed generalization and abstention tradeoffs. If no checkpoint clears these checks, deliver the reproducible implementation and the failure analysis, not a misleading production claim.

Sources: https://huggingface.co/answerdotai/ModernBERT-base ; https://huggingface.co/docs/transformers/model_doc/modernbert ; https://huggingface.co/datasets/PolyAI/banking77 ; https://huggingface.co/datasets/AmazonScience/massive ; https://huggingface.co/datasets/LocalLLaMA/typed-decisions .

## Measured execution refinements

The initial last-four-layer overfit attempt reached 79.7% accuracy after 30 epochs and did not pass. Continuing that diagnostic checkpoint with full encoder tuning at encoder LR 3e-5 and head LR 3e-4 passed at epoch 19: 98.4375% accuracy and 0.02158 NLL. This diagnostic checkpoint is never used to initialize the main research model; the frozen baseline starts from the original pinned pretrained encoder.

A separate 512-token full-backprop probe measured float32 batch 4/8/16 at approximately 5.1–5.2 examples/s, and bf16 at approximately 6.9–7.3 examples/s. Batch-8 bf16 used about 9.8 GB of driver-allocated memory at sampled points, while batch-16 reached about 14.1 GB. Therefore the main last-four/full configurations use batch 8, accumulation 2, bf16; the frozen encoder uses batch 16. Full tuning does not need gradient checkpointing at the measured batch size. Full runs remain optional and require at least 5% lower validation NLL from last-four tuning compared with frozen. These are sequential stage comparisons, not equal-compute independent ablations.

A separate 10% hash-based reserve of synthetic causal-state groups is evaluation-only. This reduced the final training corpus to 64,908 decisions. The original 30K synthetic renderings remain reproducible. Source grouping takes precedence over balanced row counts, so split sizes differ from naive percentage estimates.
