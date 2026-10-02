# Repository review before Open Decision Encoder

Reviewed source, configuration, project history, model cards, stored evaluation reports, locally available generated datasets and current uncommitted edits at parent commit `ca31272`. New encoder work is isolated; the user's existing modifications are retained.

## What has been built

| Project | Implementation | Recorded evidence |
| --- | --- | --- |
| qwen2.5-dolly | Qwen2.5-3B instruction LoRA on Dolly; shared formatting, training launcher and adapter fusion | Model card/README record validation loss 1.446 |
| feynman-explainer | Gemini synthetic explanations, Qwen LoRA, heuristic style scoring, retrain launcher and Gradio Space | Stored report improves style composite 47.9 to 82.4; source generator currently lists 575 concepts |
| devils-advocate | 20-topic synthetic persona dataset, Gemma-2-9B LoRA, Hub scripts | Training and publication workflow exists; no comparable held-out quality report checked in |
| sherlock-debugger | 20 debugging scenarios, Gemma persona LoRA, Hub scripts | Training and publication workflow exists; no functional debugging benchmark checked in |
| desi-finance-advisor | Synthetic Q&A, Mistral SFT, best-of-five candidates, Gemini selection, RSFT, judging/perplexity and local Gradio demo | Project log records 77 SFT training examples, 48 RSFT training examples, 6/9 judge wins; DPO attempt is historical |

This is a portfolio of local model experiments, not yet a shared tested training framework. Existing utilities target autoregressive MLX models. Open Decision Encoder needs a new PyTorch/MPS model contract, candidate-aware collator and probabilistic evaluation.

## Findings that matter

1. **High — Feynman evaluation includes training prompts.** Eight of the 15 evaluation strings exactly match the checked-in generation concept list: law of large numbers, confidence interval, Doppler effect, Big O, vaccines, pH, opportunity cost and inflation. All eight also occur in the locally available `data/train.jsonl` (517 examples); none occurs in `data/valid.jsonl` (58 examples). The comment claiming held-out prompts is incorrect for this realized dataset. The metric also measures readability/phrase style rather than correctness. See `foundry/feynman-explainer/eval/evaluate.py` and `generate_dataset.py`.
2. **High — finance headline evidence is too small for a reliable improvement claim.** `win_rate_results.json` records six wins versus three across nine validation examples. Requested N=50 does not make this a 50-example experiment. The current uncommitted fix correctly records the actual denominator. Failed or invalid judges still become ties at `phase5_eval/win_rate.py:140`; failures should be reported separately from substantive ties.
3. **High — RSFT validation is not unseen relative to the whole pipeline.** `phase3_dpo_dataset/generate_responses.py:20` reads the SFT training set; `prepare_rsft_data.py` later splits the generated chosen responses. Thus RSFT validation prompts were already in SFT training. A validation loss of 0.269 is not evidence of unseen-prompt generalization across both stages.
4. **Medium — regeneration changes row ordering.** Concurrent generators append `as_completed` results and then seeded row-level shuffling is used to split them. A fixed split seed cannot compensate for nondeterministic completion order. Preserve IDs, sort records and split by source group before augmentation.
5. **Medium — finance demo loses system instruction after the first turn.** `demo/gradio_app.py:53` includes the instruction only when history is empty; stored history contains the user's original message rather than that augmented prompt. Later turns lose the persona/constraints. The custom tuple-history UI also warrants a runtime check against the intended Gradio version.
6. **Medium — perplexity includes prompts and answers.** `phase5_eval/perplexity.py` computes next-token loss over the full formatted example, not just assistant response tokens. The reported 4.2888 versus 6.3922 is not a direct measure of answer quality or factuality.
7. **Medium — reproducibility and documentation lag the experiments.** Root requirements are unpinned, `test.py` is network/model inference rather than an automated test suite, and several docs mix historical DPO terminology with RSFT. The Feynman README table's rank-8 statement differs from the current rank-16 config. Existing Hub upload status was not revalidated during this review.

## Changes to carry forward

Use immutable source IDs/revisions, group-aware split boundaries, explicit data/model hashes, measured hardware gates, automatic contract tests, separate selection/calibration/test data, and evaluation metrics that match the research objective. Preserve failed experiments as evidence. Do not infer objective correctness from a teacher's distribution or a persona-style score.

The new implementation contract records the detailed plan review: actual 24 GiB hardware, explicit abstention on positive and negative examples, protected candidate token budgets, stable option IDs, genuine soft-label loss, order stress tests and benchmark-only DecisionBench handling. Legacy issues above are documented rather than silently mixed into the encoder implementation.

## Local artifact inventory

Feynman: 575 raw / 517 train / 58 validation examples. Devil's Advocate and Sherlock each have 20 raw / 18 train / 2 validation examples. Finance: 86 synthetic, 77 SFT train / 9 validation, 77 candidate sets, 54 judged triplets, 48 RSFT train / 6 validation. The Varsity scrape artifact has zero rows. These are observed local counts, not just README claims.
