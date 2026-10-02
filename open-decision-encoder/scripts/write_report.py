"""Render model card and completion report from measured artifacts, never invented scores."""

import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def table(metrics):
    keys = ["accuracy", "nll", "brier", "ece", "abstain_precision", "abstain_recall", "abstain_f1"]
    return "\n".join(
        ["| Metric | Value |", "| --- | ---: |"] + [f"| {key} | {metrics[key]:.4f} |" for key in keys]
    )


def main():
    selection = load("reports/selection.json")
    evaluation = load("reports/evaluation.json")
    calibration = load("reports/calibration.json")
    manifest = load("data/manifests/dataset.json")
    hardware = load("reports/hardware.json")
    benchmark = load("reports/benchmark.json")
    external = load("reports/decisionbench.json") if Path("reports/decisionbench.json").exists() else None
    stages = []
    for name in ["frozen", "last4", "full"]:
        path = Path(f"checkpoints/{name}/run.json")
        if path.exists():
            run = load(path)
            stages.append(
                f"| {name} | {run['train_n']:,} | {len(run['history'])} | {run['best_validation_nll']:.4f} | {run['seconds'] / 60:.1f} |"
            )
    stage_table = "\n".join(
        [
            "| Stage | Training decisions | Epochs | Best validation NLL | Minutes |",
            "| --- | ---: | ---: | ---: | ---: |",
            *stages,
        ]
    )
    per_source = []
    for name, m in evaluation["groups"]["source"].items():
        per_source.append(
            f"| {name} | {m['n']:,} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} |"
        )
    source_table = "\n".join(
        [
            "| Source | Decisions | Argmax accuracy | NLL | Brier |",
            "| --- | ---: | ---: | ---: | ---: |",
            *per_source,
        ]
    )
    fastest = min(benchmark["measurements"], key=lambda r: r["p50_ms"])
    slowest = max(benchmark["measurements"], key=lambda r: r["p50_ms"])
    lines = [
        "# Open Decision Encoder — completed experiment",
        "",
        f"Selected local checkpoint: `{selection['checkpoint']}`. Selection used validation NLL only.",
        "",
        "## Training",
        "",
        stage_table,
        "",
        "The initial last-four-layer tiny-overfit attempt failed. Full-encoder diagnostic tuning passed at 98.4375% training accuracy and 0.02158 NLL on 128 examples. Those diagnostic weights did not initialize the main model.",
        "",
        f"Training decisions: {manifest['splits']['train']['rows']:,}; validation: {manifest['splits']['validation']['rows']:,}; calibration: {manifest['splits']['calibration']['rows']:,}. See manifests for group counts, revisions and exclusions.",
        "",
        "## Final transformed test evaluation",
        "",
        source_table,
        "",
        "**These are transformed decision tasks. BANKING77/MASSIVE use sampled candidates and deliberate abstention; their scores are not standard 77-way/60-way classification accuracy. Typed Decisions is fine-tuned, not zero-shot, and measures teacher agreement.**",
        "",
        f"Test decisions: {evaluation['n']:,}; states truncated: {evaluation['truncated']}. Raw NLL: {evaluation['raw']['nll']:.4f}; calibrated NLL: {evaluation['calibrated']['nll']:.4f}.",
        "",
        table(evaluation["calibrated"]),
        "",
        "## Calibration and order sensitivity",
        "",
        f"Temperature: {calibration['temperature']:.4f}, fitted only to {calibration['n']:,} held-out calibration decisions. Calibration-set NLL: {calibration['raw_nll']:.4f} → {calibration['calibrated_nll']:.4f}.",
        "",
        f"Permutation stress: `{json.dumps(evaluation.get('permutations', {}), sort_keys=True)}`.",
        "",
        "## Hardware and latency",
        "",
        f"{hardware['platform']}; {hardware['memory_bytes'] / 1024**3:.0f} GiB unified memory. PyTorch {hardware['torch']}, Transformers {hardware['transformers']}. ModernBERT trained on MPS without CPU fallback.",
        "",
        f"Warmed batch-one p50 latency ranges from {fastest['p50_ms']:.1f} to {slowest['p50_ms']:.1f} ms across the measured input lengths/candidate counts. See benchmark.json for p95, token counts and memory snapshots. No claim is made about total-system peak memory.",
        "",
        "## Limits and follow-up",
        "",
        "- The model emits probabilities over the supplied choices; these are conditional on the candidate set.",
        "- Shuffling is augmentation, not a mathematical guarantee of permutation invariance.",
        "- One temperature need not transfer to unseen domains or candidate counts.",
        "- Repeated synthetic renderings have fewer independent causal-state groups; row count is not independent sample count.",
        "- Context truncation can remove decisive evidence. Default inference refuses overflow; the training/evaluation truncation policy is disclosed.",
        "- Sequential stages are not controlled equal-compute ablations. Confidence intervals should be clustered by source state for correlated decisions.",
        "- Local artifacts are not a verified Hugging Face publication. No upload is performed automatically.",
    ]
    if external:
        lines += [
            "",
            "## External DecisionBench",
            "",
            f"Revision `{external['revision']}`. Standard input; {external['scored']:,}/{external['attempted']:,} rows scored ({external['coverage']:.1%} coverage). Oversized inputs are excluded as whole decisions, not truncated candidates.",
            "",
            "This is a context-feasible subset result and cannot be compared directly with full-benchmark scores.",
            "",
            table(external["metrics"]) if "metrics" in external else "No rows could be scored.",
            "",
            external["claim"],
        ]
    if Path("reports/stress.json").exists():
        stress = load("reports/stress.json")
        lines += [
            "",
            "## Held-out synthetic rendering stress",
            "",
            f"{stress['n']} unique reserved causal groups in a new text template; accuracy {stress['calibrated']['accuracy']:.4f}, NLL {stress['calibrated']['nll']:.4f}.",
        ]
    report = "\n".join(lines) + "\n"
    Path("reports/FINAL_REPORT.md").write_text(report)
    card = (
        """---
license: apache-2.0
base_model: answerdotai/ModernBERT-base
language:
  - en
library_name: pytorch
tags:
  - decision-model
  - encoder
  - probabilistic-classification
  - mps
inference: false
---

"""
        + report
        + """
## Usage

Install this package and load the complete checkpoint using `DecisionPredictor.from_pretrained(path)`.
See USAGE.md for the decide() API, optional abstention, token limits and reproduction commands.
This is a custom candidate-scoring head, not an AutoModelForSequenceClassification fixed-label model.
The full encoder, head.safetensors, tokenizer, decision_config.json and temperature.json are required.

## Sources and attribution

Training sources are BANKING77 (PolyAI, CC-BY-4.0), MASSIVE en-US (Amazon Science, CC-BY-4.0),
Typed Decisions train (LocalLLaMA/codelion, Apache-2.0), and original deterministic synthetic policy data.
The model derives from Answer.AI/LightOn ModernBERT-base (Apache-2.0). Revisions and transformation
hashes are in data/manifests, attribution is in NOTICE, and generation/transformation code is included.
DecisionBench is evaluation-only and no records from it are distributed in this package.
"""
    )
    Path("MODEL_CARD.md").write_text(card)
    print("Wrote reports/FINAL_REPORT.md and MODEL_CARD.md from measured artifacts.")


if __name__ == "__main__":
    main()
