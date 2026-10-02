"""Metrics over unpadded candidate distributions, with explicit soft-label semantics."""

import numpy as np


def stable_argmax(probabilities, ids=None):
    p = np.asarray(probabilities)
    tied = np.flatnonzero(p == p.max())
    return int(min(tied, key=lambda i: ids[i])) if ids is not None else int(tied[0])


def metrics(probabilities, targets, options=None):
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("Nonempty aligned predictions and targets required")
    nll, brier, entropy, correct, confidence, expected_correct = [], [], [], [], [], []
    tp = fp = fn = 0
    ordinal_errors = []
    non_abstained = []
    for row, (p, y) in enumerate(zip(probabilities, targets)):
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError("Invalid distribution")
        if not np.isfinite(y).all() or (y < 0).any() or not np.isclose(y.sum(), 1):
            raise ValueError("Invalid target distribution")
        ids = [o["id"] for o in options[row]] if options is not None else None
        selected, gold = stable_argmax(p, ids), stable_argmax(y, ids)
        nll.append(float(-np.sum(y * np.log(np.clip(p, 1e-12, 1.0)))))
        brier.append(float(np.sum((p - y) ** 2)))
        entropy.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))))
        correct.append(float(selected == gold))
        expected_correct.append(float(y[selected]))
        confidence.append(float(p[selected]))
        if options is not None:
            pred_abs = options[row][selected].get("is_abstain", False)
            gold_abs = options[row][gold].get("is_abstain", False)
            if not pred_abs:
                non_abstained.append(row)
            tp += int(pred_abs and gold_abs)
            fp += int(pred_abs and not gold_abs)
            fn += int(not pred_abs and gold_abs)
            if all("value" in o for o in options[row]):
                values = np.array([o["value"] for o in options[row]])
                ordinal_errors.append(float(abs(p @ values - y @ values)))
    confidence, correct = np.array(confidence), np.array(correct)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(
        float((bins == i).mean() * abs(confidence[bins == i].mean() - correct[bins == i].mean()))
        for i in range(15)
        if (bins == i).any()
    )
    order = np.argsort(-confidence, kind="stable")
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(correct) + 1)
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    result = {
        "n": len(correct),
        "accuracy": float(correct.mean()),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece": ece,
        "mean_entropy": float(np.mean(entropy)),
        "soft_expected_accuracy": float(np.mean(expected_correct)),
        "mean_candidate_count": float(np.mean([len(p) for p in probabilities])),
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "abstain_support": tp + fn,
        "abstain_predictions": tp + fp,
        "risk_coverage": [
            {"coverage": (i + 1) / len(correct), "risk": float(risk[i])}
            for i in sorted(set(np.linspace(0, len(correct) - 1, min(20, len(correct)), dtype=int)))
        ],
    }
    result["risk_coverage_definition"] = (
        "Confidence-ranked coverage of all decisions, including abstention as an outcome"
    )
    if options is not None:
        result["non_abstained_coverage"] = len(non_abstained) / len(correct)
        result["non_abstained_risk"] = float(1 - correct[non_abstained].mean()) if non_abstained else None
    if ordinal_errors:
        result["expected_score_mae"] = float(np.mean(ordinal_errors))
    return result


def permutation_metrics(reference, permuted, reference_ids, permuted_ids):
    aligned = np.array([permuted[permuted_ids.index(i)] for i in reference_ids])
    ref = np.asarray(reference)
    return {
        "flip": float(stable_argmax(ref, reference_ids) != stable_argmax(aligned, reference_ids)),
        "kl": float(np.sum(ref * (np.log(np.clip(ref, 1e-12, 1)) - np.log(np.clip(aligned, 1e-12, 1))))),
        "max_probability_deviation": float(np.max(np.abs(ref - aligned))),
    }
