"""Positive scalar temperature fitted only to a dedicated calibration split."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def fit_temperature(logits, targets):
    if not logits or len(logits) != len(targets):
        raise ValueError("Calibration needs aligned nonempty rows")
    for z, y in zip(logits, targets):
        if len(z) != len(y) or not np.isfinite(z).all() or not np.isfinite(y).all():
            raise ValueError("Invalid calibration rows")

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return float(
            np.mean(
                [
                    -np.dot(y, np.asarray(z) / temperature - logsumexp(np.asarray(z) / temperature))
                    for z, y in zip(logits, targets)
                ]
            )
        )

    result = minimize_scalar(objective, bounds=(-4.6, 4.6), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature optimization failed")
    raw_nll = objective(0.0)
    temperature = float(np.exp(result.x)) if result.fun <= raw_nll else 1.0
    return {
        "temperature": temperature,
        "raw_nll": raw_nll,
        "calibrated_nll": objective(np.log(temperature)),
        "n": len(logits),
    }
