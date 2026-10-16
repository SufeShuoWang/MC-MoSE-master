from __future__ import annotations

import numpy as np


def ols_with_intercept(y: np.ndarray, x: np.ndarray, min_obs: int) -> dict[str, np.ndarray | float | int]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n_obs = int(valid.sum())
    if n_obs < min_obs:
        return {"n_obs": n_obs, "alpha": np.nan, "beta": np.full(x.shape[1], np.nan)}
    design = np.column_stack([np.ones(n_obs), x[valid]])
    coefficients = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return {"n_obs": n_obs, "alpha": float(coefficients[0]), "beta": coefficients[1:]}


def residual_sample_scale(y: np.ndarray, x: np.ndarray, min_obs: int, variance: bool = False) -> float:
    fit = ols_with_intercept(y, x, min_obs)
    beta = np.asarray(fit["beta"], dtype=float)
    if not np.all(np.isfinite(beta)):
        return float("nan")
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    residual = y[valid] - (float(fit["alpha"]) + x[valid] @ beta)
    return float(np.var(residual, ddof=1) if variance else np.std(residual, ddof=1))

