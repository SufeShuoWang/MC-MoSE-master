from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits


FLOAT64_EPSILON = np.finfo(np.float64).eps
RESIDUAL_ZERO_MULTIPLIER = 64.0


def stable_fsum(values: Iterable[float]) -> float:
    return float(math.fsum(float(value) for value in values))


def stable_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(value)
        & np.isfinite(weight)
        & (weight > 0)
    )
    if not valid.any():
        return float("nan")
    denominator = stable_fsum(weight[valid])
    if denominator <= 0:
        return float("nan")
    numerator = stable_fsum(weight[valid] * value[valid])
    return numerator / denominator


def stable_sample_variance(values: np.ndarray, *, ddof: int = 1) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) <= ddof:
        return float("nan")
    mean = stable_fsum(x) / len(x)
    centered_ss = stable_fsum((x - mean) * (x - mean))
    return centered_ss / (len(x) - ddof)


@dataclass(frozen=True)
class DeterministicOLSResult:
    coefficients: np.ndarray
    residuals: np.ndarray
    n_obs: int
    rank: int
    centered_y_sum_of_squares: float
    residual_sum_of_squares: float
    residual_zero_threshold: float
    residual_variance: float
    residual_standard_deviation: float


def deterministic_ols(
    y: np.ndarray,
    predictors: np.ndarray,
    *,
    ddof: int = 1,
) -> DeterministicOLSResult:
    response = np.asarray(y, dtype=np.float64)
    x = np.asarray(predictors, dtype=np.float64)
    if response.ndim != 1 or x.ndim != 2 or len(response) != len(x):
        raise ValueError("y must be one-dimensional and aligned with predictors")
    valid = np.isfinite(response) & np.all(np.isfinite(x), axis=1)
    response = response[valid]
    x = x[valid]
    n_obs = len(response)
    if n_obs == 0:
        raise ValueError("deterministic_ols requires at least one valid observation")
    design = np.column_stack(
        [np.ones(n_obs, dtype=np.float64), x]
    )
    with threadpool_limits(limits=1):
        coefficients, _, rank, _ = scipy.linalg.lstsq(
            design,
            response,
            cond=None,
            lapack_driver="gelsy",
            check_finite=False,
        )
    residuals = np.asarray(response - design @ coefficients, dtype=np.float64)
    mean = stable_fsum(response) / n_obs
    centered_ss = stable_fsum((response - mean) * (response - mean))
    residual_ss = stable_fsum(residuals * residuals)
    zero_threshold = (
        RESIDUAL_ZERO_MULTIPLIER
        * FLOAT64_EPSILON
        * max(centered_ss, 1.0)
    )
    if residual_ss <= zero_threshold:
        residual_ss = 0.0
    variance = (
        residual_ss / (n_obs - ddof)
        if n_obs > ddof
        else float("nan")
    )
    standard_deviation = (
        math.sqrt(variance)
        if np.isfinite(variance) and variance >= 0
        else float("nan")
    )
    return DeterministicOLSResult(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        residuals=residuals,
        n_obs=n_obs,
        rank=int(rank),
        centered_y_sum_of_squares=centered_ss,
        residual_sum_of_squares=residual_ss,
        residual_zero_threshold=zero_threshold,
        residual_variance=variance,
        residual_standard_deviation=standard_deviation,
    )
