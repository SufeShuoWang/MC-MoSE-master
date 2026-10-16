from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rank_to_half_interval(values: Any, direction: int = 1) -> pd.Series:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    series = pd.Series(values, copy=False, dtype="float64")
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = series.notna() & np.isfinite(series)
    count = int(valid.sum())
    if count < 2:
        return result
    ranks = (series.loc[valid] * direction).rank(method="average", ascending=True)
    result.loc[valid] = (ranks - 1.0) / (count - 1.0) - 0.5
    return result


def weighted_mean_same_valid_set(values: Any, weights: Any) -> float:
    value = pd.Series(values, copy=False, dtype="float64")
    weight = pd.Series(weights, copy=False, dtype="float64")
    valid = value.notna() & weight.notna() & np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return float("nan")
    return float((value.loc[valid] * weight.loc[valid]).sum() / weight.loc[valid].sum())

