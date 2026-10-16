from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MarketStats:
    mean: np.ndarray
    std: np.ndarray
    train_start: int
    train_end: int
    feature_names: tuple[str, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "market_mean": self.mean.astype(float).tolist(),
            "market_std": self.std.astype(float).tolist(),
            "market_train_start": int(self.train_start),
            "market_train_end": int(self.train_end),
            "market_feature_names": list(self.feature_names),
        }


def make_time_mask(dates: np.ndarray, start_date: int, end_date: int) -> np.ndarray:
    dates = np.asarray(dates, dtype=np.int64)
    return (dates >= int(start_date)) & (dates <= int(end_date))


def extract_market_series(data: np.ndarray, market_indices: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected data with shape (T, N, C), got {data.shape}.")
    if data.shape[1] < 1:
        raise ValueError("data must contain at least one fund.")
    return np.asarray(data[:, 0, market_indices], dtype=np.float32)


def fit_market_stats(
    data: np.ndarray,
    dates: np.ndarray,
    market_indices: np.ndarray,
    train_start: int,
    train_end: int,
    feature_names: tuple[str, ...],
    min_std: float = 1e-6,
) -> MarketStats:
    market_series = extract_market_series(data, market_indices)
    train_mask = make_time_mask(dates, train_start, train_end)
    if not np.any(train_mask):
        raise ValueError(
            f"No market observations found between {train_start} and {train_end}."
        )

    train_block = market_series[train_mask]
    mean = np.nanmean(train_block, axis=0).astype(np.float32)
    std = np.nanstd(train_block, axis=0).astype(np.float32)
    std = np.where(std < float(min_std), 1.0, std).astype(np.float32)

    return MarketStats(
        mean=mean,
        std=std,
        train_start=int(train_start),
        train_end=int(train_end),
        feature_names=tuple(feature_names),
    )


def standardize_market_sequence(market_sequence: np.ndarray, stats: MarketStats) -> np.ndarray:
    sequence = np.asarray(market_sequence, dtype=np.float32)
    return ((sequence - stats.mean) / stats.std).astype(np.float32)
