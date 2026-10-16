from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .final_alpha_schema import FundLabelSchema
from .market_standardizer import MarketStats, standardize_market_sequence


@dataclass(frozen=True)
class MonthSample:
    fund_seq: np.ndarray
    market_seq: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    date: np.int64
    time_index: np.int64

    def as_dict(self) -> dict[str, Any]:
        return {
            "fund_seq": self.fund_seq,
            "market_seq": self.market_seq,
            "target": self.target,
            "mask": self.mask,
            "date": self.date,
            "time_index": self.time_index,
        }


def build_month_indices(
    dates: np.ndarray,
    lookback: int,
    start_date: int,
    end_date: int,
) -> np.ndarray:
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    dates = np.asarray(dates, dtype=np.int64)
    month_indices: list[int] = []
    for time_index in range(int(lookback-1), len(dates)):
        date_value = int(dates[time_index])
        if int(start_date) <= date_value <= int(end_date):
            month_indices.append(time_index)
    return np.asarray(month_indices, dtype=np.int64)


def build_month_sample(
    data: np.ndarray,
    dates: np.ndarray,
    time_index: int,
    lookback: int,
    schema: FundLabelSchema,
    market_stats: MarketStats,
    valid_only: bool = False,
) -> MonthSample:
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    if time_index < lookback - 1:
        raise ValueError(
            f"time_index={time_index} does not have {lookback} history rows."
        )
    if time_index >= data.shape[0]:
        raise ValueError(f"time_index={time_index} is outside data length {data.shape[0]}.")

    window_start = int(time_index) - int(lookback) + 1
    window_stop = int(time_index) + 1

    # Determine valid rows before materializing the relatively large fund-history
    # tensor. This is particularly important for sparse monthly panels.
    target_raw_all = data[int(time_index), :, schema.target_index].astype(np.float32)
    valid_mask_all = np.isfinite(target_raw_all)
    if valid_only:
        fund_rows = np.flatnonzero(valid_mask_all)
    else:
        fund_rows = np.arange(data.shape[1], dtype=np.int64)

    history_rows = np.arange(window_start, window_stop, dtype=np.int64)
    fund_features = np.asarray(schema.fund_feature_indices, dtype=np.int64)
    fund_seq = data[np.ix_(history_rows, fund_rows, fund_features)]
    fund_seq = np.transpose(fund_seq, (1, 0, 2)).astype(np.float32)

    market_seq_one = data[
        window_start:window_stop, 0, schema.market_indices
    ].astype(np.float32)
    market_seq_one = standardize_market_sequence(market_seq_one, market_stats)
    market_seq = np.repeat(market_seq_one[None, :, :], fund_seq.shape[0], axis=0)

    target_raw = target_raw_all[fund_rows]
    mask = valid_mask_all[fund_rows].astype(np.float32)
    target = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    fund_seq = np.nan_to_num(fund_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    market_seq = np.nan_to_num(market_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    return MonthSample(
        fund_seq=fund_seq,
        market_seq=market_seq,
        target=target,
        mask=mask,
        date=np.int64(dates[int(time_index)]),
        time_index=np.int64(time_index),
    )
