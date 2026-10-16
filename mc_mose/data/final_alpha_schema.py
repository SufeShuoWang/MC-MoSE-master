from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FUND_FEATURE_COUNT = 58  # CF is excluded.
MARKET_FEATURE_NAMES = ("sentiment", "CFNAI")
TARGET_NAME = "alpha"
MISSING_VALUE = np.float32(-99.99)

@dataclass(frozen=True)
class FundLabelSchema:
    columns: tuple[str, ...]
    fund_feature_indices: np.ndarray
    fund_feature_names: tuple[str, ...]
    market_indices: np.ndarray
    market_feature_names: tuple[str, ...]
    target_index: int
    target_name: str
    alpha_index: int

    @property
    def fund_dim(self) -> int:
        return int(self.fund_feature_indices.size)

    @property
    def market_dim(self) -> int:
        return int(self.market_indices.size)

    @property
    def column_to_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.columns)}

    def as_metadata(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "fund_feature_indices": self.fund_feature_indices.tolist(),
            "fund_feature_names": list(self.fund_feature_names),
            "market_indices": self.market_indices.tolist(),
            "market_feature_names": list(self.market_feature_names),
            "target_index": int(self.target_index),
            "target_name": self.target_name,
            "alpha_index": int(self.alpha_index),
            "fund_dim": self.fund_dim,
            "market_dim": self.market_dim,
        }


@dataclass(frozen=True)
class FundLabelData:
    data: np.ndarray
    dates: np.ndarray
    instruments: np.ndarray
    columns: tuple[str, ...]
    schema: FundLabelSchema

    def as_metadata(self) -> dict[str, Any]:
        metadata = self.schema.as_metadata()
        metadata.update(
            {
                "dates": self.dates.astype(np.int64).tolist(),
                "instruments": [str(value) for value in self.instruments.tolist()],
                "num_months": int(self.data.shape[0]),
                "num_funds": int(self.data.shape[1]),
                "num_columns": int(self.data.shape[2]),
            }
        )
        return metadata


def _decode_columns(columns: np.ndarray) -> tuple[str, ...]:
    result: list[str] = []
    for value in columns.tolist():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace"))
        else:
            result.append(str(value))
    return tuple(result)


def prepare_final_alpha_schema(
    columns: np.ndarray | list[str] | tuple[str, ...],
    target_name: str = TARGET_NAME,
) -> FundLabelSchema:
    if isinstance(columns, np.ndarray):
        column_names = _decode_columns(columns)
    else:
        column_names = tuple(str(value) for value in columns)

    if target_name != TARGET_NAME:
        raise ValueError(
            f"This dataset only supports target_name={TARGET_NAME!r}, "
            f"got {target_name!r}."
        )

    expected_column_count = (
        1 + FUND_FEATURE_COUNT + len(MARKET_FEATURE_NAMES)
    )
    if len(column_names) != expected_column_count:
        raise ValueError(
            f"Expected {expected_column_count} data-channel names, "
            f"got {len(column_names)}."
        )

    if len(set(column_names)) != len(column_names):
        raise ValueError("Duplicate data-channel names are not allowed.")

    column_to_idx = {
        name: idx for idx, name in enumerate(column_names)
    }

    required = [TARGET_NAME, *MARKET_FEATURE_NAMES]
    missing = [
        name for name in required
        if name not in column_to_idx
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")

    if column_to_idx[TARGET_NAME] != 0:
        raise ValueError(
            f"{TARGET_NAME!r} must be data channel 0."
        )

    excluded_names = {
        TARGET_NAME,
        *MARKET_FEATURE_NAMES,
    }
    fund_feature_names = tuple(
        name
        for name in column_names
        if name not in excluded_names
    )

    if len(fund_feature_names) != FUND_FEATURE_COUNT:
        raise ValueError(
            f"Expected {FUND_FEATURE_COUNT} fund features, "
            f"got {len(fund_feature_names)}."
        )

    fund_feature_indices = np.asarray(
        [column_to_idx[name] for name in fund_feature_names],
        dtype=np.int64,
    )
    market_indices = np.asarray(
        [column_to_idx[name] for name in MARKET_FEATURE_NAMES],
        dtype=np.int64,
    )

    return FundLabelSchema(
        columns=column_names,
        fund_feature_indices=fund_feature_indices,
        fund_feature_names=fund_feature_names,
        market_indices=market_indices,
        market_feature_names=MARKET_FEATURE_NAMES,
        target_index=int(column_to_idx[target_name]),
        target_name=target_name,
        alpha_index=int(column_to_idx[TARGET_NAME]),
    )


def load_final_alpha_npz(
    npz_path: str | Path,
    target_name: str = TARGET_NAME,
) -> FundLabelData:
    path = Path(npz_path)

    if target_name != TARGET_NAME:
        raise ValueError(
            f"This dataset only supports target_name={TARGET_NAME!r}, "
            f"got {target_name!r}."
        )

    with np.load(path, allow_pickle=False) as raw:
        required_keys = {"data", "date", "wficn", "names"}
        missing_keys = sorted(
            required_keys.difference(raw.files)
        )
        if missing_keys:
            raise ValueError(
                f"{path} is missing required keys: {missing_keys}."
            )

        data = np.asarray(raw["data"], dtype=np.float32)
        dates = np.asarray(raw["date"], dtype=np.int64)
        instruments = np.asarray(raw["wficn"])
        stored_names_array = np.asarray(raw["names"])

    if data.ndim != 3:
        raise ValueError(
            f"Expected data to have shape (T, N, C), "
            f"got {data.shape}."
        )

    if dates.ndim != 1 or dates.shape[0] != data.shape[0]:
        raise ValueError(
            f"dates must have shape ({data.shape[0]},), "
            f"got {dates.shape}."
        )

    if (
        instruments.ndim != 1
        or instruments.shape[0] != data.shape[1]
    ):
        raise ValueError(
            f"instruments must have shape ({data.shape[1]},), "
            f"got {instruments.shape}."
        )

    if stored_names_array.ndim != 1:
        raise ValueError(
            f"names must be one-dimensional, "
            f"got {stored_names_array.shape}."
        )

    stored_names = _decode_columns(stored_names_array)

    if len(stored_names) + 1 != data.shape[2]:
        raise ValueError(
            "names must describe data channels 1 through C-1: "
            f"len(names)+1={len(stored_names)+1}, "
            f"data.shape[2]={data.shape[2]}."
        )

    # History windows require consecutive calendar months.
    calendar_month = dates % 100
    if np.any((calendar_month < 1) | (calendar_month > 12)):
        raise ValueError("date contains an invalid YYYYMM month.")

    month_serial = (
        (dates // 100) * 12
        + calendar_month
        - 1
    )
    if dates.size > 1 and np.any(np.diff(month_serial) != 1):
        raise ValueError(
            "date must contain strictly consecutive monthly observations."
        )

    # Channel zero is alpha; stored names describe predictor channels.
    columns = (TARGET_NAME, *stored_names)
    schema = prepare_final_alpha_schema(
        columns,
        target_name=target_name,
    )

    # Convert missing sentinels in blocks to limit temporary memory.
    for start in range(0, data.shape[0], 12):
        block = data[start : start + 12]
        block[block == MISSING_VALUE] = np.nan

    return FundLabelData(
        data=data,
        dates=dates,
        instruments=instruments,
        columns=columns,
        schema=schema,
    )
