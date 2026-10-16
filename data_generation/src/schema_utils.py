from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def validate_columns(actual: Sequence[str], required: Sequence[str], source: str) -> list[str]:
    missing = [column for column in required if column not in actual]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    return missing


def validate_raw_schema(actual: Sequence[str], schema: dict[str, Any], source: str) -> dict[str, Any]:
    missing = [column for column in schema.get("required_columns", []) if column not in actual]
    forbidden_present = [
        column for column in schema.get("forbidden_required_columns", []) if column in actual
    ]
    return {
        "source": source,
        "missing_required_columns": missing,
        "forbidden_columns_present": forbidden_present,
        "valid": not missing,
    }


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    positive_denominator: bool = False,
) -> Any:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0)
    if positive_denominator:
        valid &= den > 0
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    np.divide(num, den, out=out, where=valid)
    return out.item() if out.ndim == 0 else out


def assert_unique(frame: Any, keys: Sequence[str], name: str = "dataframe") -> None:
    if hasattr(frame, "is_duplicated") and frame.__class__.__module__.startswith("polars"):
        duplicate_count = int(frame.select(keys).is_duplicated().sum())
    else:
        duplicate_count = int(frame.duplicated(list(keys), keep=False).sum())
    if duplicate_count:
        raise AssertionError(f"{name} has {duplicate_count} rows on duplicate keys {list(keys)}")


def assert_many_to_one(frame: Any, left_keys: Sequence[str], right_value: str, name: str) -> None:
    if frame.__class__.__module__.startswith("polars"):
        counts = frame.group_by(list(left_keys)).agg(
            frame[right_value].n_unique().alias("_n_values")
        )
        bad = counts.filter(counts["_n_values"] > 1).height
    else:
        bad = int(
            (frame.groupby(list(left_keys), dropna=False)[right_value].nunique(dropna=True) > 1).sum()
        )
    if bad:
        raise AssertionError(f"{name} contains {bad} one-to-many keys for {list(left_keys)}")


def require_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved = [Path(path) for path in paths]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    return resolved

