from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import polars as pl


CRSP_RETURN_MISSING_CODES = {-44.0, -55.0, -66.0, -77.0, -88.0, -99.0}


def read_csv_header(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def scan_csv_columns(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    n_rows: int | None = None,
    infer_schema_length: int = 1000,
) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        columns=list(columns) if columns else None,
        n_rows=n_rows,
        infer_schema_length=infer_schema_length,
        null_values=["", "NA", "N/A", "NULL", "null"],
        ignore_errors=False,
        try_parse_dates=False,
        encoding="utf8-lossy",
    )
    return frame


def clean_cusip8(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text[:8].zfill(8)


def clean_crsp_return(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed in CRSP_RETURN_MISSING_CODES or parsed <= -1.0:
        return None
    return parsed


LEGAL_SUFFIXES = (
    "limited liability company",
    "limited partnership",
    "incorporated",
    "corporation",
    "company",
    "llc",
    "llp",
    "lp",
    "ltd",
    "inc",
    "corp",
    "co",
)


def normalize_management_name(value: Any) -> str | None:
    """Normalize names while preserving economically identifying words."""
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    changed = True
    while changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            candidate = re.sub(rf"(?:\s+|^){re.escape(suffix)}$", "", text).strip()
            if candidate != text:
                text = candidate
                changed = True
                break
    return text or None


def to_month(value: Any) -> str | None:
    if value is None:
        return None
    parsed = pl.Series([value]).str.to_date(strict=False)
    result = parsed.item()
    return None if result is None else result.strftime("%Y-%m")


def summarize_dataframe(frame: Any, date_columns: Sequence[str] = ()) -> dict[str, Any]:
    if frame.__class__.__module__.startswith("polars"):
        summary: dict[str, Any] = {
            "rows": frame.height,
            "columns": frame.width,
            "dtypes": {name: str(dtype) for name, dtype in frame.schema.items()},
            "missing_rate": {
                name: float(frame[name].null_count() / frame.height) if frame.height else None
                for name in frame.columns
            },
        }
        for column in date_columns:
            if column in frame.columns:
                values = frame[column].cast(pl.Utf8).str.to_date(strict=False)
                summary[f"{column}_min"] = values.min()
                summary[f"{column}_max"] = values.max()
        return summary
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "missing_rate": frame.isna().mean().to_dict(),
    }


def missing_report(frame: Any) -> Any:
    if frame.__class__.__module__.startswith("polars"):
        return pl.DataFrame(
            {
                "column": frame.columns,
                "missing_count": [frame[column].null_count() for column in frame.columns],
                "missing_rate": [
                    frame[column].null_count() / frame.height if frame.height else None
                    for column in frame.columns
                ],
            }
        ).sort("missing_rate", descending=True)
    result = frame.isna().agg(["sum", "mean"]).T.reset_index()
    return result.rename(columns={"index": "column", "sum": "missing_count", "mean": "missing_rate"})

