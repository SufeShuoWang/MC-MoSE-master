from __future__ import annotations

import hashlib
import csv
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tabular_header(path: str | Path, schema: dict[str, Any]) -> list[str]:
    source = Path(path)
    file_format = str(schema.get("file_format", source.suffix.lstrip("."))).lower()
    if file_format in {"xlsx", "xls"}:
        sheet = schema.get("sheet_name")
        return list(pd.read_excel(source, sheet_name=sheet, nrows=0).columns)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def read_tabular_sample(
    path: str | Path,
    schema: dict[str, Any],
    *,
    n_rows: int | None = None,
) -> pd.DataFrame:
    source = Path(path)
    file_format = str(schema.get("file_format", source.suffix.lstrip("."))).lower()
    if file_format in {"xlsx", "xls"}:
        return pd.read_excel(source, sheet_name=schema.get("sheet_name"), nrows=n_rows)
    return pd.read_csv(source, nrows=n_rows, low_memory=False)


def _normalize_month(values: pd.Series, date_format: str) -> pd.Series:
    text = values.astype("string").str.strip()
    if date_format == "%Y%m":
        text = text.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    parsed = pd.to_datetime(text, format=date_format, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def _clean_source(
    frame: pd.DataFrame,
    *,
    date_column: str,
    date_format: str,
    value_column: str,
    output_column: str,
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {date_column, value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")
    clean = frame[[date_column, value_column]].copy()
    clean["observation_month"] = _normalize_month(clean[date_column], date_format)
    clean[output_column] = pd.to_numeric(clean[value_column], errors="coerce")
    bad_dates = clean.loc[clean["observation_month"].isna(), [date_column, value_column]].copy()
    if not bad_dates.empty:
        bad_dates.insert(0, "source", source_name)
    clean = clean.loc[clean["observation_month"].notna(), ["observation_month", output_column]]
    duplicates = clean.loc[
        clean.duplicated(["observation_month"], keep=False),
        ["observation_month", output_column],
    ].copy()
    if not duplicates.empty:
        duplicates.insert(0, "source", source_name)
    if not duplicates.empty:
        raise ValueError(f"{source_name} contains duplicate observation_month values.")
    return clean.sort_values("observation_month").reset_index(drop=True), bad_dates


def build_monthly_macro_features(
    sentiment_path: str | Path,
    cfnai_path: str | Path,
    *,
    sentiment_sheet: str = "DATA",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sentiment_source = pd.read_excel(
        sentiment_path,
        sheet_name=sentiment_sheet,
        usecols=["yearmo", "SENT"],
    )
    cfnai_source = pd.read_csv(
        cfnai_path,
        usecols=["observation_date", "cfnai"],
        low_memory=False,
    )
    sentiment, sentiment_bad_dates = _clean_source(
        sentiment_source,
        date_column="yearmo",
        date_format="%Y%m",
        value_column="SENT",
        output_column="Sentiment",
        source_name="Sentiment",
    )
    cfnai, cfnai_bad_dates = _clean_source(
        cfnai_source,
        date_column="observation_date",
        date_format="%Y-%m-%d",
        value_column="cfnai",
        output_column="CFNAI",
        source_name="CFNAI",
    )
    start = min(sentiment["observation_month"].min(), cfnai["observation_month"].min())
    end = max(sentiment["observation_month"].max(), cfnai["observation_month"].max())
    axis = pd.DataFrame({"observation_month": pd.date_range(start, end, freq="MS")})
    combined = (
        axis.merge(sentiment, on="observation_month", how="left", validate="one_to_one")
        .merge(cfnai, on="observation_month", how="left", validate="one_to_one")
        .sort_values("observation_month")
        .reset_index(drop=True)
    )
    if combined.duplicated(["observation_month"]).any():
        raise AssertionError("Combined macro table is not unique by observation_month.")

    coverage_rows = []
    for name, source, value in (
        ("Sentiment", sentiment, "Sentiment"),
        ("CFNAI", cfnai, "CFNAI"),
    ):
        coverage_rows.append(
            {
                "source": name,
                "source_rows": len(source),
                "observation_month_min": source["observation_month"].min(),
                "observation_month_max": source["observation_month"].max(),
                "valid_value_count": int(source[value].notna().sum()),
                "missing_value_count": int(source[value].isna().sum()),
                "value_min": source[value].min(),
                "value_max": source[value].max(),
            }
        )
    missing_rows = []
    for name in ("Sentiment", "CFNAI"):
        for month in combined.loc[combined[name].isna(), "observation_month"]:
            missing_rows.append(
                {"source": name, "observation_month": month, "reason": "missing_or_outside_source_coverage"}
            )
    audits = {
        "coverage": pd.DataFrame(coverage_rows),
        "duplicate_months": pd.DataFrame(
            columns=["source", "observation_month", "value"]
        ),
        "missing_months": pd.DataFrame(missing_rows),
        "input_hashes": pd.DataFrame(
            [
                {
                    "source": "Sentiment",
                    "path": str(Path(sentiment_path)),
                    "sha256": sha256_file(sentiment_path),
                },
                {
                    "source": "CFNAI",
                    "path": str(Path(cfnai_path)),
                    "sha256": sha256_file(cfnai_path),
                },
            ]
        ),
        "bad_dates": pd.concat(
            [sentiment_bad_dates, cfnai_bad_dates],
            ignore_index=True,
        ),
    }
    return combined, audits
