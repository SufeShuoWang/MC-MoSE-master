from __future__ import annotations

import hashlib
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SENTINEL = -99.99


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def compare_parquet_rename_only(
    old_path: str | Path,
    new_path: str | Path,
    *,
    old_column: str,
    new_column: str,
) -> dict[str, Any]:
    """Compare two Parquet files exactly after one declared column rename."""
    old_path = Path(old_path)
    new_path = Path(new_path)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=true")
    try:
        old_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(old_path)]
        ).fetch_df()
        new_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(new_path)]
        ).fetch_df()
        old_names = old_schema["column_name"].astype(str).tolist()
        new_names = new_schema["column_name"].astype(str).tolist()
        expected_names = [new_column if name == old_column else name for name in old_names]
        if expected_names != new_names:
            return {
                "passed": False,
                "row_count_agreement": False,
                "schema_name_agreement": False,
                "dtype_agreement": False,
                "ordered_value_agreement": False,
                "old_columns": old_names,
                "new_columns": new_names,
            }
        old_types = old_schema["column_type"].astype(str).tolist()
        new_types = new_schema["column_type"].astype(str).tolist()
        dtype_agreement = old_types == new_types
        old_select = ", ".join(
            f"{_quoted(name)} AS {_quoted(new_column if name == old_column else name)}"
            for name in old_names
        )
        new_select = ", ".join(_quoted(name) for name in new_names)
        query = f"""
        WITH old_rows AS (
          SELECT row_number() OVER () AS _position, {old_select}
          FROM read_parquet(?)
        ),
        new_rows AS (
          SELECT row_number() OVER () AS _position, {new_select}
          FROM read_parquet(?)
        ),
        forward_diff AS (
          SELECT * FROM old_rows EXCEPT ALL SELECT * FROM new_rows
        ),
        reverse_diff AS (
          SELECT * FROM new_rows EXCEPT ALL SELECT * FROM old_rows
        )
        SELECT
          (SELECT count(*) FROM old_rows) AS old_rows,
          (SELECT count(*) FROM new_rows) AS new_rows,
          (SELECT count(*) FROM forward_diff) AS forward_diff_rows,
          (SELECT count(*) FROM reverse_diff) AS reverse_diff_rows
        """
        counts = connection.execute(query, [str(old_path), str(new_path)]).fetchone()
        old_rows, new_rows, forward_diff, reverse_diff = map(int, counts)
        ordered_value_agreement = forward_diff == 0 and reverse_diff == 0
        passed = (
            old_rows == new_rows
            and dtype_agreement
            and ordered_value_agreement
        )
        return {
            "passed": passed,
            "old_rows": old_rows,
            "new_rows": new_rows,
            "row_count_agreement": old_rows == new_rows,
            "schema_name_agreement": True,
            "dtype_agreement": dtype_agreement,
            "ordered_value_agreement": ordered_value_agreement,
            "forward_diff_rows": forward_diff,
            "reverse_diff_rows": reverse_diff,
            "renamed_column": {"old": old_column, "new": new_column},
        }
    finally:
        connection.close()


def compare_label_columns(
    old_path: str | Path,
    new_path: str | Path,
    *,
    keys: Iterable[str],
    old_column: str,
    new_column: str,
) -> dict[str, Any]:
    key_columns = list(keys)
    old = pd.read_parquet(old_path, columns=[*key_columns, old_column]).rename(
        columns={old_column: "legacy_project_rret"}
    )
    new = pd.read_parquet(new_path, columns=[*key_columns, new_column]).rename(
        columns={new_column: "project_alpha"}
    )
    merged = old.merge(new, on=key_columns, how="outer", validate="one_to_one", indicator=True)
    left = pd.to_numeric(merged["legacy_project_rret"], errors="coerce").to_numpy()
    right = pd.to_numeric(merged["project_alpha"], errors="coerce").to_numpy()
    left_valid = np.isfinite(left)
    right_valid = np.isfinite(right)
    common = left_valid & right_valid
    exact = left[common] == right[common]
    difference = np.abs(left[common] - right[common])
    return {
        "n_compared": int(len(merged)),
        "key_match_rate": float((merged["_merge"] == "both").mean()),
        "exact_match_rate": float(exact.mean()) if len(exact) else math.nan,
        "missing_mask_agreement": float((left_valid == right_valid).mean()),
        "dtype_agreement": str(old["legacy_project_rret"].dtype)
        == str(new["project_alpha"].dtype),
        "match_rate_1e_12": float((difference <= 1e-12).mean()) if len(difference) else math.nan,
        "match_rate_1e_10": float((difference <= 1e-10).mean()) if len(difference) else math.nan,
        "max_absolute_difference": float(difference.max()) if len(difference) else math.nan,
        "mean_absolute_difference": float(difference.mean()) if len(difference) else math.nan,
    }


def _zip_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_npz_members(old_path: str | Path, new_path: str | Path) -> dict[str, Any]:
    expected = [
        "X",
        "X_macro",
        "y",
        "dates",
        "feature_months",
        "wficn",
        "feature_names",
        "macro_feature_names",
    ]
    with zipfile.ZipFile(old_path) as old_zip, zipfile.ZipFile(new_path) as new_zip:
        old_members = [Path(name).stem for name in old_zip.namelist() if name.endswith(".npy")]
        new_members = [Path(name).stem for name in new_zip.namelist() if name.endswith(".npy")]
        rows = []
        for key in expected:
            member = f"{key}.npy"
            old_hash = _zip_member_sha256(old_zip, member)
            new_hash = _zip_member_sha256(new_zip, member)
            rows.append(
                {
                    "key": key,
                    "old_npy_sha256": old_hash,
                    "new_npy_sha256": new_hash,
                    "exact": old_hash == new_hash,
                }
            )
    return {
        "expected_keys": expected,
        "old_keys": old_members,
        "new_keys": new_members,
        "key_order_exact": old_members == expected and new_members == expected,
        "arrays": rows,
        "all_arrays_exact": all(row["exact"] for row in rows),
        "X_exact_match_rate": 1.0 if next(row for row in rows if row["key"] == "X")["exact"] else 0.0,
        "X_macro_exact_match_rate": 1.0
        if next(row for row in rows if row["key"] == "X_macro")["exact"]
        else 0.0,
        "y_exact_match_rate": 1.0 if next(row for row in rows if row["key"] == "y")["exact"] else 0.0,
        "axes_exact": all(
            next(row for row in rows if row["key"] == key)["exact"]
            for key in (
                "dates",
                "feature_months",
                "wficn",
                "feature_names",
                "macro_feature_names",
            )
        ),
    }


def _coerce_wficn(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round().astype("Int64")


def _month_from_yyyymm(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(values, errors="coerce").round().astype("Int64").astype("string"),
        format="%Y%m",
        errors="coerce",
    )


def _metrics(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
    a = np.where(np.isclose(a, SENTINEL, rtol=0, atol=1e-4), np.nan, a)
    b = np.where(np.isclose(b, SENTINEL, rtol=0, atol=1e-4), np.nan, b)
    av, bv = np.isfinite(a), np.isfinite(b)
    valid = av & bv
    x, y = a[valid], b[valid]
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(spearmanr(x, y).statistic)
    else:
        pearson = math.nan
        spearman = math.nan
    return {
        "n_positions": int(len(a)),
        "n_common": int(valid.sum()),
        "missing_agreement": float(np.mean(av == bv)) if len(a) else math.nan,
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(np.mean(np.abs(x - y))) if len(x) else math.nan,
        "rmse": float(np.sqrt(np.mean(np.square(x - y)))) if len(x) else math.nan,
    }


def compare_project_alpha_to_official(
    project_panel_path: str | Path,
    official_csv_path: str | Path,
) -> dict[str, Any]:
    project = pd.read_parquet(
        project_panel_path, columns=["wficn", "target_month", "alpha"]
    ).rename(columns={"alpha": "project_alpha", "target_month": "month"})
    project["month"] = pd.to_datetime(project["month"]).dt.to_period("M").dt.to_timestamp()
    project["wficn"] = _coerce_wficn(project["wficn"])
    official = pd.read_csv(
        official_csv_path,
        usecols=["Date", "wficn", "rret", "alpha"],
        low_memory=False,
    ).rename(columns={"rret": "official_rret_raw", "alpha": "official_alpha"})
    official["month"] = _month_from_yyyymm(official["Date"])
    official["wficn"] = _coerce_wficn(official["wficn"])
    if official.duplicated(["wficn", "month"]).any():
        raise AssertionError("Official abnormal_return.csv has duplicate WFICN-month keys.")
    aligned = project.merge(
        official[["wficn", "month", "official_rret_raw", "official_alpha"]],
        on=["wficn", "month"],
        how="inner",
        validate="one_to_one",
    )
    return {
        "project_label_name": "alpha",
        "project_label_semantic": "rolling_carhart_abnormal_return",
        "official_alpha_relevance": "primary_semantic_reference",
        "official_raw_rret_relevance": "raw_return_reference_only",
        "label_name_collision_status": "resolved",
        "project_alpha_vs_official_alpha": _metrics(
            aligned["project_alpha"], aligned["official_alpha"]
        ),
        "project_alpha_vs_official_raw_rret": _metrics(
            aligned["project_alpha"], aligned["official_rret_raw"]
        ),
        "official_raw_rret_preserved": True,
        "official_columns_read": ["Date", "wficn", "rret", "alpha"],
    }
