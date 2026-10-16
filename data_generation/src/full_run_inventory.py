from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .macro_data import read_tabular_sample


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _source(path: Path) -> str:
    escaped = str(path).replace("'", "''")
    return (
        f"read_csv_auto('{escaped}', header=true, all_varchar=true, "
        "sample_size=200000, ignore_errors=false)"
    )


def run_full_inventory_audit(
    schemas: dict[str, Any],
    paths: dict[str, Path],
    *,
    temp_directory: Path,
    memory_limit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Out-of-core full-file counts, date coverage, key duplicates and mapping risks."""
    temp_directory.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{str(temp_directory).replace(chr(39), chr(39) * 2)}'")
    con.execute("SET preserve_insertion_order=false")
    dates: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {}

    try:
        for table_name, schema in schemas.items():
            if table_name in {"version", "note", "processed_outputs"} or not isinstance(schema, dict):
                continue
            path = paths[schema["path_key"]]
            file_format = str(schema.get("file_format", path.suffix.lstrip("."))).lower()
            if file_format in {"xlsx", "xls"}:
                frame = read_tabular_sample(path, schema)
                row_count = int(len(frame))
                row_counts[path.name] = row_count
                for raw_date in schema.get("date_columns", []):
                    values = frame[raw_date].astype("string").str.strip()
                    if schema.get("date_format") == "%Y%m":
                        values = values.str.replace(r"\.0$", "", regex=True).str.zfill(6)
                    parsed = pd.to_datetime(
                        values,
                        format=schema.get("date_format"),
                        errors="coerce",
                    )
                    nonmissing = int(values.notna().sum())
                    parsed_count = int(parsed.notna().sum())
                    dates.append(
                        {
                            "table": table_name,
                            "column": raw_date,
                            "full_nonmissing": nonmissing,
                            "full_parsed": parsed_count,
                            "parse_rate": parsed_count / nonmissing if nonmissing else None,
                            "full_min": parsed.min(),
                            "full_max": parsed.max(),
                        }
                    )
                key_columns = list(schema.get("primary_key", []))
                if key_columns:
                    duplicate = int(frame.duplicated(key_columns, keep=False).sum())
                    keys.append(
                        {
                            "table": table_name,
                            "keys": "|".join(key_columns),
                            "full_rows": row_count,
                            "full_duplicate_excess_rows": duplicate,
                            "duplicate_risk": duplicate > 0,
                        }
                    )
                continue
            source = _source(path)
            row_count = int(con.execute(f"SELECT count(*) FROM {source}").fetchone()[0])
            row_counts[path.name] = row_count

            for raw_date in schema.get("date_columns", []):
                disk_name = raw_date
                if raw_date in {"yyyymm", "yyyymmdd"}:
                    disk_name = "column0"
                fmt = "%Y%m" if raw_date == "yyyymm" else ("%Y%m%d" if raw_date == "yyyymmdd" else None)
                value = f"trim({_q(disk_name)})"
                parsed = (
                    f"try_strptime({value}, '{fmt}')"
                    if fmt
                    else f"coalesce(try_cast({value} AS DATE), try_strptime({value}, '%Y%m%d'))"
                )
                result = con.execute(
                    f"""SELECT count(*) FILTER (WHERE nullif({value}, '') IS NOT NULL),
                               count({parsed}), min({parsed}), max({parsed})
                        FROM {source}"""
                ).fetchone()
                nonmissing, parsed_count, minimum, maximum = result
                dates.append(
                    {
                        "table": table_name,
                        "column": raw_date,
                        "full_nonmissing": int(nonmissing),
                        "full_parsed": int(parsed_count),
                        "parse_rate": parsed_count / nonmissing if nonmissing else None,
                        "full_min": minimum,
                        "full_max": maximum,
                    }
                )

            key_columns = [
                "column0" if key in {"yyyymm", "yyyymmdd"} else key
                for key in schema.get("primary_key", [])
            ]
            if key_columns:
                group = ", ".join(_q(key) for key in key_columns)
                duplicate = con.execute(
                    f"""SELECT coalesce(sum(n - 1), 0)
                        FROM (SELECT count(*) n FROM {source} GROUP BY {group} HAVING count(*) > 1)"""
                ).fetchone()[0]
                keys.append(
                    {
                        "table": table_name,
                        "keys": "|".join(schema.get("primary_key", [])),
                        "full_rows": row_count,
                        "full_duplicate_excess_rows": int(duplicate),
                        "duplicate_risk": int(duplicate) > 0,
                    }
                )

        risks: list[dict[str, Any]] = []
        relationships = [
            (
                "crsp_fundno->wficn",
                paths["mflink_crsp"],
                ["crsp_fundno"],
                "wficn",
            ),
            (
                "fundno+fdate+rdate->wficn",
                paths["mflink_s12"],
                ["fundno", "fdate", "rdate"],
                "wficn",
            ),
        ]
        for label, path, left_keys, right_key in relationships:
            source = _source(path)
            group = ", ".join(_q(key) for key in left_keys)
            count = con.execute(
                f"""SELECT count(*) FROM (
                    SELECT {group} FROM {source}
                    GROUP BY {group} HAVING count(DISTINCT {_q(right_key)}) > 1)"""
            ).fetchone()[0]
            risks.append({"relationship": label, "full_conflict_keys": int(count)})

        stock_source = _source(paths["crsp_monthly"])
        count = con.execute(
            f"""SELECT count(*) FROM (
                SELECT lpad(substr(trim("CUSIP"), 1, 8), 8, '0') cusip8,
                       date_trunc('month', try_cast("MthCalDt" AS DATE)) AS month_value
                FROM {stock_source}
                GROUP BY 1, 2 HAVING count(DISTINCT "PERMNO") > 1)"""
        ).fetchone()[0]
        risks.append({"relationship": "cusip8+month->PERMNO", "full_conflict_keys": int(count)})
    finally:
        con.close()

    return pd.DataFrame(dates), pd.DataFrame(keys), pd.DataFrame(risks), row_counts
