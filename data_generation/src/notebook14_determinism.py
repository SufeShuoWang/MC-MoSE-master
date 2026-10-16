from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic_write import duckdb_copy_csv_atomic, duckdb_copy_parquet_atomic


MONTHLY_RAW_COLUMNS = (
    "PERMNO",
    "CUSIP",
    "MthCalDt",
    "MthRet",
    "MthRetFlg",
    "MthPrc",
    "MthCap",
    "MthVol",
    "ShrOut",
    "SecInfoStartDt",
    "SecInfoEndDt",
    "ShareType",
    "SecurityType",
    "SecuritySubType",
    "USIncFlg",
    "IssuerType",
    "PrimaryExch",
    "ConditionalType",
    "TradingStatusFlg",
)

MONTHLY_CONTENT_COLUMNS = (
    "cusip",
    "mthcaldt",
    "mthret",
    "mthretflg",
    "mthprc",
    "mthcap",
    "mthvol",
    "shrout",
    "secinfostartdt",
    "secinfoenddt",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "common_stock",
    "market_equity_thousand",
)


def _escape(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _quoted(columns: tuple[str, ...]) -> str:
    return ",".join(f'"{column}"' for column in columns)


def _distinct_value_count(column: str) -> str:
    quoted = f'"{column}"'
    return (
        f"(count(DISTINCT {quoted})"
        f"+CASE WHEN count(*) FILTER (WHERE {quoted} IS NULL)>0 THEN 1 ELSE 0 END)"
    )


def _conflict_fields_expression() -> str:
    values = ",".join(
        (
            f"CASE WHEN {_distinct_value_count(column)}>1 "
            f"THEN '{column}' ELSE NULL END"
        )
        for column in MONTHLY_CONTENT_COLUMNS
    )
    return f"to_json(list_filter([{values}], value -> value IS NOT NULL))"


def _raw_record_struct() -> str:
    fields = ",".join(
        f"{column} := raw_{column.lower()}" for column in MONTHLY_RAW_COLUMNS
    )
    return f"struct_pack({fields})"


def _raw_record_order() -> str:
    return ",".join(
        f"raw_{column.lower()} ASC NULLS FIRST" for column in MONTHLY_RAW_COLUMNS
    )


def _canonical_record_struct() -> str:
    fields = ",".join(
        [
            "permno := permno",
            "month := month_value",
            *[
                f"{column} := \"{column}\""
                for column in MONTHLY_CONTENT_COLUMNS
            ],
        ]
    )
    return f"struct_pack({fields})"


def build_monthly_crsp_deterministic(
    connection: Any,
    *,
    source_path: str | Path,
    normalized_stage_path: str | Path,
    clean_stage_path: str | Path,
    conflict_audit_path: str | Path,
    exact_duplicate_audit_path: str | Path,
    limit_rows: int | None = None,
) -> dict[str, int]:
    """Build the deterministic Notebook-14 monthly CRSP output and audits.

    Exact duplicate canonical rows are collapsed. If one PERMNO-month key has
    more than one distinct canonical content row, every raw occurrence for
    that key is excluded and retained in the conflict audit.
    """

    source = _escape(source_path)
    normalized_stage = Path(normalized_stage_path)
    clean_stage = Path(clean_stage_path)
    normalized_stage.unlink(missing_ok=True)
    clean_stage.unlink(missing_ok=True)
    limit_clause = f"LIMIT {int(limit_rows)}" if limit_rows is not None else ""
    projection = ",".join(MONTHLY_RAW_COLUMNS)
    raw_projection = ",".join(
        f'"{column}" AS raw_{column.lower()}' for column in MONTHLY_RAW_COLUMNS
    )
    normalized_query = f"""
    WITH raw AS (
      SELECT {projection}
      FROM read_csv_auto('{source}', header=true, all_varchar=true)
      {limit_clause}
    )
    SELECT
      {raw_projection},
      try_cast(PERMNO AS BIGINT) AS permno,
      CUSIP AS cusip,
      MthCalDt AS mthcaldt,
      date_trunc('month', try_cast(MthCalDt AS DATE))::DATE AS month_value,
      CASE WHEN try_cast(MthRet AS DOUBLE) IN (-66,-77,-88,-99) THEN NULL
           ELSE try_cast(MthRet AS DOUBLE) END AS mthret,
      MthRetFlg AS mthretflg,
      CASE WHEN try_cast(MthPrc AS DOUBLE) IN (-66,-77,-88,-99) THEN NULL
           ELSE try_cast(MthPrc AS DOUBLE) END AS mthprc,
      try_cast(MthCap AS DOUBLE) AS mthcap,
      try_cast(MthVol AS DOUBLE) AS mthvol,
      try_cast(ShrOut AS DOUBLE) AS shrout,
      SecInfoStartDt AS secinfostartdt,
      SecInfoEndDt AS secinfoenddt,
      ShareType AS sharetype,
      SecurityType AS securitytype,
      SecuritySubType AS securitysubtype,
      USIncFlg AS usincflg,
      IssuerType AS issuertype,
      PrimaryExch AS primaryexch,
      ConditionalType AS conditionaltype,
      TradingStatusFlg AS tradingstatusflg,
      (
        ShareType='NS' AND SecurityType='EQTY' AND SecuritySubType='COM'
        AND USIncFlg='Y' AND IssuerType IN ('ACOR','CORP')
        AND PrimaryExch IN ('N','A','Q') AND ConditionalType='RW'
        AND TradingStatusFlg='A'
      ) AS common_stock,
      abs(
        CASE WHEN try_cast(MthPrc AS DOUBLE) IN (-66,-77,-88,-99) THEN NULL
             ELSE try_cast(MthPrc AS DOUBLE) END
      ) * try_cast(ShrOut AS DOUBLE) AS market_equity_thousand
    FROM raw
    """
    duckdb_copy_parquet_atomic(connection, normalized_query, normalized_stage)
    normalized = _escape(normalized_stage)
    content = _quoted(MONTHLY_CONTENT_COLUMNS)
    group_columns = f"permno,month_value,{content}"
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW notebook14_valid_monthly AS
        SELECT * FROM read_parquet('{normalized}')
        WHERE permno IS NOT NULL AND month_value IS NOT NULL
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW notebook14_canonical_variants AS
        SELECT {group_columns}, count(*)::BIGINT AS raw_record_count
        FROM notebook14_valid_monthly
        GROUP BY {group_columns}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW notebook14_conflict_keys AS
        SELECT permno, month_value, count(*)::BIGINT AS canonical_variant_count
        FROM notebook14_canonical_variants
        GROUP BY permno, month_value
        HAVING count(*) > 1
        """
    )

    clean_query = f"""
    SELECT
      permno,cusip,mthcaldt,mthret,mthretflg,mthprc,mthcap,mthvol,shrout,
      secinfostartdt,secinfoenddt,sharetype,securitytype,securitysubtype,
      usincflg,issuertype,primaryexch,conditionaltype,tradingstatusflg,
      month_value AS "month",common_stock,market_equity_thousand
    FROM notebook14_canonical_variants variants
    WHERE NOT EXISTS (
      SELECT 1 FROM notebook14_conflict_keys conflicts
      WHERE conflicts.permno=variants.permno
        AND conflicts.month_value=variants.month_value
    )
    ORDER BY permno,month_value
    """
    duckdb_copy_parquet_atomic(connection, clean_query, clean_stage)

    raw_struct = _raw_record_struct()
    raw_order = _raw_record_order()
    conflict_fields = _conflict_fields_expression()
    conflict_query = f"""
    SELECT
      valid.permno,
      valid.month_value AS "month",
      concat(
        cast(valid.permno AS VARCHAR),'|',
        strftime(valid.month_value,'%Y-%m')
      ) AS conflict_key,
      to_json(list({raw_struct} ORDER BY {raw_order})) AS raw_records,
      {conflict_fields} AS conflict_fields,
      (
        {_distinct_value_count("common_stock")} > 1
      ) AS affected_common_stock_flag,
      count(*)::BIGINT AS excluded_record_count
    FROM notebook14_valid_monthly valid
    INNER JOIN notebook14_conflict_keys conflicts
      USING (permno,month_value)
    GROUP BY valid.permno,valid.month_value
    ORDER BY valid.permno,valid.month_value
    """
    duckdb_copy_csv_atomic(connection, conflict_query, conflict_audit_path)

    canonical_struct = _canonical_record_struct()
    exact_duplicate_query = f"""
    SELECT
      permno,
      month_value AS "month",
      concat(cast(permno AS VARCHAR),'|',strftime(month_value,'%Y-%m'))
        AS duplicate_key,
      to_json({canonical_struct}) AS canonical_record,
      raw_record_count,
      raw_record_count-1 AS removed_record_count
    FROM notebook14_canonical_variants
    WHERE raw_record_count>1
    ORDER BY permno,month_value,canonical_record
    """
    duckdb_copy_csv_atomic(
        connection,
        exact_duplicate_query,
        exact_duplicate_audit_path,
    )

    raw_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{normalized}')"
        ).fetchone()[0]
    )
    valid_key_count = int(
        connection.execute(
            "SELECT count(*) FROM notebook14_valid_monthly"
        ).fetchone()[0]
    )
    exact_duplicate_variant_count, exact_duplicate_record_count = (
        connection.execute(
            """
            SELECT
              count(*) FILTER (WHERE raw_record_count>1),
              coalesce(sum(raw_record_count-1)
                FILTER (WHERE raw_record_count>1),0)
            FROM notebook14_canonical_variants
            """
        ).fetchone()
    )
    conflict_key_count, conflict_record_count = connection.execute(
        """
        SELECT count(DISTINCT (valid.permno,valid.month_value)), count(*)
        FROM notebook14_valid_monthly valid
        INNER JOIN notebook14_conflict_keys conflicts
          USING (permno,month_value)
        """
    ).fetchone()
    affected_common_stock_key_count = int(
        connection.execute(
            f"""
            SELECT count(*) FROM (
              SELECT permno,month_value
              FROM notebook14_valid_monthly valid
              INNER JOIN notebook14_conflict_keys conflicts
                USING (permno,month_value)
              GROUP BY permno,month_value
              HAVING {_distinct_value_count("common_stock")}>1
            )
            """
        ).fetchone()[0]
    )
    clean = _escape(clean_stage)
    clean_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{clean}')"
        ).fetchone()[0]
    )
    duplicate_key_count = int(
        connection.execute(
            f"""
            SELECT coalesce(sum(n-1),0) FROM (
              SELECT count(*) AS n
              FROM read_parquet('{clean}')
              GROUP BY permno,"month"
              HAVING count(*)>1
            )
            """
        ).fetchone()[0]
    )
    if duplicate_key_count:
        raise AssertionError(
            f"Notebook 14 retained {duplicate_key_count} duplicate PERMNO-month rows"
        )
    return {
        "raw_record_count": raw_count,
        "invalid_key_record_count": raw_count - valid_key_count,
        "exact_duplicate_variant_count": int(exact_duplicate_variant_count),
        "exact_duplicate_record_count": int(exact_duplicate_record_count),
        "conflict_key_count": int(conflict_key_count),
        "conflict_record_count": int(conflict_record_count),
        "affected_common_stock_key_count": affected_common_stock_key_count,
        "clean_record_count": clean_count,
        "duplicate_key_count": duplicate_key_count,
    }
