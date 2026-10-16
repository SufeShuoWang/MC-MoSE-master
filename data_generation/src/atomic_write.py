from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    return Path(name)


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    target = Path(path)
    temporary = _temporary_sibling(target)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_json_atomic(path: str | Path, value: Any) -> Path:
    return write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )


def write_csv_atomic(frame: Any, path: str | Path) -> Path:
    target = Path(path)
    temporary = _temporary_sibling(target)
    try:
        if hasattr(frame, "write_csv"):
            frame.write_csv(temporary)
        elif hasattr(frame, "to_csv"):
            frame.to_csv(temporary, index=False)
        else:
            raise TypeError("Expected a Polars or pandas DataFrame.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_parquet_atomic(frame: Any, path: str | Path) -> Path:
    target = Path(path)
    temporary = _temporary_sibling(target)
    try:
        if hasattr(frame, "write_parquet"):
            frame.write_parquet(temporary)
        elif hasattr(frame, "to_parquet"):
            frame.to_parquet(temporary, index=False)
        else:
            raise TypeError("Expected a Polars or pandas DataFrame.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_npz_atomic(path: str | Path, **arrays: Any) -> Path:
    """Atomically write a compressed NumPy archive without changing its suffix."""
    import numpy as np

    target = Path(path)
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def sink_parquet_atomic(lazy_frame: Any, path: str | Path, **sink_options: Any) -> Path:
    """Stream a Polars LazyFrame to a sibling file and atomically publish it."""
    target = Path(path)
    temporary = _temporary_sibling(target)
    try:
        options = {"compression": "zstd", "maintain_order": True}
        options.update(sink_options)
        lazy_frame.sink_parquet(temporary, **options)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def duckdb_copy_parquet_atomic(
    connection: Any,
    query: str,
    path: str | Path,
    *,
    compression: str = "zstd",
) -> Path:
    """Run an out-of-core DuckDB COPY into a sibling Parquet and publish atomically."""
    target = Path(path)
    temporary = _temporary_sibling(target)
    escaped = str(temporary).replace("'", "''")
    try:
        connection.execute(
            f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION {compression.upper()})"
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def duckdb_copy_csv_atomic(connection: Any, query: str, path: str | Path) -> Path:
    target = Path(path)
    temporary = _temporary_sibling(target)
    escaped = str(temporary).replace("'", "''")
    try:
        connection.execute(
            f"COPY ({query}) TO '{escaped}' (FORMAT CSV, HEADER TRUE)"
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
