from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .atomic_write import write_json_atomic
from .config_utils import config_hash
from .runtime_paths import (
    execution_context,
    flush_access_manifest,
    register_path_access,
    run_log_root,
)


def _file_metadata(root: Path, values: list[str]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    raw_root = root / "Data_processing" / "manual_processing"
    for value in values:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raw_candidate = raw_root / Path(value).name
            candidate = raw_candidate if raw_candidate.exists() else candidate
        if candidate.exists():
            register_path_access(
                root,
                candidate,
                access_type="read",
                stage=os.getenv("KANIEL_NOTEBOOK_NAME", notebook_stage_from_value(value)),
            )
        item: dict[str, Any] = {"declared_path": value, "exists": candidate.exists()}
        if candidate.exists() and candidate.is_file():
            stat = candidate.stat()
            item.update(
                {
                    "resolved_path": str(candidate.resolve()),
                    "size_bytes": stat.st_size,
                    "modified_time_ns": stat.st_mtime_ns,
                }
            )
        metadata.append(item)
    return metadata


def notebook_stage_from_value(value: str) -> str:
    return os.getenv("KANIEL_NOTEBOOK_NAME", "run_summary")


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for name in (
        "polars", "pandas", "numpy", "pyarrow", "scipy", "sklearn",
        "statsmodels", "duckdb", "openpyxl",
    ):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            versions[name] = f"unavailable: {error}"
    return versions


def write_run_summary(
    root: Path,
    notebook_name: str,
    *,
    stage: str,
    started_at: float,
    input_files: list[str],
    output_files: list[str],
    input_row_counts: dict[str, int | None],
    output_row_counts: dict[str, int | None],
    primary_key: list[str],
    duplicate_key_count: int,
    excluded_row_count: int,
    exception_file: str | None,
    date_min: str | None = None,
    date_max: str | None = None,
    missing_rate_by_key_field: dict[str, float | None] | None = None,
    status: str = "passed",
    notes: list[str] | None = None,
) -> Path:
    summary = {
        "stage": stage,
        "notebook": notebook_name,
        "status": status,
        "run_mode": os.getenv("KANIEL_RUN_MODE", "unknown"),
        "execution_context": execution_context(),
        "run_time_seconds": round(time.time() - started_at, 6),
        "input_files": input_files,
        "input_file_metadata": _file_metadata(root, input_files),
        "input_row_counts": input_row_counts,
        "output_files": output_files,
        "output_file_metadata": _file_metadata(root, output_files),
        "output_row_counts": output_row_counts,
        "date_min": date_min,
        "date_max": date_max,
        "primary_key": primary_key,
        "duplicate_key_count": duplicate_key_count,
        "missing_rate_by_key_field": missing_rate_by_key_field or {},
        "excluded_row_count": excluded_row_count,
        "exception_file": exception_file,
        "config_hash": config_hash(root / "config"),
        "notes": notes or [],
    }
    summary_path = write_json_atomic(
        run_log_root(root) / f"{notebook_name}_run_summary.json",
        summary,
    )
    flush_access_manifest(root)
    return summary_path


def load_run_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
