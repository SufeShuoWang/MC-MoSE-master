from __future__ import annotations

import csv
import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import Any


EXECUTION_CONTEXTS = {
    "production",
    "clean_rebuild_compute",
    "clean_rebuild_compare",
}
_ACCESS_LOCK = threading.RLock()
_ACCESS_ROWS: dict[str, dict[str, Any]] = {}
_AUDIT_HOOK_INSTALLED = False
_AUDIT_REENTRY = threading.local()


def execution_context() -> str:
    value = os.getenv("KANIEL_EXECUTION_CONTEXT", "production").strip().lower()
    if value not in EXECUTION_CONTEXTS:
        raise ValueError(
            f"KANIEL_EXECUTION_CONTEXT must be one of {sorted(EXECUTION_CONTEXTS)}, got {value!r}"
        )
    return value


def clean_run_id() -> str:
    value = os.getenv("KANIEL_CLEAN_RUN_ID", "").strip()
    if execution_context() == "clean_rebuild_compute" and not value:
        raise ValueError("KANIEL_CLEAN_RUN_ID is required for clean_rebuild_compute.")
    if value and (
        Path(value).name != value
        or not value.startswith("schema_v3_clean_rebuild_")
        or any(part in value for part in ("/", "\\", ".."))
    ):
        raise ValueError(f"Unsafe KANIEL_CLEAN_RUN_ID: {value!r}")
    return value


def clean_roots(root: Path) -> dict[str, Path]:
    run_id = clean_run_id()
    return {
        "data": root / "data" / "reproducibility_check" / run_id,
        "outputs": root / "outputs" / "reproducibility_check" / run_id,
        "logs": root / "logs" / "reproducibility_check" / run_id,
    }


def clean_output_path(root: Path, formal_relative: str | Path) -> Path:
    relative = Path(formal_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Clean output path must be safe and relative: {formal_relative}")
    roots = clean_roots(root)
    if not relative.parts:
        raise ValueError("Clean output path cannot be empty.")
    head, *tail = relative.parts
    if head == "data":
        return roots["data"].joinpath(*tail)
    if head == "outputs":
        return roots["outputs"].joinpath(*tail)
    if head == "logs":
        return roots["logs"].joinpath(*tail)
    if head == "docs":
        return roots["outputs"].joinpath("docs", *tail)
    raise ValueError(f"Unsupported clean output namespace: {formal_relative}")


def run_log_root(root: Path) -> Path:
    if execution_context() == "clean_rebuild_compute":
        return clean_roots(root)["logs"]
    return root / "logs"


def _inside(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def _classify_clean_read(root: Path, path: Path) -> tuple[bool, str]:
    resolved = path.resolve()
    if not _inside(resolved, root):
        return True, "python_or_system_runtime"
    roots = clean_roots(root)
    if any(_inside(resolved, value) for value in roots.values()):
        return True, "current_clean_run"
    allowed = (
        root / "config",
        root / "src",
        root / "notebooks",
        root / "docs",
        root / "Data_processing",
    )
    if any(_inside(resolved, value) for value in allowed):
        return True, "approved_source_or_raw_input"
    forbidden = (
        root / "outputs",
        root / "data",
        root / "validation" / "kaniel_official_v2" / "outputs",
    )
    if any(_inside(resolved, value) for value in forbidden):
        return False, "old_intermediate_or_formal_result"
    return True, "repository_runtime_file"


def _access_stage() -> str:
    return os.getenv("KANIEL_NOTEBOOK_NAME", "orchestration")


def register_path_access(
    root: Path,
    path: str | Path,
    *,
    access_type: str,
    stage: str | None = None,
    enforce: bool = True,
) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    allowed, reason = (True, "production_context")
    if execution_context() == "clean_rebuild_compute" and access_type == "read":
        allowed, reason = _classify_clean_read(root, resolved)
    key = f"{resolved}|{access_type}"
    with _ACCESS_LOCK:
        _ACCESS_ROWS.setdefault(
            key,
            {
                "path": str(candidate),
                "resolved_path": str(resolved),
                "access_stage": stage or _access_stage(),
                "access_type": access_type,
                "allowed": allowed,
                "allow_reason": reason,
                "sha256": None,
            },
        )
    if enforce and not allowed:
        raise RuntimeError(
            "CLEAN_REBUILD_FORBIDDEN_READ: "
            f"{resolved} ({reason}, stage={stage or _access_stage()})"
        )
    return resolved


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flush_access_manifest(root: Path) -> Path | None:
    if execution_context() != "clean_rebuild_compute":
        return None
    manifest = clean_roots(root)["outputs"] / "reports" / "input_access_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                existing[(row["resolved_path"], row["access_type"])] = row
    with _ACCESS_LOCK:
        pending = list(_ACCESS_ROWS.values())
    for row in pending:
        key = (row["resolved_path"], row["access_type"])
        if key in existing:
            continue
        value = dict(row)
        value["allowed"] = str(bool(value["allowed"])).lower()
        value["sha256"] = _sha256(Path(value["resolved_path"])) if value["access_type"] == "read" else None
        existing[key] = {name: "" if item is None else str(item) for name, item in value.items()}
    fields = [
        "path", "resolved_path", "stage", "access_stage", "access_type",
        "allowed", "allow_reason", "sha256",
    ]
    for row in existing.values():
        row["stage"] = row.get("access_stage", "")
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(existing.values(), key=lambda row: (row["access_stage"], row["resolved_path"])))
    os.replace(temporary, manifest)
    return manifest


def install_clean_read_guard(root: Path) -> None:
    global _AUDIT_HOOK_INSTALLED
    if execution_context() != "clean_rebuild_compute" or _AUDIT_HOOK_INSTALLED:
        return

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or getattr(_AUDIT_REENTRY, "active", False):
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 else "r"
        is_read = not isinstance(mode, str) or not any(flag in mode for flag in ("w", "a", "x"))
        if not is_read:
            return
        try:
            _AUDIT_REENTRY.active = True
            candidate = Path(args[0])
            resolved = candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
            monitored = (
                root / "config",
                root / "src",
                root / "notebooks",
                root / "docs",
                root / "Data_processing",
                root / "data",
                root / "outputs",
                root / "logs" / "reproducibility_check",
                root / "validation" / "kaniel_official_v2" / "outputs",
            )
            if _inside(resolved, root) and any(
                _inside(resolved, value) for value in monitored
            ):
                register_path_access(root, resolved, access_type="read")
        finally:
            _AUDIT_REENTRY.active = False

    sys.addaudithook(audit_hook)
    _AUDIT_HOOK_INSTALLED = True


def active_release_root(root: Path) -> Path | None:
    value = os.getenv("KANIEL_RELEASE_ROOT", "").strip()
    if not value:
        return None
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"KANIEL_RELEASE_ROOT escapes PROJECT_ROOT: {candidate}")
    return candidate


def output_path(root: Path, run_mode: str, notebook_name: str, formal_relative: str) -> Path:
    formal = root / formal_relative
    if run_mode == "full":
        if execution_context() == "clean_rebuild_compute":
            return clean_output_path(root, formal_relative)
        release = active_release_root(root)
        if release is not None:
            return release / formal_relative
        return formal
    return root / "data" / "audit" / "smoke" / notebook_name / formal.name


def upstream_path(
    root: Path,
    run_mode: str,
    upstream_notebook: str,
    formal_relative: str,
) -> Path:
    if run_mode == "full":
        if execution_context() == "clean_rebuild_compute":
            staged = clean_output_path(root, formal_relative)
            register_path_access(
                root,
                staged,
                access_type="read",
                stage=upstream_notebook,
            )
            if not staged.exists():
                raise FileNotFoundError(
                    f"Clean upstream output is missing; formal fallback is forbidden: {staged}"
                )
            return staged
        release = active_release_root(root)
        if release is not None:
            staged = release / formal_relative
            if staged.exists():
                return staged
        return root / formal_relative
    smoke = root / "data" / "audit" / "smoke" / upstream_notebook / Path(formal_relative).name
    return smoke if smoke.exists() else root / formal_relative
