from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes project root: {resolved}")
    return resolved


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_staged_release(
    root: Path,
    release_root: Path,
    archive_root: Path,
    relative_paths: Iterable[str],
    *,
    metadata_relative_path: str,
    manifest_relative_path: str,
    schema_version: str = "2.0.0",
    release_name: str | None = None,
    lock_name: str = ".release_publish.lock",
) -> dict[str, object]:
    root = root.resolve()
    release_root = _inside(root, release_root)
    archive_root = _inside(root, archive_root)
    paths = list(dict.fromkeys(relative_paths))
    if metadata_relative_path not in paths:
        raise ValueError("metadata_relative_path must be part of relative_paths.")
    ordered = [value for value in paths if value != metadata_relative_path] + [
        metadata_relative_path
    ]
    for relative in ordered:
        staged = _inside(root, release_root / relative)
        if not staged.is_file():
            raise FileNotFoundError(f"Missing staged release file: {staged}")

    lock_path = _inside(root, root / "outputs" / lock_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    published: list[str] = []
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        for relative in ordered:
            staged = release_root / relative
            target = _inside(root, root / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".publish.tmp",
                dir=target.parent,
            )
            os.close(handle)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(staged, temporary)
                if file_sha256(temporary) != file_sha256(staged):
                    raise IOError(f"Staged copy hash mismatch for {relative}")
                os.replace(temporary, target)
                published.append(relative)
            finally:
                temporary.unlink(missing_ok=True)

        file_rows = []
        for relative in ordered:
            target = root / relative
            file_rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            )
        manifest = {
            "schema_version": schema_version,
            "release_name": release_name,
            "published_utc": datetime.now(timezone.utc).isoformat(),
            "release_root": str(release_root.relative_to(root)),
            "archive_root": str(archive_root.relative_to(root)),
            "published_files": file_rows,
            "metadata_published_last": True,
        }
        manifest_path = _inside(root, root / manifest_relative_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
        return manifest
    except Exception:
        for relative in reversed(published):
            target = _inside(root, root / relative)
            archived = _inside(root, archive_root / relative)
            if archived.is_file():
                handle, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.",
                    suffix=".rollback.tmp",
                    dir=target.parent,
                )
                os.close(handle)
                temporary = Path(temporary_name)
                try:
                    shutil.copy2(archived, temporary)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)
