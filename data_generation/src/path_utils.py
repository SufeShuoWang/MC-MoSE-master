from __future__ import annotations

from pathlib import Path
from typing import Iterable


ROOT_MARKERS = (
    "config/paths.yaml",
    "notebooks",
    "src",
)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root without creating or accepting a nested project root."""
    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in ROOT_MARKERS):
            if candidate.name == "Kaniel_Rebuild":
                raise RuntimeError("Nested Kaniel_Rebuild project roots are forbidden.")
            return candidate
    raise FileNotFoundError(
        f"Could not locate PROJECT_ROOT from {current}; required markers: {ROOT_MARKERS}"
    )


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def ensure_project_directories(root: Path, relative_paths: Iterable[str]) -> list[Path]:
    created: list[Path] = []
    for relative in relative_paths:
        target = (root / relative).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Directory escapes PROJECT_ROOT: {target}")
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return created
