from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .atomic_write import write_text_atomic
from .path_utils import resolve_project_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def load_project_config(root: Path) -> dict[str, Any]:
    config_dir = root / "config"
    names = ("paths", "parameters", "fund_filter", "feature_directions", "schemas")
    configs = {name: load_yaml(config_dir / f"{name}.yaml") for name in names}
    paths = configs["paths"]
    resolved_paths: dict[str, Path] = {}
    path_options: dict[str, dict[str, Any]] = {}
    for key, value in paths.items():
        if key == "project_root":
            continue
        if isinstance(value, dict):
            if "path" not in value:
                raise ValueError(f"Nested path configuration {key!r} must contain 'path'.")
            resolved_paths[key] = resolve_project_path(root, value["path"])
            path_options[key] = {name: option for name, option in value.items() if name != "path"}
        else:
            resolved_paths[key] = resolve_project_path(root, value)
            path_options[key] = {}
    configs["resolved_paths"] = resolved_paths
    configs["path_options"] = path_options
    parameters = configs["parameters"]
    if parameters.get("schema_version") == "3.0.0":
        expected = {
            "label_name": "alpha",
            "label_semantic_name": "rolling_carhart_abnormal_return",
            "label_formula_version": "rolling_36m_ffc4_beta_without_intercept_subtraction",
            "legacy_project_label_name": "rret",
            "label_rename_only": True,
            "label_values_changed": False,
            "official_model_label_name": "alpha",
            "official_raw_return_field": "rret",
            "official_raw_return_alias": "official_rret_raw",
            "official_same_semantic_reference": "alpha",
            "official_same_name_collision_resolved": True,
        }
        mismatches = {
            key: {"expected": value, "actual": parameters.get(key)}
            for key, value in expected.items()
            if parameters.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Invalid schema v3 label configuration: {mismatches}")
    return configs


def config_hash(config_dir: str | Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(config_dir).glob("*.yaml")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_update_feature_directions(
    path: str | Path,
    stock_directions: dict[str, int],
    fund_directions: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    all_directions = {**stock_directions, **fund_directions}
    if len(stock_directions) != 46:
        raise ValueError(f"Expected 46 stock directions, got {len(stock_directions)}")
    if len(fund_directions) != 13:
        raise ValueError(f"Expected 13 fund directions, got {len(fund_directions)}")
    if len(all_directions) != 59 or any(value not in (-1, 1) for value in all_directions.values()):
        raise ValueError("Final directions must contain exactly 59 unique ±1 values.")
    forbidden = {"rret", "alpha", "Sentiment", "sentiment", "CFNAI", "cfnai"}
    if forbidden & set(all_directions):
        raise ValueError("Labels and macro variables cannot appear in feature directions.")
    payload = {
        "version": 1,
        "status": "final_frozen",
        "method": metadata,
        "stock_directions": stock_directions,
        "fund_directions": fund_directions,
        "directions": all_directions,
        "direction_sources": {
            **{name: "positive_stock_factor_premium" for name in stock_directions},
            **{name: "fixed_fund_rule" for name in fund_directions},
        },
    }
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    write_text_atomic(path, text)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
