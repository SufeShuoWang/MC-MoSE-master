from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compatible_layout_no_cf import (
    NoCFPostprocessError,
    remove_cf_from_compatible_layout,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "npz"
    / "kaniel_1980_2024_model_ready_3.2_compatible_layout.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "npz"
    / "kaniel_1980_2024_model_ready_3.2_no_CF_compatible_layout.npz"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "npz"
    / "dataset_validation_report_no_CF.md"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove only the exact core CF feature from an already validated "
            "Kaniel-compatible extended-layout NPZ. No feature is recalculated."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing no-CF output and report.",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=8,
        help="Number of months per out-of-core copy/validation chunk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = remove_cf_from_compatible_layout(
            args.input,
            args.output,
            args.report,
            overwrite=args.overwrite,
            expected_input_shape=(540, 6207, 62),
            expected_date_range=(198001, 202412),
            chunk_months=args.chunk_months,
        )
    except NoCFPostprocessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Validation report: {args.report}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
