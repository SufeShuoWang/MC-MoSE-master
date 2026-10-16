from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compatible_layout_adapter import LayoutAdapterError, convert_layout


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "npz"
    / "kaniel_1980_2024_model_ready_3.2.npz"
)
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "Data_processing"
    / "official_kaniel_v2"
    / "Data"
    / "alpha_imputed_chara_sentiment.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "npz"
    / "kaniel_1980_2024_model_ready_3.2_compatible_layout.npz"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "npz" / "dataset_validation_report.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repackage the existing Kaniel extension as a Kaniel-compatible "
            "extended layout without recalculating features."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing compatible-layout output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = convert_layout(
            args.input,
            args.reference,
            args.output,
            args.report,
            overwrite=args.overwrite,
            expected_dimensions=(540, 6207, 59),
        )
    except LayoutAdapterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Validation report: {args.report}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
