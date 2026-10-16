"""Train independent MC-MoSE seeds using the existing training implementation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678,
                 123456789, 1234567890]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, default=ROOT / "data/fund_2023_2024.npz")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--patch_scales", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    for name, default in {
        "lookback": 10, "hidden_dim": 128, "market_embed_dim": 16,
        "mlp_hidden_dim": 64, "num_heads": 4, "num_shared_experts": 12,
        "num_routed_experts": 48, "topk_routed": 12, "balance_threshold": 2,
        "epochs": 5, "patience": 3, "train_start": 202310, "train_end": 202406,
        "val_start": 202407, "val_end": 202409, "test_start": 202410, "test_end": 202412,
    }.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in {
        "dropout": 0.1, "rank_loss_alpha": 0.1, "lambda_sparse": 1e-4,
        "lambda_diversity": 1e-4, "lambda_lb": 1e-4, "lr": 1e-3,
        "weight_decay": 1e-4, "min_delta": 1e-6,
    }.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.set_defaults(
        version="mc_mose", target_name="alpha", num_workers=0,
        train_valid_only=True, cache_samples=True, pin_memory=True,
        persistent_workers=False, prefetch_factor=2,
        collect_test_predictions=True, test_every_epoch=False,
    )
    args = parser.parse_args(argv)
    args.data_path = _resolve_path(args.data_path)
    args.output_dir = _resolve_path(args.output_dir)
    validate_args(args)
    return args


def _resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be distinct.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Seeds must be integers in [0, 2**32).")
    for name in ("lookback", "hidden_dim", "market_embed_dim", "mlp_hidden_dim",
                 "num_heads", "num_shared_experts", "num_routed_experts",
                 "topk_routed", "balance_threshold", "epochs", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if not args.patch_scales or any(scale <= 0 for scale in args.patch_scales):
        raise ValueError("patch_scales must contain positive integers.")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads.")
    if args.topk_routed > args.num_routed_experts:
        raise ValueError("topk_routed must not exceed num_routed_experts.")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    for name in ("rank_loss_alpha", "lambda_sparse", "lambda_diversity", "lambda_lb",
                 "weight_decay", "min_delta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive.")
    if not (args.train_start <= args.train_end < args.val_start <= args.val_end
            < args.test_start <= args.test_end):
        raise ValueError("Training, validation, and test periods must be ordered and disjoint.")
    if args.device not in {"auto", "cpu", "cuda"} and not args.device.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>.")


def write_seed_summary(rows: list[dict], output_dir: Path) -> Path:
    destination = output_dir / "summaries/multi_seed_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set.intersection(*(set(row) for row in rows)))
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "n", "mean", "std"])
        for name in fields:
            if name.startswith("num_") or name == "seed":
                continue
            values = np.asarray([row[name] for row in rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metric across seeds: {name}")
            std = float(values.std(ddof=1)) if len(values) > 1 else ""
            writer.writerow([name, len(values), float(values.mean()), std])
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from mc_mose.training.trainer import train_mc_mose

    summary = args.output_dir / "summaries/multi_seed_summary.csv"
    if summary.exists():
        summary.unlink()
    rows = []
    for seed in args.seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        run_args.run_name = f"mc_mose_seed{seed}"
        print(f"Starting seed {seed}", flush=True)
        try:
            result = train_mc_mose(
                args=run_args, device=None if args.device == "auto" else args.device,
                collect_test_predictions=True, verbose=True,
            )
        except Exception as error:
            print(f"Stopped at seed {seed}: {error}", flush=True)
            raise
        rows.append({"seed": seed, **result.test_metrics})
        print(f"checkpoint={result.paths['checkpoint']}", flush=True)
        print(f"summary={result.paths['summary_json']}", flush=True)
        del result
    print(f"multi_seed_summary={write_seed_summary(rows, args.output_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
