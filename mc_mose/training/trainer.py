from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn

from mc_mose.data.fund_panel_month_dataset import create_dataloaders
from mc_mose.utils.artifact_io import (
    build_run_paths,
    save_checkpoint,
    write_summary_json,
)
from mc_mose.utils.batch_ops import (
    move_to_device,
    select_valid_fund_rows,
)
from mc_mose.utils.seed import set_seed
from mc_mose.models import MCMoSE

from .evaluator import evaluate_mc_mose
from .loss_builder import MCMoSELossBuilder
from .prediction_exporter import PredictionExportResult, export_prediction_rows


@dataclass(frozen=True)
class EarlyStoppingState:
    patience: int
    min_delta: float = 1e-6
    best_value: float = float("inf")
    best_epoch: int = 0
    counter: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")

    def step(self, value: float, epoch: int) -> tuple["EarlyStoppingState", bool]:
        if value < self.best_value - self.min_delta:
            return (
                EarlyStoppingState(
                    patience=self.patience,
                    min_delta=self.min_delta,
                    best_value=float(value),
                    best_epoch=int(epoch),
                    counter=0,
                ),
                True,
            )

        return (
            EarlyStoppingState(
                patience=self.patience,
                min_delta=self.min_delta,
                best_value=self.best_value,
                best_epoch=self.best_epoch,
                counter=self.counter + 1,
            ),
            False,
        )

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


@dataclass
class MCMoSETrainResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_val_loss: float
    best_epoch: int
    test_metrics: dict[str, float | int]
    prediction_rows: list[dict[str, Any]]
    prediction_export: PredictionExportResult | None
    paths: dict[str, Path]
    args: dict[str, Any]
    metadata: dict[str, Any]


def train_one_epoch_mc_mose(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_builder: MCMoSELossBuilder,
    device: torch.device | str,
) -> dict[str, float]:
    model.train()

    loss_totals: torch.Tensor | None = None
    num_batches = 0
    non_blocking = _uses_non_blocking_transfer(loader, device)

    for batch in loader:
        selected_batch = select_valid_fund_rows(batch)
        if selected_batch is None:
            continue

        batch_on_device = move_to_device(
            selected_batch,
            device,
            non_blocking=non_blocking,
        )
        
        optimizer.zero_grad(set_to_none=True)
        model_output = model(batch_on_device)
        prediction, aux = _extract_prediction_and_aux(model_output)
        loss_output = loss_builder(
            prediction=prediction,
            batch=batch_on_device,
            aux=aux,
            update_balance=True,
        )
        loss_output.loss.backward()
        optimizer.step()

        tensor_metrics = loss_output.as_tensor_dict()
        values = torch.stack(
            tuple(tensor_metrics[key] for key in _LOSS_METRIC_KEYS)
        ).detach().to(dtype=torch.float64)
        loss_totals = values if loss_totals is None else loss_totals + values
        num_batches += 1

    return _average_loss_totals(loss_totals, num_batches)


def train_mc_mose(
    args: Any,
    device: torch.device | str | None = None,
    collect_test_predictions: bool = True,
    verbose: bool = True,
) -> MCMoSETrainResult:
    seed = int(_get_arg(args, "seed", 42))
    set_seed(seed)

    runtime_device = _resolve_device(
        device if device is not None else _get_arg(args, "device", None)
    )
    run_name = str(_get_arg(args, "run_name", "mc_mose"))
    output_dir = _get_arg(args, "output_dir", "outputs")
    paths = build_run_paths(output_dir=output_dir, run_name=run_name)
    _reset_log(paths["log"])

    loader_started = time.perf_counter()
    train_loader, val_loader, test_loader, metadata = _create_loaders(args)
    loader_seconds = time.perf_counter() - loader_started

    model = build_mc_mose_model(args=args, metadata=metadata).to(runtime_device)
    optimizer = build_optimizer_from_args(args=args, model=model)
    loss_builder = build_loss_builder_from_args(args)
    if runtime_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)
    early_state = EarlyStoppingState(
        patience=int(_get_arg(args, "patience", 3)),
        min_delta=float(_get_arg(args, "min_delta", 1e-6)),
    )

    history: list[dict[str, float | int]] = []
    args_dict = _args_to_dict(args)
    args_dict.setdefault("version", "v1")
    args_dict["run_name"] = run_name
    args_dict["output_dir"] = str(output_dir)

    _log(
        paths["log"],
        (
            f"run_name={run_name} device={runtime_device} seed={seed} "
            f"loader_seconds={loader_seconds:.3f}"
        ),
        verbose=verbose,
    )

    epochs = int(_get_arg(args, "epochs", 5))
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    test_every_epoch = bool(_get_arg(args, "test_every_epoch", False))

    for epoch in range(1, epochs + 1):
        train_started = time.perf_counter()
        train_metrics = train_one_epoch_mc_mose(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_builder=loss_builder,
            device=runtime_device,
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        val_output = evaluate_mc_mose(
            model=model,
            loader=val_loader,
            loss_builder=loss_builder,
            device=runtime_device,
            collect_predictions=False,
        )
        val_seconds = time.perf_counter() - val_started
        val_metrics = val_output.metrics
        epoch_test_metrics: Mapping[str, float | int] = {}
        test_seconds: float | None = None
        if test_every_epoch:
            test_started = time.perf_counter()
            epoch_test_metrics = evaluate_mc_mose(
                model=model,
                loader=test_loader,
                loss_builder=loss_builder,
                device=runtime_device,
                collect_predictions=False,
            ).metrics
            test_seconds = time.perf_counter() - test_started
        early_state, improved = early_state.step(
            value=float(val_metrics["loss"]),
            epoch=epoch,
        )

        row = _build_history_row(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=epoch_test_metrics,
            best_val_loss=early_state.best_value,
            early_stop_counter=early_state.counter,
        )
        row["train_seconds"] = train_seconds
        row["val_seconds"] = val_seconds
        if test_seconds is not None:
            row["test_seconds"] = test_seconds
        history.append(row)

        _log(paths["log"], _format_epoch_row(row), verbose=verbose)

        if improved:
            save_checkpoint(
                paths["checkpoint"],
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_builder_state_dict": loss_builder.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": dict(val_metrics),
                    "test_metrics": dict(epoch_test_metrics),
                    "args": args_dict,
                    "metadata": metadata,
                },
            )

        if early_state.should_stop:
            _log(paths["log"], f"early_stop_epoch={epoch}", verbose=verbose)
            break

    _load_best_checkpoint(
        model=model,
        checkpoint_path=paths["checkpoint"],
        device=runtime_device,
    )

    final_test_started = time.perf_counter()
    final_test_output = evaluate_mc_mose(
        model=model,
        loader=test_loader,
        loss_builder=loss_builder,
        device=runtime_device,
        collect_predictions=collect_test_predictions,
    )
    final_test_seconds = time.perf_counter() - final_test_started
    test_metrics = final_test_output.metrics
    prediction_export = None
    prediction_export_seconds = 0.0
    if collect_test_predictions:
        prediction_export_started = time.perf_counter()
        prediction_export = export_prediction_rows(
            prediction_rows=final_test_output.prediction_rows,
            instruments=metadata["instruments"],
            output_path=paths["test_predictions_csv"],
        )
        prediction_export_seconds = time.perf_counter() - prediction_export_started

    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(runtime_device))
        if runtime_device.type == "cuda"
        else None
    )
    summary = {
        "args": args_dict,
        "metadata": metadata,
        "expert_hparams": _expert_hparams_from_args(args),
        "expert_diagnostics": {},
        "best_val_loss": early_state.best_value,
        "best_epoch": early_state.best_epoch,
        "test_metrics": dict(test_metrics),
        "history": history,
        "best_checkpoint": str(paths["checkpoint"]),
        "test_predictions": prediction_export.output_path if prediction_export else None,
        "num_test_prediction_rows": (
            prediction_export.num_rows if prediction_export else 0
        ),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "timing": {
            "loader_seconds": loader_seconds,
            "final_test_seconds": final_test_seconds,
            "prediction_export_seconds": prediction_export_seconds,
        },
    }
    write_summary_json(paths["summary_json"], summary)

    _log(paths["log"], f"test_metrics={dict(test_metrics)}", verbose=verbose)
    if peak_gpu_memory_bytes is not None:
        _log(
            paths["log"],
            f"peak_gpu_memory_mb={peak_gpu_memory_bytes / (1024 ** 2):.1f}",
            verbose=verbose,
        )
    if prediction_export:
        _log(
            paths["log"],
            f"test_predictions={prediction_export.output_path}",
            verbose=verbose,
        )
    _log(paths["log"], f"summary={paths['summary_json']}", verbose=verbose)

    return MCMoSETrainResult(
        model=model,
        history=history,
        best_val_loss=early_state.best_value,
        best_epoch=early_state.best_epoch,
        test_metrics=dict(test_metrics),
        prediction_rows=final_test_output.prediction_rows,
        prediction_export=prediction_export,
        paths=paths,
        args=args_dict,
        metadata=dict(metadata),
    )


def build_mc_mose_model(args: Any, metadata: Mapping[str, Any]) -> MCMoSE:
    fund_dim = int(_get_arg(args, "fund_dim", metadata["fund_dim"]))
    market_dim = int(_get_arg(args, "market_dim", metadata["market_dim"]))

    return MCMoSE(
        fund_dim=fund_dim,
        market_dim=market_dim,
        lookback=int(_get_arg(args, "lookback", metadata.get("lookback", 10))),
        patch_scales=_parse_patch_scales(
            _get_arg(args, "patch_scales", [2, 3, 4, 5, 6])
        ),
        hidden_dim=int(_get_arg(args, "hidden_dim", 128)),
        market_embed_dim=int(_get_arg(args, "market_embed_dim", 16)),
        num_shared_experts=int(_get_arg(args, "num_shared_experts", 12)),
        num_routed_experts=int(_get_arg(args, "num_routed_experts", 48)),
        topk_routed=int(_get_arg(args, "topk_routed", 12)),
        num_heads=int(_get_arg(args, "num_heads", 4)),
        mlp_hidden_dim=int(_get_arg(args, "mlp_hidden_dim", 64)),
        dropout=float(_get_arg(args, "dropout", 0.1)),
    )


def build_loss_builder_from_args(args: Any) -> MCMoSELossBuilder:
    return MCMoSELossBuilder(
        rank_loss_alpha=float(_get_arg(args, "rank_loss_alpha", 1.0)),
        lambda_sparse=float(_get_arg(args, "lambda_sparse", 0.0)),
        lambda_diversity=float(_get_arg(args, "lambda_diversity", 0.0)),
        lambda_lb=float(_get_arg(args, "lambda_lb", 0.0)),
        balance_threshold=int(_get_arg(args, "balance_threshold", 2)),
        num_routed_experts=int(_get_arg(args, "num_routed_experts", 0)),
        topk_routed=int(_get_arg(args, "topk_routed", 0)),
    )


def build_optimizer_from_args(
    args: Any,
    model: nn.Module,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(_get_arg(args, "lr", 1e-3)),
        weight_decay=float(_get_arg(args, "weight_decay", 1e-4)),
    )


def _create_loaders(args: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    return create_dataloaders(
        data_path=_get_arg(args, "data_path", "data/fund_2023_2024.npz"),
        target_name=str(_get_arg(args, "target_name", "alpha")),
        lookback=int(_get_arg(args, "lookback", 10)),
        train_start=int(_get_arg(args, "train_start", 202310)),
        train_end=int(_get_arg(args, "train_end", 202406)),
        val_start=int(_get_arg(args, "val_start", 202407)),
        val_end=int(_get_arg(args, "val_end", 202409)),
        test_start=int(_get_arg(args, "test_start", 202410)),
        test_end=int(_get_arg(args, "test_end", 202412)),
        num_workers=int(_get_arg(args, "num_workers", 0)),
        train_valid_only=bool(_get_arg(args, "train_valid_only", True)),
        val_valid_only=True,
        cache_samples=bool(_get_arg(args, "cache_samples", True)),
        pin_memory=bool(_get_arg(args, "pin_memory", True)),
        persistent_workers=bool(_get_arg(args, "persistent_workers", False)),
        prefetch_factor=int(_get_arg(args, "prefetch_factor", 2)),
    )


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        if isinstance(device, str) and device.lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    if not checkpoint_path.exists():
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def _extract_prediction_and_aux(
    model_output: Any,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if isinstance(model_output, torch.Tensor):
        return model_output, {}
    if isinstance(model_output, tuple) and len(model_output) > 0:
        prediction = model_output[0]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("the first model output must be a torch.Tensor.")
        aux: Mapping[str, Any] = {}
        if len(model_output) > 1:
            if not isinstance(model_output[1], Mapping):
                raise TypeError("the second model output must be an aux mapping.")
            aux = model_output[1]
        return prediction, aux
    if isinstance(model_output, Mapping):
        prediction = model_output.get("prediction", model_output.get("pred"))
        if isinstance(prediction, torch.Tensor):
            aux = model_output.get("aux", model_output)
            if not isinstance(aux, Mapping):
                raise TypeError("model_output['aux'] must be a mapping.")
            return prediction, aux
    raise TypeError("model output must contain a prediction tensor.")


def _build_history_row(
    epoch: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float | int],
    test_metrics: Mapping[str, float | int],
    best_val_loss: float,
    early_stop_counter: int,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "early_stop_counter": int(early_stop_counter),
    }
    for prefix, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ):
        for key in _METRIC_KEYS:
            if key in metrics:
                row[f"{prefix}_{key}"] = float(metrics[key])
    return row


def _format_epoch_row(row: Mapping[str, float | int]) -> str:
    parts = [f"epoch={int(row['epoch']):03d}"]
    for key in (
        "train_loss",
        "train_base_loss",
        "train_reg_loss",
        "train_rank_loss",
        "train_sparse_loss",
        "train_diversity_loss",
        "train_load_balance_loss",
        "val_loss",
        "val_ic",
        "val_rank_ic",
        "test_loss",
        "test_ic",
        "test_rank_ic",
        "train_seconds",
        "val_seconds",
        "test_seconds",
    ):
        if key in row:
            parts.append(f"{key}={float(row[key]):.6f}")
    parts.append(f"counter={int(row['early_stop_counter'])}")
    return " ".join(parts)


def _average_loss_totals(
    totals: torch.Tensor | None,
    num_batches: int,
) -> dict[str, float]:
    denom = max(num_batches, 1)
    if totals is None:
        values = [0.0] * len(_LOSS_METRIC_KEYS)
    else:
        values = (totals / denom).cpu().tolist()
    return {
        key: float(value)
        for key, value in zip(_LOSS_METRIC_KEYS, values, strict=True)
    }


def _uses_non_blocking_transfer(
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
) -> bool:
    return (
        torch.device(device).type == "cuda"
        and bool(getattr(loader, "pin_memory", False))
    )


def _reset_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _log(path: Path, message: str, verbose: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    if verbose:
        print(message)


def _parse_patch_scales(value: Any) -> list[int]:
    if isinstance(value, str):
        cleaned = value.strip().strip("[]()")
        if not cleaned:
            raise ValueError("patch_scales must not be empty.")
        return [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    return [int(part) for part in value]


def _expert_hparams_from_args(args: Any) -> dict[str, Any]:
    return {
        "num_shared_experts": int(_get_arg(args, "num_shared_experts", 12)),
        "num_routed_experts": int(_get_arg(args, "num_routed_experts", 48)),
        "topk_routed": int(_get_arg(args, "topk_routed", 12)),
        "lambda_sparse": float(_get_arg(args, "lambda_sparse", 0.0)),
        "lambda_diversity": float(_get_arg(args, "lambda_diversity", 0.0)),
        "lambda_lb": float(_get_arg(args, "lambda_lb", 0.0)),
        "balance_threshold": int(_get_arg(args, "balance_threshold", 2)),
    }


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _args_to_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, Mapping):
        items = args.items()
    else:
        items = vars(args).items()
    return {str(key): _to_plain_value(value) for key, value in items}


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


_LOSS_METRIC_KEYS = (
    "loss",
    "base_loss",
    "reg_loss",
    "rank_loss",
    "sparse_loss",
    "diversity_loss",
    "load_balance_loss",
)

_METRIC_KEYS = (
    *_LOSS_METRIC_KEYS,
    "ic",
    "icir",
    "rank_ic",
    "rank_icir",
    "num_ic_months",
    "num_rank_ic_months",
)
