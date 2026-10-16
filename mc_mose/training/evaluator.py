from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from mc_mose.utils.batch_ops import move_to_device, select_valid_fund_rows
from mc_mose.utils.metric_utils import collect_monthly_ic, summarize_ic_lists

from .loss_builder import MCMoSELossBuilder, build_loss_builder_from_args


@dataclass(frozen=True)
class MCMoSEEvaluationOutput:
    metrics: dict[str, float | int]
    prediction_rows: list[dict[str, Any]]

    def as_log_dict(self) -> dict[str, float | int]:
        return dict(self.metrics)


class MCMoSEEvaluator:
    """Evaluate MC-MoSE on month batches."""

    def __init__(
        self,
        loss_builder: MCMoSELossBuilder,
        device: torch.device | str | None = None,
        collect_predictions: bool = False,
    ) -> None:
        self.loss_builder = loss_builder
        self.device = device
        self.collect_predictions = bool(collect_predictions)

    def __call__(
        self,
        model: nn.Module,
        loader: Iterable[Mapping[str, torch.Tensor]],
    ) -> MCMoSEEvaluationOutput:
        return evaluate_mc_mose(
            model=model,
            loader=loader,
            loss_builder=self.loss_builder,
            device=self.device,
            collect_predictions=self.collect_predictions,
        )


def evaluate_mc_mose(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    loss_builder: MCMoSELossBuilder,
    device: torch.device | str | None = None,
    collect_predictions: bool = False,
) -> MCMoSEEvaluationOutput:
    """Run evaluation and return loss, IC metrics, and optional predictions."""

    model.eval()
    _validate_prediction_loader(loader, collect_predictions)

    loss_totals: torch.Tensor | None = None
    num_batches = 0
    non_blocking = _uses_non_blocking_transfer(loader, device)

    monthly_ic: list[torch.Tensor] = []
    monthly_rank_ic: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            evaluation_batch = batch
            if not collect_predictions:
                selected_batch = select_valid_fund_rows(batch)
                if selected_batch is None:
                    num_batches += 1
                    continue
                evaluation_batch = selected_batch

            batch_on_device = _prepare_batch(
                evaluation_batch,
                device,
                non_blocking=non_blocking,
            )
            model_output = model(batch_on_device)
            prediction, aux = _extract_prediction_and_aux(model_output)

            loss_output = loss_builder(
                prediction=prediction,
                batch=batch_on_device,
                aux=aux,
                update_balance=False,
            )
            values = torch.stack(
                (
                    loss_output.loss,
                    loss_output.base_loss,
                    loss_output.reg_loss,
                    loss_output.rank_loss,
                    loss_output.sparse_loss,
                    loss_output.diversity_loss,
                    loss_output.load_balance_loss,
                )
            ).detach().to(dtype=torch.float64)
            loss_totals = values if loss_totals is None else loss_totals + values
            num_batches += 1

            target = batch_on_device["target"].reshape(-1)
            mask = batch_on_device["mask"].reshape(-1)
            ic, rank_ic = collect_monthly_ic(prediction, target, mask)
            if ic is not None:
                monthly_ic.append(ic.detach())
            if rank_ic is not None:
                monthly_rank_ic.append(rank_ic.detach())

            if collect_predictions:
                prediction_rows.extend(
                    _collect_prediction_rows(
                        prediction=prediction,
                        batch=batch_on_device,
                    )
                )

    denom = max(num_batches, 1)
    if loss_totals is None:
        averaged_losses = [0.0] * 7
    else:
        averaged_losses = (loss_totals / denom).cpu().tolist()
    metrics: dict[str, float | int] = {
        "loss": float(averaged_losses[0]),
        "base_loss": float(averaged_losses[1]),
        "reg_loss": float(averaged_losses[2]),
        "rank_loss": float(averaged_losses[3]),
        "sparse_loss": float(averaged_losses[4]),
        "diversity_loss": float(averaged_losses[5]),
        "load_balance_loss": float(averaged_losses[6]),
    }
    metrics.update(summarize_ic_lists(monthly_ic, monthly_rank_ic))

    return MCMoSEEvaluationOutput(
        metrics=metrics,
        prediction_rows=prediction_rows,
    )


def build_evaluator_from_args(
    args: Any,
    device: torch.device | str | None = None,
    collect_predictions: bool = False,
) -> MCMoSEEvaluator:
    return MCMoSEEvaluator(
        loss_builder=build_loss_builder_from_args(args),
        device=device,
        collect_predictions=collect_predictions,
    )


def _prepare_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str | None,
    *,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    if device is None:
        return dict(batch)
    return move_to_device(batch, device, non_blocking=non_blocking)


def _validate_prediction_loader(
    loader: Iterable[Mapping[str, torch.Tensor]],
    collect_predictions: bool,
) -> None:
    if not collect_predictions:
        return
    dataset = getattr(loader, "dataset", None)
    if bool(getattr(dataset, "valid_only", False)):
        raise ValueError(
            "Prediction export requires a full-universe loader; "
            "valid_only datasets would corrupt fund-index mapping."
        )


def _uses_non_blocking_transfer(
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str | None,
) -> bool:
    if device is None or not bool(getattr(loader, "pin_memory", False)):
        return False
    return torch.device(device).type == "cuda"


def _extract_prediction_and_aux(
    model_output: Any,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if isinstance(model_output, torch.Tensor):
        return model_output, {}

    if isinstance(model_output, tuple):
        if len(model_output) == 0:
            raise ValueError("model returned an empty tuple.")
        prediction = model_output[0]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("the first model output must be a torch.Tensor.")

        aux: Mapping[str, Any] = {}
        if len(model_output) > 1:
            aux_candidate = model_output[1]
            if not isinstance(aux_candidate, Mapping):
                raise TypeError("the second model output must be an aux mapping.")
            aux = aux_candidate
        return prediction, aux

    if isinstance(model_output, Mapping):
        prediction = model_output.get("prediction", model_output.get("pred"))
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(
                "model_output must contain a prediction tensor under "
                "'prediction' or 'pred'."
            )

        aux_candidate = model_output.get("aux")
        if aux_candidate is None:
            aux_candidate = model_output
        if not isinstance(aux_candidate, Mapping):
            raise TypeError("model_output['aux'] must be a mapping when provided.")
        return prediction, aux_candidate

    raise TypeError("model output must contain a prediction tensor and optional aux.")


def _collect_prediction_rows(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    pred_values = prediction.detach().cpu().reshape(-1).tolist()
    target_values = batch["target"].detach().cpu().reshape(-1).tolist()
    mask_values = batch["mask"].detach().cpu().reshape(-1).bool().tolist()
    date = _optional_int_scalar(batch, "date")
    time_index = _optional_int_scalar(batch, "time_index")

    rows: list[dict[str, Any]] = []
    for fund_index, y_pred in enumerate(pred_values):
        valid = bool(mask_values[fund_index])
        rows.append(
            {
                "date": date,
                "time_index": time_index,
                "fund_index": fund_index,
                "y_true": float(target_values[fund_index]) if valid else None,
                "y_pred": float(y_pred),
                "mask": valid,
            }
        )
    return rows


def _optional_int_scalar(
    batch: Mapping[str, torch.Tensor],
    key: str,
) -> int | None:
    if key not in batch:
        return None
    value = batch[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)
