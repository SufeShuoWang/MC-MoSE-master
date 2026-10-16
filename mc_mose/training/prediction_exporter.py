from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mc_mose.utils.artifact_io import format_instrument
from mc_mose.utils.batch_ops import move_to_device


PREDICTION_COLUMNS = ("date", "instrument", "y_true", "y_pred")


@dataclass(frozen=True)
class PredictionExportResult:
    output_path: str
    num_rows: int
    columns: tuple[str, ...] = PREDICTION_COLUMNS


def export_prediction_rows(
    prediction_rows: Iterable[Mapping[str, Any]],
    instruments: Sequence[Any],
    output_path: str | Path,
) -> PredictionExportResult:
    """Write collected prediction rows to CSV."""

    instrument_names = [format_instrument(value) for value in instruments]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    num_rows = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()

        for row in prediction_rows:
            fund_index = int(row.get("fund_index", num_rows))
            writer.writerow(
                {
                    "date": _format_date(row.get("date")),
                    "instrument": _instrument_name(instrument_names, fund_index),
                    "y_true": _format_target(row.get("y_true"), row.get("mask", True)),
                    "y_pred": _format_prediction(row.get("y_pred")),
                }
            )
            num_rows += 1

    return PredictionExportResult(
        output_path=str(output),
        num_rows=num_rows,
    )


def export_mc_mose_predictions(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    instruments: Sequence[Any],
    output_path: str | Path,
) -> PredictionExportResult:
    """Run MC-MoSE inference on a loader and write predictions to CSV."""

    rows = collect_mc_mose_prediction_rows(
        model=model,
        loader=loader,
        device=device,
    )
    return export_prediction_rows(
        prediction_rows=rows,
        instruments=instruments,
        output_path=output_path,
    )


def collect_mc_mose_prediction_rows(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    """Collect prediction rows from a MC-MoSE model and month loader."""

    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_to_device(batch, device)
            prediction = _extract_prediction(model(batch_on_device))
            rows.extend(
                _rows_from_batch(
                    prediction=prediction,
                    batch=batch_on_device,
                )
            )

    return rows


def _rows_from_batch(
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


def _extract_prediction(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, tuple) and len(model_output) > 0:
        prediction = model_output[0]
        if isinstance(prediction, torch.Tensor):
            return prediction

    if isinstance(model_output, Mapping):
        prediction = model_output.get("prediction", model_output.get("pred"))
        if isinstance(prediction, torch.Tensor):
            return prediction

    raise TypeError("model output must contain a prediction tensor.")


def _instrument_name(instrument_names: Sequence[str], fund_index: int) -> str:
    if 0 <= fund_index < len(instrument_names):
        return instrument_names[fund_index]
    return str(fund_index)


def _format_date(value: Any) -> str | int:
    if value is None:
        return ""
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _format_target(value: Any, valid: Any) -> str | float:
    if not bool(valid) or value is None:
        return "NaN"
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _format_prediction(value: Any) -> float:
    if value is None:
        raise ValueError("y_pred must not be None.")
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


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
