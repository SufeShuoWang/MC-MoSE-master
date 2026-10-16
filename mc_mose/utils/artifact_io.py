from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "test_predictions": root / "test_predictions",
        "summaries": root / "summaries",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def format_instrument(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_default)
    return str(output_path)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    import torch

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return str(output_path)


def export_test_predictions(
    model: Any,
    loader: Any,
    device: Any,
    instruments: Any,
    output_path: str | Path,
) -> str:
    import torch

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    instrument_names = [format_instrument(value) for value in instruments]

    model.eval()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "instrument", "y_true", "y_pred"])

        with torch.no_grad():
            for batch in loader:
                fund_seq = batch["fund_seq"].to(device)
                market_seq = batch["market_seq"].to(device)
                target = batch["target"].detach().cpu().view(-1).numpy()
                mask = batch["mask"].detach().cpu().view(-1).numpy().astype(bool)
                date = int(batch["date"].item())

                pred = model(fund_seq, market_seq)
                pred = pred.detach().cpu().view(-1).numpy()

                for idx, y_pred in enumerate(pred):
                    instrument = (
                        instrument_names[idx]
                        if idx < len(instrument_names)
                        else str(idx)
                    )
                    y_true = target[idx].item() if mask[idx] else "NaN"
                    writer.writerow([date, instrument, y_true, y_pred.item()])

    return str(output)


def build_run_paths(output_dir: str | Path, run_name: str) -> dict[str, Path]:
    dirs = ensure_output_dirs(output_dir)
    return {
        **dirs,
        "checkpoint": dirs["checkpoints"] / f"{run_name}_best.pt",
        "test_predictions_csv": dirs["test_predictions"]
        / f"{run_name}_test_predictions.csv",
        "summary_json": dirs["summaries"] / f"{run_name}_summary.json",
        "log": dirs["logs"] / f"{run_name}.log",
    }
