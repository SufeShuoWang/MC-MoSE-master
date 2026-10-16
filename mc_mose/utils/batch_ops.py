from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(
    batch: Mapping[str, Any],
    device: Any,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=bool(non_blocking))
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def detach_to_cpu(batch: Mapping[str, Any]) -> dict[str, Any]:
    detached: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "detach"):
            detached[key] = value.detach().cpu()
        else:
            detached[key] = value
    return detached


def tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def select_valid_fund_rows(
    batch: Mapping[str, Any],
) -> dict[str, Any] | None:
    valid = batch["mask"].reshape(-1).bool()

    if not torch.any(valid):
        return None

    selected = dict(batch)
    for key in ("fund_seq", "market_seq", "target", "mask"):
        selected[key] = selected[key][valid]

    return selected
