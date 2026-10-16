from __future__ import annotations

import torch


def masked_pearson_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    x = x.view(-1).float()
    y = y.view(-1).float()
    mask = mask.view(-1).bool()

    x = x[mask]
    y = y[mask]

    if x.numel() < 2:
        return None

    x = x - x.mean()
    y = y - y.mean()

    denom = torch.sqrt((x.pow(2).sum() + eps) * (y.pow(2).sum() + eps))
    return (x * y).sum() / denom


def rank_1d(x: torch.Tensor) -> torch.Tensor:
    values = x.view(-1)
    if values.numel() == 0:
        return values.float()

    sorted_values, order = torch.sort(values)
    _, counts = torch.unique_consecutive(
        sorted_values,
        return_counts=True,
    )
    ends = counts.cumsum(dim=0)
    starts = ends - counts
    group_ranks = 0.5 * (starts + ends - 1).to(dtype=torch.float32)
    sorted_ranks = torch.repeat_interleave(
        group_ranks,
        counts,
        output_size=values.numel(),
    )

    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def summarize_monthly_values(values: list[torch.Tensor]) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0

    stacked = torch.stack(values)
    mean = stacked.mean()

    if stacked.numel() < 2:
        ir = torch.tensor(0.0, device=stacked.device)
    else:
        std = stacked.std(unbiased=True)
        ir = mean / std.clamp_min(1e-8)

    return float(mean.detach().cpu()), float(ir.detach().cpu())


def collect_monthly_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    ic = masked_pearson_corr(pred, target, mask)

    valid_mask = mask.view(-1).bool()
    valid_pred = pred.view(-1)[valid_mask]
    valid_target = target.view(-1)[valid_mask]
    if valid_pred.numel() < 2:
        return ic, None

    pred_rank = rank_1d(valid_pred)
    target_rank = rank_1d(valid_target)
    rank_mask = torch.ones_like(pred_rank)
    rank_ic = masked_pearson_corr(pred_rank, target_rank, rank_mask)

    return ic, rank_ic


def summarize_ic_lists(
    monthly_ic: list[torch.Tensor],
    monthly_rank_ic: list[torch.Tensor],
) -> dict[str, float | int]:
    ic, icir = summarize_monthly_values(monthly_ic)
    rank_ic, rank_icir = summarize_monthly_values(monthly_rank_ic)

    return {
        "ic": ic,
        "icir": icir,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "num_ic_months": len(monthly_ic),
        "num_rank_ic_months": len(monthly_rank_ic),
    }
