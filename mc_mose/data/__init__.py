"""Data loading utilities for MC-MoSE."""

from .final_alpha_schema import (
    FundLabelData,
    FundLabelSchema,
    load_final_alpha_npz,
    prepare_final_alpha_schema,
)
from .market_standardizer import MarketStats, fit_market_stats
from .month_sample_builder import build_month_indices, build_month_sample

__all__ = [
    "FundLabelData",
    "FundLabelSchema",
    "FundPanelMonthDataset",
    "MarketStats",
    "build_month_indices",
    "build_month_sample",
    "create_dataloaders",
    "create_dataloaders_from_args",
    "create_month_dataloader",
    "fit_market_stats",
    "load_final_alpha_npz",
    "prepare_final_alpha_schema",
]


def __getattr__(name):
    if name in {
        "FundPanelMonthDataset",
        "create_dataloaders",
        "create_dataloaders_from_args",
        "create_month_dataloader",
    }:
        from . import fund_panel_month_dataset

        return getattr(fund_panel_month_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
