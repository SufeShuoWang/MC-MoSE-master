from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .final_alpha_schema import FundLabelData, load_final_alpha_npz
from .market_standardizer import MarketStats, fit_market_stats
from .month_sample_builder import build_month_indices, build_month_sample


@dataclass(frozen=True)
class DataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    metadata: dict[str, Any]


class FundPanelMonthDataset(Dataset):
    def __init__(
        self,
        panel: FundLabelData,
        month_indices: np.ndarray,
        lookback: int,
        market_stats: MarketStats,
        valid_only: bool = False,
        cache_samples: bool = False,
    ) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        self.panel = panel
        self.month_indices = np.asarray(month_indices, dtype=np.int64)
        self.lookback = int(lookback)
        self.market_stats = market_stats
        self.valid_only = bool(valid_only)
        self.cache_samples = bool(cache_samples)
        self._cached_samples: tuple[dict[str, torch.Tensor], ...] | None = None
        if self.cache_samples:
            # Cache tensor objects once and treat them as immutable. __getitem__
            # returns a new mapping so downstream code cannot alter the cache map.
            self._cached_samples = tuple(
                self._build_tensor_sample(index)
                for index in range(int(self.month_indices.shape[0]))
            )

    def __len__(self) -> int:
        return int(self.month_indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._cached_samples is not None:
            return dict(self._cached_samples[idx])
        return self._build_tensor_sample(idx)

    def _build_tensor_sample(self, idx: int) -> dict[str, torch.Tensor]:
        time_index = int(self.month_indices[idx])
        sample = build_month_sample(
            data=self.panel.data,
            dates=self.panel.dates,
            time_index=time_index,
            lookback=self.lookback,
            schema=self.panel.schema,
            market_stats=self.market_stats,
            valid_only=self.valid_only,
        )
        return {
            "fund_seq": torch.as_tensor(sample.fund_seq, dtype=torch.float32),
            "market_seq": torch.as_tensor(sample.market_seq, dtype=torch.float32),
            "target": torch.as_tensor(sample.target, dtype=torch.float32),
            "mask": torch.as_tensor(sample.mask, dtype=torch.float32),
            "date": torch.as_tensor(int(sample.date), dtype=torch.long),
            "time_index": torch.as_tensor(int(sample.time_index), dtype=torch.long),
        }

    @property
    def fund_dim(self) -> int:
        return self.panel.schema.fund_dim

    @property
    def market_dim(self) -> int:
        return self.panel.schema.market_dim

    @property
    def dates(self) -> np.ndarray:
        return self.panel.dates[self.month_indices]


def create_month_dataloader(
    dataset: FundPanelMonthDataset,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    worker_count = int(num_workers)
    if worker_count < 0:
        raise ValueError("num_workers must be non-negative.")

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "shuffle": bool(shuffle),
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
    }
    if worker_count > 0:
        if int(prefetch_factor) <= 0:
            raise ValueError("prefetch_factor must be positive when num_workers > 0.")
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**loader_kwargs)


def _require_non_empty_split(name: str, indices: np.ndarray) -> None:
    if indices.size == 0:
        raise ValueError(f"{name} split has no month indices.")


def create_dataloaders(
    data_path: str | Path,
    target_name: str = "alpha",
    lookback: int = 6,
    train_start: int = 201201,
    train_end: int = 202012,
    val_start: int = 202101,
    val_end: int = 202212,
    test_start: int = 202301,
    test_end: int = 202412,
    num_workers: int = 0,
    train_valid_only: bool = False,
    val_valid_only: bool = False,
    cache_samples: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    panel = load_final_alpha_npz(data_path, target_name=target_name)
    schema = panel.schema

    train_indices = build_month_indices(panel.dates, lookback, train_start, train_end)
    val_indices = build_month_indices(panel.dates, lookback, val_start, val_end)
    test_indices = build_month_indices(panel.dates, lookback, test_start, test_end)

    _require_non_empty_split("train", train_indices)
    _require_non_empty_split("val", val_indices)
    _require_non_empty_split("test", test_indices)

    market_fit_start_index = (
        int(train_indices[0]) - int(lookback) + 1)
    market_fit_start = int(
        panel.dates[market_fit_start_index])
    
    market_stats = fit_market_stats(
        data=panel.data,
        dates=panel.dates,
        market_indices=schema.market_indices,
        train_start=market_fit_start,
        train_end=train_end,
        feature_names=schema.market_feature_names,
    )

    train_dataset = FundPanelMonthDataset(
        panel=panel,
        month_indices=train_indices,
        lookback=lookback,
        market_stats=market_stats,
        valid_only=train_valid_only,
        cache_samples=cache_samples,
    )
    val_dataset = FundPanelMonthDataset(
        panel=panel,
        month_indices=val_indices,
        lookback=lookback,
        market_stats=market_stats,
        valid_only=val_valid_only,
        cache_samples=cache_samples,
    )
    test_dataset = FundPanelMonthDataset(
        panel=panel,
        month_indices=test_indices,
        lookback=lookback,
        market_stats=market_stats,
        valid_only=False,
        cache_samples=False,
    )

    train_loader = create_month_dataloader(
        train_dataset,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = create_month_dataloader(
        val_dataset,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = create_month_dataloader(
        test_dataset,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    metadata = {
        **panel.as_metadata(),
        **market_stats.as_metadata(),
        "lookback": int(lookback),
        "train_start": int(train_start),
        "train_end": int(train_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "test_end": int(test_end),
        "train_valid_only": bool(train_valid_only),
        "val_valid_only": bool(val_valid_only),
        "test_valid_only": False,
        "cache_samples": bool(cache_samples),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": (
            bool(persistent_workers) if int(num_workers) > 0 else False
        ),
        "prefetch_factor": (
            int(prefetch_factor) if int(num_workers) > 0 else None
        ),
        "split_month_counts": {
            "train": int(train_indices.size),
            "val": int(val_indices.size),
            "test": int(test_indices.size),
        },
        "split_dates": {
            "train": panel.dates[train_indices].astype(int).tolist(),
            "val": panel.dates[val_indices].astype(int).tolist(),
            "test": panel.dates[test_indices].astype(int).tolist(),
        },
    }

    return train_loader, val_loader, test_loader, metadata


def create_dataloaders_from_args(args: Any) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    return create_dataloaders(
        data_path=args.data_path,
        target_name=getattr(args, "target_name", "alpha"),
        lookback=int(args.lookback),
        train_start=int(args.train_start),
        train_end=int(args.train_end),
        val_start=int(args.val_start),
        val_end=int(args.val_end),
        test_start=int(args.test_start),
        test_end=int(args.test_end),
        num_workers=int(getattr(args, "num_workers", 0)),
        train_valid_only=bool(getattr(args, "train_valid_only", False)),
        val_valid_only=bool(getattr(args, "val_valid_only", False)),
        cache_samples=bool(getattr(args, "cache_samples", False)),
        pin_memory=bool(getattr(args, "pin_memory", False)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)),
    )
