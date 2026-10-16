from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from .numerical_determinism import (
    FLOAT64_EPSILON,
    RESIDUAL_ZERO_MULTIPLIER,
    deterministic_ols,
    stable_fsum,
    stable_sample_variance,
)


OUTPUT_COLUMNS = [
    "permno",
    "month",
    "IdioVol",
    "Resid_Var",
    "Spread",
    "SUV",
    "Variance",
    "n_ff3",
    "n_ff3_2",
    "n_suv",
    "n_ret_2",
    "n_spread",
    "idio_rank",
    "resid_var_rank",
    "suv_rank",
    "idio_zero_residual",
    "resid_var_zero_residual",
    "suv_zero_residual",
]

SUV_COLUMN_NAME = "SUV"
SUV_METHOD_LEGACY = "legacy_same_sample_zero"
SUV_METHOD_MONTHLY_OOS = "monthly_oos_suv_v1"
SUV_MISSING_REASONS = (
    "insufficient_estimation_obs",
    "insufficient_evaluation_obs",
    "rank_deficient",
    "zero_sigma",
    "missing_daily_input",
)
SUV_OOS_OUTPUT_COLUMNS = [
    "permno",
    "month",
    SUV_COLUMN_NAME,
    "n_est",
    "n_eval",
    "sigma_hat",
    "estimation_start",
    "estimation_end",
    "evaluation_start",
    "evaluation_end",
    "method",
    "missing_reason",
]


def _valid_suv_observation(
    returns: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(returns)
        & np.isfinite(volumes)
        & (returns != -99.99)
        & (volumes != -99.99)
    )


def legacy_same_sample_zero(
    returns: np.ndarray,
    volumes: np.ndarray,
    *,
    min_obs: int = 20,
) -> dict[str, Any]:
    """Retain the deprecated same-sample SUV rule used by the v3.1 history."""
    simple = np.asarray(returns, dtype=np.float64)
    volume = np.asarray(volumes, dtype=np.float64)
    if simple.ndim != 1 or volume.ndim != 1 or len(simple) != len(volume):
        return {
            SUV_COLUMN_NAME: np.nan,
            "n_suv": 0,
            "suv_rank": 0,
            "suv_zero_residual": False,
            "missing_reason": "missing_daily_input",
            "method": SUV_METHOD_LEGACY,
            "deprecated": True,
        }
    valid = _valid_suv_observation(simple, volume)
    response = volume[valid]
    predictors = np.column_stack(
        [
            np.maximum(simple[valid], 0.0),
            np.maximum(-simple[valid], 0.0),
        ]
    )
    result: dict[str, Any] = {
        SUV_COLUMN_NAME: np.nan,
        "n_suv": int(len(response)),
        "suv_rank": 0,
        "suv_zero_residual": False,
        "missing_reason": (
            "insufficient_estimation_obs" if len(response) < min_obs else None
        ),
        "method": SUV_METHOD_LEGACY,
        "deprecated": True,
    }
    if len(response) < min_obs:
        return result
    fit = deterministic_ols(response, predictors, ddof=1)
    result["suv_rank"] = fit.rank
    result["suv_zero_residual"] = fit.residual_sum_of_squares == 0.0
    if fit.rank < 3:
        result["missing_reason"] = "rank_deficient"
    elif (
        fit.residual_sum_of_squares > fit.residual_zero_threshold
        and fit.residual_standard_deviation > 0
    ):
        # With an intercept, the mean in-sample OLS residual is mechanical zero.
        result[SUV_COLUMN_NAME] = 0.0
        result["missing_reason"] = None
    else:
        result["missing_reason"] = "zero_sigma"
    return result


def _empty_oos_result(
    evaluation_start: np.datetime64 | None,
    evaluation_end: np.datetime64 | None,
    reason: str,
    *,
    n_est: int = 0,
    n_eval: int = 0,
    rank: int = 0,
    estimation_start: np.datetime64 | None = None,
    estimation_end: np.datetime64 | None = None,
    sigma_hat: float = np.nan,
) -> dict[str, Any]:
    if reason not in SUV_MISSING_REASONS:
        raise ValueError(f"Unknown SUV missing reason: {reason}")
    return {
        SUV_COLUMN_NAME: np.nan,
        "estimation_start": estimation_start,
        "estimation_end": estimation_end,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "n_est": int(n_est),
        "n_eval": int(n_eval),
        "sigma_hat": float(sigma_hat),
        "suv_rank": int(rank),
        "missing_reason": reason,
        "method": SUV_METHOD_MONTHLY_OOS,
    }


def _monthly_oos_suv_sorted(
    dates: np.ndarray,
    returns: np.ndarray,
    volumes: np.ndarray,
    evaluation_month: str | np.datetime64 | pd.Timestamp,
    *,
    estimation_window: int,
    min_estimation_obs: int,
    min_evaluation_obs: int,
    gap_trading_days: int,
    sigma_floor: float,
) -> dict[str, Any]:
    evaluation_month_value = np.datetime64(evaluation_month, "M")
    evaluation_start = evaluation_month_value.astype("datetime64[D]")
    evaluation_end_exclusive = (
        evaluation_month_value + np.timedelta64(1, "M")
    ).astype("datetime64[D]")
    evaluation_end = evaluation_end_exclusive - np.timedelta64(1, "D")
    if (
        dates.ndim != 1
        or returns.ndim != 1
        or volumes.ndim != 1
        or not (len(dates) == len(returns) == len(volumes))
        or len(dates) == 0
    ):
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "missing_daily_input",
        )
    valid = (~np.isnat(dates)) & _valid_suv_observation(returns, volumes)
    if not valid.any():
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "missing_daily_input",
        )
    valid_positions = np.flatnonzero(valid)
    valid_dates = dates[valid_positions]
    estimation_stop = int(np.searchsorted(valid_dates, evaluation_start, side="left"))
    if gap_trading_days:
        estimation_stop = max(0, estimation_stop - gap_trading_days)
    estimation_positions = valid_positions[
        max(0, estimation_stop - estimation_window) : estimation_stop
    ]
    evaluation_left = int(
        np.searchsorted(valid_dates, evaluation_start, side="left")
    )
    evaluation_right = int(
        np.searchsorted(valid_dates, evaluation_end_exclusive, side="left")
    )
    evaluation_positions = valid_positions[evaluation_left:evaluation_right]
    n_est = len(estimation_positions)
    n_eval = len(evaluation_positions)
    estimation_start = (
        dates[estimation_positions[0]] if n_est else None
    )
    estimation_end = dates[estimation_positions[-1]] if n_est else None
    if n_est < min_estimation_obs:
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "insufficient_estimation_obs",
            n_est=n_est,
            n_eval=n_eval,
            estimation_start=estimation_start,
            estimation_end=estimation_end,
        )
    if n_eval < min_evaluation_obs:
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "insufficient_evaluation_obs",
            n_est=n_est,
            n_eval=n_eval,
            estimation_start=estimation_start,
            estimation_end=estimation_end,
        )
    estimation_returns = returns[estimation_positions]
    estimation_volumes = volumes[estimation_positions]
    estimation_predictors = np.column_stack(
        [
            np.maximum(estimation_returns, 0.0),
            np.maximum(-estimation_returns, 0.0),
        ]
    ).astype(np.float64, copy=False)
    fit = deterministic_ols(
        estimation_volumes.astype(np.float64, copy=False),
        estimation_predictors,
        ddof=1,
    )
    if fit.rank < 3:
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "rank_deficient",
            n_est=n_est,
            n_eval=n_eval,
            rank=fit.rank,
            estimation_start=estimation_start,
            estimation_end=estimation_end,
        )
    denominator = n_est - 3
    sigma_hat = (
        float(np.sqrt(fit.residual_sum_of_squares / denominator))
        if denominator > 0
        else np.nan
    )
    if not np.isfinite(sigma_hat) or sigma_hat <= sigma_floor:
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "zero_sigma",
            n_est=n_est,
            n_eval=n_eval,
            rank=fit.rank,
            estimation_start=estimation_start,
            estimation_end=estimation_end,
            sigma_hat=sigma_hat,
        )
    evaluation_returns = returns[evaluation_positions]
    evaluation_volumes = volumes[evaluation_positions]
    evaluation_design = np.column_stack(
        [
            np.ones(n_eval, dtype=np.float64),
            np.maximum(evaluation_returns, 0.0),
            np.maximum(-evaluation_returns, 0.0),
        ]
    )
    predicted = evaluation_design @ fit.coefficients
    daily_suv = (evaluation_volumes - predicted) / sigma_hat
    monthly_suv = stable_fsum(daily_suv) / n_eval
    if not np.isfinite(monthly_suv):
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "missing_daily_input",
            n_est=n_est,
            n_eval=n_eval,
            rank=fit.rank,
            estimation_start=estimation_start,
            estimation_end=estimation_end,
            sigma_hat=sigma_hat,
        )
    return {
        SUV_COLUMN_NAME: float(monthly_suv),
        "estimation_start": estimation_start,
        "estimation_end": estimation_end,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "n_est": int(n_est),
        "n_eval": int(n_eval),
        "sigma_hat": sigma_hat,
        "suv_rank": int(fit.rank),
        "missing_reason": None,
        "method": SUV_METHOD_MONTHLY_OOS,
    }


def monthly_oos_suv_v1(
    dates: np.ndarray,
    returns: np.ndarray,
    volumes: np.ndarray,
    evaluation_month: str | np.datetime64 | pd.Timestamp,
    *,
    estimation_window: int = 60,
    min_estimation_obs: int = 40,
    min_evaluation_obs: int = 10,
    gap_trading_days: int = 0,
    sigma_floor: float = 1e-12,
) -> dict[str, Any]:
    """Compute one Garfinkel-style monthly out-of-sample abnormal volume."""
    if estimation_window < min_estimation_obs:
        raise ValueError("estimation_window must be at least min_estimation_obs")
    if min_estimation_obs <= 3:
        raise ValueError("min_estimation_obs must exceed the three OLS parameters")
    if min_evaluation_obs < 1 or gap_trading_days < 0 or sigma_floor < 0:
        raise ValueError("Invalid monthly_oos_suv_v1 configuration")
    dates_array = np.asarray(dates).astype("datetime64[D]")
    returns_array = np.asarray(returns, dtype=np.float64)
    volumes_array = np.asarray(volumes, dtype=np.float64)
    if (
        dates_array.ndim != 1
        or returns_array.ndim != 1
        or volumes_array.ndim != 1
        or not (
            len(dates_array) == len(returns_array) == len(volumes_array)
        )
    ):
        evaluation_start = np.datetime64(evaluation_month, "M").astype(
            "datetime64[D]"
        )
        evaluation_end = (
            (np.datetime64(evaluation_month, "M") + np.timedelta64(1, "M")).astype(
                "datetime64[D]"
            )
            - np.timedelta64(1, "D")
        )
        return _empty_oos_result(
            evaluation_start,
            evaluation_end,
            "missing_daily_input",
        )
    canonical_order = np.argsort(dates_array, kind="stable")
    return _monthly_oos_suv_sorted(
        dates_array[canonical_order],
        returns_array[canonical_order],
        volumes_array[canonical_order],
        evaluation_month,
        estimation_window=estimation_window,
        min_estimation_obs=min_estimation_obs,
        min_evaluation_obs=min_evaluation_obs,
        gap_trading_days=gap_trading_days,
        sigma_floor=sigma_floor,
    )


def monthly_oos_suv_frame(
    daily: pd.DataFrame,
    evaluation_months: Iterable[str | np.datetime64 | pd.Timestamp],
    *,
    estimation_window: int = 60,
    min_estimation_obs: int = 40,
    min_evaluation_obs: int = 10,
    gap_trading_days: int = 0,
    sigma_floor: float = 1e-12,
) -> pd.DataFrame:
    """Deterministic batch adapter used by tests and the resumable SUV CLI."""
    required = {"permno", "date", "dlyret", "dlyvol"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise KeyError(f"Daily SUV input is missing columns: {missing}")
    frame = daily[["permno", "date", "dlyret", "dlyvol"]].copy()
    frame["permno"] = pd.to_numeric(frame["permno"], errors="raise").astype(
        np.int64
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["dlyret"] = pd.to_numeric(frame["dlyret"], errors="coerce").astype(
        np.float64
    )
    frame["dlyvol"] = pd.to_numeric(frame["dlyvol"], errors="coerce").astype(
        np.float64
    )
    frame = frame.sort_values(["permno", "date"], kind="stable")
    months = np.array(
        sorted({np.datetime64(month, "M") for month in evaluation_months}),
        dtype="datetime64[M]",
    )
    rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for permno, group in frame.groupby("permno", sort=True, observed=True):
            dates = group["date"].to_numpy(dtype="datetime64[D]")
            returns = group["dlyret"].to_numpy(dtype=np.float64)
            volumes = group["dlyvol"].to_numpy(dtype=np.float64)
            for month in months:
                result = _monthly_oos_suv_sorted(
                    dates,
                    returns,
                    volumes,
                    month,
                    estimation_window=estimation_window,
                    min_estimation_obs=min_estimation_obs,
                    min_evaluation_obs=min_evaluation_obs,
                    gap_trading_days=gap_trading_days,
                    sigma_floor=sigma_floor,
                )
                rows.append(
                    {
                        "permno": int(permno),
                        "month": month.astype("datetime64[D]"),
                        **{
                            column: result[column]
                            for column in SUV_OOS_OUTPUT_COLUMNS
                            if column not in {"permno", "month"}
                        },
                    }
                )
    return pd.DataFrame(rows, columns=SUV_OOS_OUTPUT_COLUMNS)


def _adjacent_month(left: np.datetime64, right: np.datetime64) -> bool:
    return (
        left.astype("datetime64[M]") + np.timedelta64(1, "M")
        == right.astype("datetime64[M]")
    )


def _valid_ff3(group: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.isfinite(group["simple_excess"])
        & np.isfinite(group["mktrf"])
        & np.isfinite(group["smb"])
        & np.isfinite(group["hml"])
    )


def _ff3_arrays(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    groups = [current] if previous is None else [previous, current]
    y = np.concatenate([group["simple_excess"] for group in groups])
    predictors = np.column_stack(
        [
            np.concatenate([group["mktrf"] for group in groups]),
            np.concatenate([group["smb"] for group in groups]),
            np.concatenate([group["hml"] for group in groups]),
        ]
    )
    valid = np.isfinite(y) & np.all(np.isfinite(predictors), axis=1)
    return y[valid], predictors[valid]


def deterministic_daily_group_features(
    current: dict[str, np.ndarray],
    *,
    previous: dict[str, np.ndarray] | None,
    idio_min_obs: int,
    resid_var_min_obs: int,
    variance_min_obs: int,
    suv_min_obs: int,
) -> dict[str, Any]:
    y_current, x_current = _ff3_arrays(current)
    n_ff3 = len(y_current)
    idio_vol = np.nan
    idio_rank = 0
    idio_zero = False
    if n_ff3 >= idio_min_obs:
        fit = deterministic_ols(y_current, x_current, ddof=1)
        idio_rank = fit.rank
        idio_zero = fit.residual_sum_of_squares == 0.0
        idio_vol = fit.residual_standard_deviation

    use_previous = previous is not None and _adjacent_month(
        previous["month"][0],
        current["month"][0],
    )
    prior = previous if use_previous else None
    y_two, x_two = _ff3_arrays(current, prior)
    n_ff3_2 = len(y_two)
    resid_var = np.nan
    resid_var_rank = 0
    resid_var_zero = False
    if n_ff3_2 >= resid_var_min_obs:
        fit_two = deterministic_ols(y_two, x_two, ddof=1)
        resid_var_rank = fit_two.rank
        resid_var_zero = fit_two.residual_sum_of_squares == 0.0
        resid_var = fit_two.residual_variance

    spread_values = current["relative_spread"]
    spread_values = spread_values[np.isfinite(spread_values)]
    n_spread = len(spread_values)
    spread = stable_fsum(spread_values) / n_spread if n_spread else np.nan

    simple = current["simple_excess"]
    volume = current["dlyvol"]
    valid_suv = np.isfinite(simple) & np.isfinite(volume)
    suv_y = volume[valid_suv]
    suv_x = np.column_stack(
        [np.maximum(simple[valid_suv], 0.0), np.maximum(-simple[valid_suv], 0.0)]
    )
    n_suv = len(suv_y)
    suv = np.nan
    suv_rank = 0
    suv_zero = False
    if n_suv >= suv_min_obs:
        suv_fit = deterministic_ols(suv_y, suv_x, ddof=1)
        suv_rank = suv_fit.rank
        suv_zero = suv_fit.residual_sum_of_squares == 0.0
        if (
            suv_fit.residual_sum_of_squares
            > suv_fit.residual_zero_threshold
            and suv_fit.residual_standard_deviation > 0
        ):
            # An OLS model with an intercept has zero mean residual exactly.
            # Return that mathematical value rather than ranking floating tails.
            suv = 0.0

    return_values = [current["dlyret"]]
    if use_previous:
        return_values.insert(0, previous["dlyret"])
    returns = np.concatenate(return_values)
    returns = returns[np.isfinite(returns)]
    n_ret_2 = len(returns)
    variance = (
        stable_sample_variance(returns, ddof=1)
        if n_ret_2 >= variance_min_obs
        else np.nan
    )
    return {
        "permno": int(current["permno"][0]),
        "month": current["month"][0],
        "IdioVol": float(idio_vol),
        "Resid_Var": float(resid_var),
        "Spread": float(spread),
        "SUV": float(suv),
        "Variance": float(variance),
        "n_ff3": int(n_ff3),
        "n_ff3_2": int(n_ff3_2),
        "n_suv": int(n_suv),
        "n_ret_2": int(n_ret_2),
        "n_spread": int(n_spread),
        "idio_rank": int(idio_rank),
        "resid_var_rank": int(resid_var_rank),
        "suv_rank": int(suv_rank),
        "idio_zero_residual": bool(idio_zero),
        "resid_var_zero_residual": bool(resid_var_zero),
        "suv_zero_residual": bool(suv_zero),
    }


def _batched_ols(
    tasks: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    batch_size: int = 20_000,
) -> dict[int, dict[str, Any]]:
    """Deterministic stacked reduced-QR with explicit gelsy fallback."""
    by_length: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for task in tasks:
        by_length.setdefault(len(task[1]), []).append(task)
    results: dict[int, dict[str, Any]] = {}
    for n_obs in sorted(by_length):
        bucket = by_length[n_obs]
        for start in range(0, len(bucket), batch_size):
            subset = bucket[start : start + batch_size]
            y = np.stack([task[1] for task in subset]).astype(np.float64, copy=False)
            predictors = np.stack([task[2] for task in subset]).astype(
                np.float64,
                copy=False,
            )
            design = np.concatenate(
                [
                    np.ones((len(subset), n_obs, 1), dtype=np.float64),
                    predictors,
                ],
                axis=2,
            )
            q, r = np.linalg.qr(design, mode="reduced")
            diagonal = np.abs(np.diagonal(r, axis1=1, axis2=2))
            tolerance = (
                FLOAT64_EPSILON
                * max(n_obs, design.shape[2])
                * np.max(diagonal, axis=1)
            )
            full_rank = np.all(diagonal > tolerance[:, None], axis=1)
            full_positions = np.flatnonzero(full_rank)
            if len(full_positions):
                q_ty = np.einsum(
                    "bnp,bn->bp",
                    q[full_positions],
                    y[full_positions],
                    optimize=False,
                )
                coefficients = np.linalg.solve(
                    r[full_positions],
                    q_ty[..., None],
                )[..., 0]
                residuals = (
                    y[full_positions]
                    - np.einsum(
                        "bnp,bp->bn",
                        design[full_positions],
                        coefficients,
                        optimize=False,
                    )
                )
                for local, position in enumerate(full_positions):
                    response = y[position]
                    residual = residuals[local]
                    mean = stable_fsum(response) / n_obs
                    centered_ss = stable_fsum(
                        (response - mean) * (response - mean)
                    )
                    rss = stable_fsum(residual * residual)
                    threshold = (
                        RESIDUAL_ZERO_MULTIPLIER
                        * FLOAT64_EPSILON
                        * max(centered_ss, 1.0)
                    )
                    if rss <= threshold:
                        rss = 0.0
                    results[subset[position][0]] = {
                        "rank": int(design.shape[2]),
                        "rss": float(rss),
                        "threshold": float(threshold),
                        "variance": float(rss / (n_obs - 1)),
                        "sd": float(np.sqrt(rss / (n_obs - 1))),
                    }
            for position in np.flatnonzero(~full_rank):
                fit = deterministic_ols(
                    y[position],
                    predictors[position],
                    ddof=1,
                )
                results[subset[position][0]] = {
                    "rank": fit.rank,
                    "rss": fit.residual_sum_of_squares,
                    "threshold": fit.residual_zero_threshold,
                    "variance": fit.residual_variance,
                    "sd": fit.residual_standard_deviation,
                }
    return results


def _process_group_buffer(
    pairs: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]],
    *,
    idio_min_obs: int,
    resid_var_min_obs: int,
    variance_min_obs: int,
    suv_min_obs: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    idio_tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    resid_tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    suv_tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for row_index, (current, previous) in enumerate(pairs):
        y_current, x_current = _ff3_arrays(current)
        use_previous = previous is not None and _adjacent_month(
            previous["month"][0],
            current["month"][0],
        )
        prior = previous if use_previous else None
        y_two, x_two = _ff3_arrays(current, prior)
        spread_values = current["relative_spread"]
        spread_values = spread_values[np.isfinite(spread_values)]
        spread = (
            stable_fsum(spread_values) / len(spread_values)
            if len(spread_values)
            else np.nan
        )
        simple = current["simple_excess"]
        volume = current["dlyvol"]
        valid_suv = np.isfinite(simple) & np.isfinite(volume)
        suv_y = volume[valid_suv]
        suv_x = np.column_stack(
            [
                np.maximum(simple[valid_suv], 0.0),
                np.maximum(-simple[valid_suv], 0.0),
            ]
        )
        return_values = [current["dlyret"]]
        if use_previous:
            return_values.insert(0, previous["dlyret"])
        returns = np.concatenate(return_values)
        returns = returns[np.isfinite(returns)]
        output.append(
            {
                "permno": int(current["permno"][0]),
                "month": current["month"][0],
                "IdioVol": np.nan,
                "Resid_Var": np.nan,
                "Spread": float(spread),
                "SUV": np.nan,
                "Variance": (
                    stable_sample_variance(returns, ddof=1)
                    if len(returns) >= variance_min_obs
                    else np.nan
                ),
                "n_ff3": int(len(y_current)),
                "n_ff3_2": int(len(y_two)),
                "n_suv": int(len(suv_y)),
                "n_ret_2": int(len(returns)),
                "n_spread": int(len(spread_values)),
                "idio_rank": 0,
                "resid_var_rank": 0,
                "suv_rank": 0,
                "idio_zero_residual": False,
                "resid_var_zero_residual": False,
                "suv_zero_residual": False,
            }
        )
        if len(y_current) >= idio_min_obs:
            idio_tasks.append((row_index, y_current, x_current))
        if len(y_two) >= resid_var_min_obs:
            resid_tasks.append((row_index, y_two, x_two))
        if len(suv_y) >= suv_min_obs:
            suv_tasks.append((row_index, suv_y, suv_x))
    for row_index, result in _batched_ols(idio_tasks).items():
        output[row_index]["IdioVol"] = result["sd"]
        output[row_index]["idio_rank"] = result["rank"]
        output[row_index]["idio_zero_residual"] = result["rss"] == 0.0
    for row_index, result in _batched_ols(resid_tasks).items():
        output[row_index]["Resid_Var"] = result["variance"]
        output[row_index]["resid_var_rank"] = result["rank"]
        output[row_index]["resid_var_zero_residual"] = result["rss"] == 0.0
    for row_index, result in _batched_ols(suv_tasks).items():
        output[row_index]["suv_rank"] = result["rank"]
        output[row_index]["suv_zero_residual"] = result["rss"] == 0.0
        if result["rss"] > result["threshold"] and result["sd"] > 0:
            output[row_index]["SUV"] = 0.0
    return output


def _as_group(arrays: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values[start:stop])
        for name, values in arrays.items()
    }


def write_deterministic_daily_month_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_relation: str,
    output_path: Path,
    idio_min_obs: int,
    resid_var_min_obs: int,
    variance_min_obs: int,
    suv_min_obs: int,
    rows_per_batch: int = 500_000,
    output_chunk_groups: int = 100_000,
) -> dict[str, int]:
    """Stream stable permno/date groups and atomically write deterministic features."""
    duplicate_count = int(
        connection.execute(
            f"""SELECT coalesce(sum(n-1),0) FROM (
            SELECT count(*) n FROM {source_relation}
            GROUP BY permno,"date" HAVING count(*)>1)"""
        ).fetchone()[0]
    )
    if duplicate_count:
        raise AssertionError(f"permno+date duplicates: {duplicate_count}")
    query = f"""SELECT
      CAST(permno AS BIGINT) permno,
      CAST("date" AS DATE) "date",
      date_trunc('month',"date")::DATE "month",
      CAST(simple_excess AS DOUBLE) simple_excess,
      CAST(mktrf AS DOUBLE) mktrf,
      CAST(smb AS DOUBLE) smb,
      CAST(hml AS DOUBLE) hml,
      CAST(dlyvol AS DOUBLE) dlyvol,
      CAST(dlyret AS DOUBLE) dlyret,
      CASE WHEN dlyask>0 AND dlybid>0 AND dlyask>=dlybid
        THEN CAST(2*(dlyask-dlybid)/(dlyask+dlybid) AS DOUBLE)
      END relative_spread
    FROM {source_relation}
    ORDER BY permno,"date" """
    reader = connection.execute(query).fetch_record_batch(
        rows_per_batch=rows_per_batch
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    group_buffer: list[
        tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]
    ] = []
    pending = pd.DataFrame()
    previous: dict[str, np.ndarray] | None = None
    previous_permno: int | None = None
    group_count = 0

    def flush_rows() -> None:
        nonlocal writer
        if not rows:
            return
        table = pa.Table.from_pandas(
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS),
            preserve_index=False,
        )
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        rows.clear()

    def consume(frame: pd.DataFrame) -> None:
        nonlocal previous, previous_permno, group_count
        arrays = {
            name: frame[name].to_numpy()
            for name in [
                "permno", "date", "month", "simple_excess", "mktrf", "smb",
                "hml", "dlyvol", "dlyret", "relative_spread",
            ]
        }
        permno = arrays["permno"]
        month = arrays["month"].astype("datetime64[M]")
        boundaries = np.flatnonzero(
            np.r_[True, (permno[1:] != permno[:-1]) | (month[1:] != month[:-1]), True]
        )
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            current = _as_group(arrays, int(left), int(right))
            current["month"] = current["month"].astype("datetime64[M]")
            current_permno = int(current["permno"][0])
            prior = previous if previous_permno == current_permno else None
            group_buffer.append((current, prior))
            previous = current
            previous_permno = current_permno
            group_count += 1
            if len(group_buffer) >= output_chunk_groups:
                rows.extend(
                    _process_group_buffer(
                        group_buffer,
                        idio_min_obs=idio_min_obs,
                        resid_var_min_obs=resid_var_min_obs,
                        variance_min_obs=variance_min_obs,
                        suv_min_obs=suv_min_obs,
                    )
                )
                group_buffer.clear()
                flush_rows()

    with threadpool_limits(limits=1):
        for batch in reader:
            frame = batch.to_pandas()
            if not pending.empty:
                frame = pd.concat([pending, frame], ignore_index=True)
            key_change = (
                frame["permno"].ne(frame["permno"].shift(-1))
                | frame["month"].ne(frame["month"].shift(-1))
            )
            last_group_start = int(
                np.flatnonzero(key_change.to_numpy())[-2] + 1
            ) if key_change.sum() >= 2 else 0
            complete = frame.iloc[:last_group_start]
            pending = frame.iloc[last_group_start:].copy()
            if not complete.empty:
                consume(complete)
        if not pending.empty:
            consume(pending)
        if group_buffer:
            rows.extend(
                _process_group_buffer(
                    group_buffer,
                    idio_min_obs=idio_min_obs,
                    resid_var_min_obs=resid_var_min_obs,
                    variance_min_obs=variance_min_obs,
                    suv_min_obs=suv_min_obs,
                )
            )
            group_buffer.clear()
        flush_rows()
    if writer is not None:
        writer.close()
    if not temporary.is_file():
        raise RuntimeError("deterministic daily feature writer produced no output")
    os.replace(temporary, output_path)
    return {
        "daily_key_duplicates": duplicate_count,
        "daily_month_groups": int(group_count),
    }


def write_monthly_oos_suv_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_relation: str,
    output_path: Path,
    start_month: str | np.datetime64 | None = None,
    end_month: str | np.datetime64 | None = None,
    estimation_window: int = 60,
    min_estimation_obs: int = 40,
    min_evaluation_obs: int = 10,
    gap_trading_days: int = 0,
    sigma_floor: float = 1e-12,
    rows_per_batch: int = 500_000,
    output_chunk_groups: int = 100_000,
) -> dict[str, Any]:
    """Stream all securities and atomically write monthly_oos_suv_v1."""
    duplicate_count = int(
        connection.execute(
            f"""SELECT coalesce(sum(n-1),0) FROM (
            SELECT count(*) n FROM {source_relation}
            GROUP BY permno,"date" HAVING count(*)>1)"""
        ).fetchone()[0]
    )
    if duplicate_count:
        raise AssertionError(f"permno+date duplicates: {duplicate_count}")
    query = f"""SELECT
      CAST(permno AS BIGINT) permno,
      CAST("date" AS DATE) "date",
      CAST(dlyret AS DOUBLE) dlyret,
      CAST(dlyvol AS DOUBLE) dlyvol
    FROM {source_relation}
    ORDER BY permno,"date" """
    reader = connection.execute(query).fetch_record_batch(
        rows_per_batch=rows_per_batch
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    output_rows: list[dict[str, Any]] = []
    pending = pd.DataFrame()
    group_count = 0
    valid_count = 0
    missing_counts = {reason: 0 for reason in SUV_MISSING_REASONS}
    start_value = (
        np.datetime64(start_month, "M") if start_month is not None else None
    )
    end_value = np.datetime64(end_month, "M") if end_month is not None else None

    def flush_rows() -> None:
        nonlocal writer
        if not output_rows:
            return
        table = pa.Table.from_pandas(
            pd.DataFrame(output_rows, columns=SUV_OOS_OUTPUT_COLUMNS),
            preserve_index=False,
        )
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        output_rows.clear()

    def consume(frame: pd.DataFrame) -> None:
        nonlocal group_count, valid_count
        for permno, group in frame.groupby("permno", sort=True, observed=True):
            dates = group["date"].to_numpy(dtype="datetime64[D]")
            returns = group["dlyret"].to_numpy(dtype=np.float64)
            volumes = group["dlyvol"].to_numpy(dtype=np.float64)
            observed_months = np.unique(dates.astype("datetime64[M]"))
            if start_value is not None:
                observed_months = observed_months[observed_months >= start_value]
            if end_value is not None:
                observed_months = observed_months[observed_months <= end_value]
            for month in observed_months:
                result = _monthly_oos_suv_sorted(
                    dates,
                    returns,
                    volumes,
                    month,
                    estimation_window=estimation_window,
                    min_estimation_obs=min_estimation_obs,
                    min_evaluation_obs=min_evaluation_obs,
                    gap_trading_days=gap_trading_days,
                    sigma_floor=sigma_floor,
                )
                reason = result["missing_reason"]
                if reason is None:
                    valid_count += 1
                else:
                    missing_counts[reason] += 1
                output_rows.append(
                    {
                        "permno": int(permno),
                        "month": month.astype("datetime64[D]"),
                        **{
                            column: result[column]
                            for column in SUV_OOS_OUTPUT_COLUMNS
                            if column not in {"permno", "month"}
                        },
                    }
                )
                group_count += 1
                if len(output_rows) >= output_chunk_groups:
                    flush_rows()

    with threadpool_limits(limits=1):
        for batch in reader:
            frame = batch.to_pandas()
            if not pending.empty:
                frame = pd.concat([pending, frame], ignore_index=True)
            last_permno = int(frame["permno"].iloc[-1])
            complete = frame.loc[frame["permno"] != last_permno]
            pending = frame.loc[frame["permno"] == last_permno].copy()
            if not complete.empty:
                consume(complete)
        if not pending.empty:
            consume(pending)
        flush_rows()
    if writer is not None:
        writer.close()
    if not temporary.is_file():
        raise RuntimeError("monthly_oos_suv_v1 writer produced no output")
    os.replace(temporary, output_path)
    return {
        "daily_key_duplicates": duplicate_count,
        "daily_month_groups": int(group_count),
        "valid_suv_count": int(valid_count),
        "missing_reason_counts": missing_counts,
        "method": SUV_METHOD_MONTHLY_OOS,
    }
