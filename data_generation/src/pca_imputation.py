from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class PCAImputationResult:
    values: np.ndarray
    imputed_mask: np.ndarray
    eigenvalues: np.ndarray
    status: str
    effective_rank: np.ndarray
    coefficient_norm: np.ndarray
    condition_after_truncation: np.ndarray


def pairwise_covariance(values: np.ndarray) -> np.ndarray:
    """Pairwise sample covariance in canonical float64 column order."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    feature_count = x.shape[1]
    covariance = np.full((feature_count, feature_count), np.nan, dtype=np.float64)
    for left in range(feature_count):
        for right in range(left, feature_count):
            valid = np.isfinite(x[:, left]) & np.isfinite(x[:, right])
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue
            left_values = x[valid, left]
            right_values = x[valid, right]
            left_centered = left_values - np.mean(left_values, dtype=np.float64)
            right_centered = right_values - np.mean(right_values, dtype=np.float64)
            value = float(np.dot(left_centered, right_centered) / (n_valid - 1))
            covariance[left, right] = covariance[right, left] = value
    return covariance


def _canonical_loadings(
    values: np.ndarray,
    *,
    factors: int,
) -> tuple[np.ndarray | None, np.ndarray, str]:
    covariance = pairwise_covariance(values)
    if not np.all(np.isfinite(covariance)):
        return None, np.array([], dtype=np.float64), "incomplete_covariance"
    covariance = np.asarray((covariance + covariance.T) / 2.0, dtype=np.float64)
    try:
        with threadpool_limits(limits=1):
            eigenvalues, eigenvectors = scipy.linalg.eigh(
                covariance,
                driver="evd",
                check_finite=True,
            )
    except (np.linalg.LinAlgError, ValueError):
        return None, np.array([], dtype=np.float64), "eigendecomposition_failed"
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    loadings = np.asarray(
        eigenvectors[:, order[: min(int(factors), values.shape[1])]],
        dtype=np.float64,
    )
    for component in range(loadings.shape[1]):
        column = loadings[:, component]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            loadings[:, component] *= -1.0
    return loadings, eigenvalues, "ok"


def _solve_pattern_direct_tsvd(
    values: np.ndarray,
    rows: np.ndarray,
    observed: np.ndarray,
    loadings: np.ndarray,
    *,
    relative_singular_value_threshold: float,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    local_loadings = np.asarray(loadings[observed], dtype=np.float64)
    with threadpool_limits(limits=1):
        left, singular_values, right_t = np.linalg.svd(
            local_loadings,
            full_matrices=False,
        )
    if singular_values.size == 0 or singular_values[0] <= 0:
        return (
            np.full((len(rows), int((~observed).sum())), np.nan, dtype=np.float64),
            0,
            np.full(len(rows), np.nan, dtype=np.float64),
            np.inf,
        )
    retained = singular_values > (
        float(relative_singular_value_threshold) * singular_values[0]
    )
    rank = int(retained.sum())
    inverse = np.divide(
        1.0,
        singular_values,
        out=np.zeros_like(singular_values),
        where=retained,
    )
    observed_columns = np.flatnonzero(observed)
    observed_values = values[np.ix_(rows, observed_columns)]
    coefficients = ((observed_values @ left) * inverse) @ right_t
    missing_values = coefficients @ loadings[~observed].T
    coefficient_norms = np.linalg.norm(coefficients, axis=1)
    condition = (
        float(singular_values[0] / singular_values[retained][-1])
        if rank
        else np.inf
    )
    return missing_values, rank, coefficient_norms, condition


def pca_impute_ranked_features(
    values: np.ndarray,
    *,
    factors: int = 10,
    minimum_observed: int = 10,
    relative_singular_value_threshold: float = 0.05,
) -> PCAImputationResult:
    """Missing-only monthly PCA imputation with direct deterministic float64 TSVD."""
    if not 0 < relative_singular_value_threshold < 1:
        raise ValueError("relative_singular_value_threshold must lie in (0, 1)")
    original = np.asarray(values, dtype=np.float64)
    if original.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    original = original.copy()
    loadings, eigenvalues, status = _canonical_loadings(
        original,
        factors=int(factors),
    )
    row_count = original.shape[0]
    empty_rank = np.full(row_count, -1, dtype=np.int16)
    empty_norm = np.full(row_count, np.nan, dtype=np.float64)
    empty_condition = np.full(row_count, np.nan, dtype=np.float64)
    if loadings is None:
        return PCAImputationResult(
            original,
            np.zeros_like(original, dtype=bool),
            eigenvalues,
            status,
            empty_rank,
            empty_norm,
            empty_condition,
        )

    result = original.copy()
    imputed = np.zeros_like(original, dtype=bool)
    effective_rank = empty_rank.copy()
    coefficient_norm = empty_norm.copy()
    condition = empty_condition.copy()
    observed_mask = np.isfinite(original)
    eligible = (
        (observed_mask.sum(axis=1) >= int(minimum_observed))
        & (~observed_mask.all(axis=1))
    )
    if eligible.any():
        eligible_rows = np.flatnonzero(eligible)
        packed = np.packbits(observed_mask[eligible_rows], axis=1)
        unique_patterns, inverse = np.unique(packed, axis=0, return_inverse=True)
        for pattern_index, packed_pattern in enumerate(unique_patterns):
            rows = eligible_rows[inverse == pattern_index]
            observed = np.unpackbits(
                packed_pattern,
                count=original.shape[1],
            ).astype(bool)
            missing_columns = np.flatnonzero(~observed)
            filled, rank, norms, pattern_condition = _solve_pattern_direct_tsvd(
                original,
                rows,
                observed,
                loadings,
                relative_singular_value_threshold=relative_singular_value_threshold,
            )
            result[np.ix_(rows, missing_columns)] = filled
            imputed[np.ix_(rows, missing_columns)] = np.isfinite(filled)
            effective_rank[rows] = rank
            coefficient_norm[rows] = norms
            condition[rows] = pattern_condition

    originally_observed = np.isfinite(original)
    if not np.array_equal(result[originally_observed], original[originally_observed]):
        raise AssertionError("PCA imputation modified an originally observed value")
    if not np.array_equal(np.isnan(result[~originally_observed]), np.isnan(original[~originally_observed]) & ~imputed[~originally_observed]):
        raise AssertionError("PCA imputation mask accounting is inconsistent")
    return PCAImputationResult(
        result,
        imputed,
        eigenvalues,
        status,
        effective_rank,
        coefficient_norm,
        condition,
    )
