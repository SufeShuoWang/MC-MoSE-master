from __future__ import annotations

import gc
import hashlib
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .atomic_write import write_text_atomic


REQUIRED_INPUT_KEYS = (
    "X",
    "y",
    "X_macro",
    "dates",
    "wficn",
    "feature_names",
)
OUTPUT_KEYS = ("data", "wficn", "names", "date")
OUTPUT_SENTINEL = np.float64(-99.99)
FEATURE_ALIASES = {"ages": "age"}
MACRO_OUTPUT_NAMES = ("sentiment", "CFNAI")


class LayoutAdapterError(RuntimeError):
    """Raised when a layout-only conversion cannot be completed safely."""


@dataclass(frozen=True)
class NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_keys(archive: Mapping[str, Any] | np.lib.npyio.NpzFile) -> list[str]:
    if hasattr(archive, "files"):
        return list(archive.files)
    return list(archive.keys())


def _read_array_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version in {(2, 0), (3, 0)}:
        return np.lib.format.read_array_header_2_0(stream)
    raise LayoutAdapterError(f"Unsupported NPY header version: {version!r}")


def inspect_npz_headers(path: str | Path) -> tuple[list[str], dict[str, NpyHeader]]:
    keys: list[str] = []
    headers: dict[str, NpyHeader] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.filename.endswith(".npy"):
                    continue
                key = Path(member.filename).stem
                with archive.open(member) as stream:
                    shape, fortran_order, dtype = _read_array_header(stream)
                keys.append(key)
                headers[key] = NpyHeader(
                    shape=tuple(int(value) for value in shape),
                    dtype=np.dtype(dtype),
                    fortran_order=bool(fortran_order),
                )
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise LayoutAdapterError(f"Could not inspect NPZ headers for {path}: {exc}") from exc
    return keys, headers


def _parse_month_string(value: str) -> tuple[int, int]:
    text = value.strip()
    compact = re.fullmatch(r"(\d{4})(\d{2})", text)
    separated = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?", text)
    match = compact or separated
    if match is None:
        try:
            month = np.datetime64(text, "M")
        except (TypeError, ValueError) as exc:
            raise LayoutAdapterError(f"Invalid month value: {value!r}") from exc
        if np.isnat(month):
            raise LayoutAdapterError(f"Invalid month value: {value!r}")
        canonical = np.datetime_as_string(month, unit="M")
        match = re.fullmatch(r"(\d{4})-(\d{2})", canonical)
    if match is None:
        raise LayoutAdapterError(f"Invalid month value: {value!r}")
    year, month_number = int(match.group(1)), int(match.group(2))
    if not 1000 <= year <= 9999 or not 1 <= month_number <= 12:
        raise LayoutAdapterError(f"Invalid month value: {value!r}")
    return year, month_number


def convert_dates_to_yyyymm(dates: np.ndarray | Sequence[Any]) -> np.ndarray:
    """Explicitly convert supported monthly representations to int64 YYYYMM."""
    values = np.asarray(dates)
    if values.ndim != 1:
        raise LayoutAdapterError(f"dates must be one-dimensional; got {values.shape}")
    if values.size == 0:
        raise LayoutAdapterError("dates must not be empty")

    parsed: list[tuple[int, int]] = []
    if values.dtype.kind == "M":
        months = values.astype("datetime64[M]")
        if np.isnat(months).any():
            raise LayoutAdapterError("dates contains NaT")
        for text in np.datetime_as_string(months, unit="M").tolist():
            parsed.append(_parse_month_string(str(text)))
    elif values.dtype.kind in {"i", "u"}:
        for raw in values.tolist():
            integer = int(raw)
            parsed.append(_parse_month_string(f"{integer:06d}"))
    else:
        for raw in values.tolist():
            parsed.append(_parse_month_string(str(raw)))

    result = np.asarray(
        [year * 100 + month for year, month in parsed],
        dtype=np.int64,
    )
    ordinal = np.asarray(
        [year * 12 + month - 1 for year, month in parsed],
        dtype=np.int64,
    )
    if len(ordinal) > 1 and not np.all(np.diff(ordinal) == 1):
        raise LayoutAdapterError("dates must be strictly increasing consecutive months")
    return result


def _missing_mask(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind not in {"f"}:
        raise LayoutAdapterError(f"Expected floating-point data; got {array.dtype}")
    typed_sentinel = np.asarray(-99.99, dtype=array.dtype).item()
    return np.asarray(array == typed_sentinel, dtype=bool)


def _require_finite_or_sentinel(array: np.ndarray, *, name: str) -> None:
    if array.dtype.kind != "f":
        raise LayoutAdapterError(f"{name} must be floating point; got {array.dtype}")
    if array.ndim == 3:
        for index in range(array.shape[2]):
            if not np.isfinite(array[:, :, index]).all():
                raise LayoutAdapterError(
                    f"{name} contains NaN or infinity in channel {index}"
                )
    elif not np.isfinite(array).all():
        raise LayoutAdapterError(f"{name} contains NaN or infinity")


def _unique_strings(values: np.ndarray, *, name: str) -> list[str]:
    if values.ndim != 1:
        raise LayoutAdapterError(f"{name} must be one-dimensional; got {values.shape}")
    strings = values.astype(str).tolist()
    if any(not value for value in strings):
        raise LayoutAdapterError(f"{name} contains an empty value")
    if len(set(strings)) != len(strings):
        raise LayoutAdapterError(f"{name} contains duplicate values")
    return strings


def inspect_input(
    archive: Mapping[str, Any] | np.lib.npyio.NpzFile,
    reference_names: np.ndarray | Sequence[str],
    expected_dimensions: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Validate the source archive without cleaning or changing its contents."""
    keys = _archive_keys(archive)
    missing_keys = [key for key in REQUIRED_INPUT_KEYS if key not in keys]
    if missing_keys:
        raise LayoutAdapterError(
            "Input NPZ is missing required keys: " + ", ".join(missing_keys)
        )

    x = np.asarray(archive["X"])
    y = np.asarray(archive["y"])
    x_macro = np.asarray(archive["X_macro"])
    dates = np.asarray(archive["dates"])
    wficn = np.asarray(archive["wficn"])
    feature_names = np.asarray(archive["feature_names"])

    if x.ndim != 3:
        raise LayoutAdapterError(f"X must be 3D; got {x.shape}")
    if y.ndim != 2:
        raise LayoutAdapterError(f"y must be 2D; got {y.shape}")
    if x_macro.ndim != 2:
        raise LayoutAdapterError(f"X_macro must be 2D; got {x_macro.shape}")
    if dates.ndim != 1:
        raise LayoutAdapterError(f"dates must be 1D; got {dates.shape}")
    if wficn.ndim != 1:
        raise LayoutAdapterError(f"wficn must be 1D; got {wficn.shape}")
    if feature_names.ndim != 1:
        raise LayoutAdapterError(
            f"feature_names must be 1D; got {feature_names.shape}"
        )

    time_count, fund_count, feature_count = x.shape
    if not (
        time_count == y.shape[0] == x_macro.shape[0] == len(dates)
    ):
        raise LayoutAdapterError(
            "Time axis mismatch across X, y, X_macro, and dates"
        )
    if not (fund_count == y.shape[1] == len(wficn)):
        raise LayoutAdapterError("Fund axis mismatch across X, y, and wficn")
    if feature_count != len(feature_names):
        raise LayoutAdapterError(
            "Feature axis mismatch between X and feature_names"
        )
    if x_macro.shape[1] != 2:
        raise LayoutAdapterError(
            f"X_macro must have exactly two channels; got {x_macro.shape[1]}"
        )
    if expected_dimensions is not None and (
        time_count,
        fund_count,
        feature_count,
    ) != expected_dimensions:
        raise LayoutAdapterError(
            "Formal input dimensions do not match "
            f"{expected_dimensions}: got {(time_count, fund_count, feature_count)}"
        )
    if expected_dimensions is not None and str(wficn.dtype) != "<U6":
        raise LayoutAdapterError(
            f"Formal wficn dtype must remain <U6; got {wficn.dtype}"
        )

    source_feature_names = _unique_strings(
        feature_names, name="feature_names"
    )
    source_wficn = _unique_strings(wficn, name="wficn")
    ref_names = _unique_strings(
        np.asarray(reference_names), name="reference names"
    )
    if len(ref_names) < 2 or ref_names[-1].lower() != "sentiment":
        raise LayoutAdapterError(
            "Reference names must end with the sentiment channel"
        )
    reference_core_names = ref_names[:-1]
    if len(reference_core_names) != feature_count:
        raise LayoutAdapterError(
            "Reference core feature count does not match input X: "
            f"{len(reference_core_names)} != {feature_count}"
        )

    source_positions = {name: index for index, name in enumerate(source_feature_names)}
    source_names_in_reference_order = [
        FEATURE_ALIASES.get(name, name) for name in reference_core_names
    ]
    missing_features = [
        name
        for name in source_names_in_reference_order
        if name not in source_positions
    ]
    unexpected_features = sorted(
        set(source_feature_names) - set(source_names_in_reference_order)
    )
    if missing_features or unexpected_features:
        raise LayoutAdapterError(
            "Feature mapping is not one-to-one; "
            f"missing={missing_features}, unexpected={unexpected_features}"
        )
    feature_source_indices = [
        source_positions[name] for name in source_names_in_reference_order
    ]
    if len(set(feature_source_indices)) != feature_count:
        raise LayoutAdapterError("Feature mapping contains duplicate source channels")

    macro_indices = {"Sentiment": 0, "CFNAI": 1}
    macro_name_source = "documented X_macro order"
    if "macro_feature_names" in keys:
        macro_names = _unique_strings(
            np.asarray(archive["macro_feature_names"]),
            name="macro_feature_names",
        )
        if len(macro_names) != x_macro.shape[1]:
            raise LayoutAdapterError(
                "X_macro axis mismatch with macro_feature_names"
            )
        macro_lookup = {name.lower(): index for index, name in enumerate(macro_names)}
        if "sentiment" not in macro_lookup or "cfnai" not in macro_lookup:
            raise LayoutAdapterError(
                "macro_feature_names must contain Sentiment and CFNAI"
            )
        macro_indices = {
            "Sentiment": macro_lookup["sentiment"],
            "CFNAI": macro_lookup["cfnai"],
        }
        macro_name_source = "macro_feature_names"

    _require_finite_or_sentinel(x, name="X")
    _require_finite_or_sentinel(y, name="y")
    _require_finite_or_sentinel(x_macro, name="X_macro")
    converted_dates = convert_dates_to_yyyymm(dates)

    return {
        "keys": keys,
        "time_count": time_count,
        "fund_count": fund_count,
        "feature_count": feature_count,
        "wficn_dtype": str(wficn.dtype),
        "feature_source_indices": feature_source_indices,
        "source_feature_names": source_feature_names,
        "reference_core_names": reference_core_names,
        "source_names_in_reference_order": source_names_in_reference_order,
        "macro_indices": macro_indices,
        "macro_name_source": macro_name_source,
        "dates_yyyymm": converted_dates,
        "first_date": int(converted_dates[0]),
        "last_date": int(converted_dates[-1]),
        "wficn_count": len(source_wficn),
    }


def _load_reference(reference_path: Path) -> dict[str, Any]:
    key_order, headers = inspect_npz_headers(reference_path)
    required = ("data", "permno", "names", "date")
    missing = [key for key in required if key not in headers]
    if missing:
        raise LayoutAdapterError(
            "Reference NPZ is missing required keys: " + ", ".join(missing)
        )
    with np.load(reference_path, allow_pickle=False) as reference:
        names = np.asarray(reference["names"])
        dates = np.asarray(reference["date"])
    if names.ndim != 1 or dates.ndim != 1:
        raise LayoutAdapterError("Reference names and date must be one-dimensional")
    if headers["data"].shape[-1] != len(names) + 1:
        raise LayoutAdapterError(
            "Reference data channel count does not equal len(names) + 1"
        )
    if headers["data"].dtype.kind != "f":
        raise LayoutAdapterError("Reference data dtype must be floating point")
    if headers["date"].dtype.kind not in {"i", "u"}:
        raise LayoutAdapterError("Reference date dtype must be integer")
    ref_date = convert_dates_to_yyyymm(dates)
    return {
        "key_order": key_order,
        "headers": headers,
        "names": names,
        "dates": ref_date.astype(headers["date"].dtype, copy=False),
        "data_dtype": headers["data"].dtype,
        "names_dtype": headers["names"].dtype,
        "date_dtype": headers["date"].dtype,
        "data_shape": headers["data"].shape,
    }


def _temporary_sibling(target: Path, *, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(handle)
    return Path(name)


def _copy_numeric_channel(source: np.ndarray, destination: np.ndarray) -> int:
    mask = _missing_mask(source)
    destination[...] = source
    destination[mask] = OUTPUT_SENTINEL
    return int(mask.sum())


def _normalized_vector(source: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    mask = _missing_mask(source)
    result = source.astype(dtype, copy=True)
    result[mask] = np.asarray(-99.99, dtype=dtype)
    return result


def _assert_array_equal(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    checks: list[dict[str, Any]],
) -> None:
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
        checks.append({"name": name, "passed": False, "details": str(exc)[:1200]})
        raise LayoutAdapterError(f"Validation failed: {name}") from exc
    checks.append(
        {
            "name": name,
            "passed": True,
            "details": f"shape={actual.shape}, dtype={actual.dtype}",
        }
    )


def _require_check(
    name: str,
    condition: bool,
    details: str,
    checks: list[dict[str, Any]],
) -> None:
    checks.append({"name": name, "passed": bool(condition), "details": details})
    if not condition:
        raise LayoutAdapterError(f"Validation failed: {name}: {details}")


def _validate_output(
    output_path: Path,
    source: np.lib.npyio.NpzFile,
    inspection: dict[str, Any],
    reference: dict[str, Any],
    output_names: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    with np.load(output_path, allow_pickle=False) as converted:
        _require_check(
            "output_key_order",
            list(converted.files) == list(OUTPUT_KEYS),
            f"actual={converted.files}, expected={list(OUTPUT_KEYS)}",
            checks,
        )
        data = np.asarray(converted["data"])
        output_wficn = np.asarray(converted["wficn"])
        names = np.asarray(converted["names"])
        output_dates = np.asarray(converted["date"])

        expected_shape = (
            inspection["time_count"],
            inspection["fund_count"],
            len(output_names) + 1,
        )
        _require_check(
            "data_shape",
            data.shape == expected_shape,
            f"actual={data.shape}, expected={expected_shape}",
            checks,
        )
        _require_check(
            "feature_channel_relation",
            data.shape[-1] == len(names) + 1,
            f"channels={data.shape[-1]}, names={len(names)}",
            checks,
        )
        _require_check(
            "time_axis_relation",
            data.shape[0] == len(output_dates),
            f"data_time={data.shape[0]}, dates={len(output_dates)}",
            checks,
        )
        _require_check(
            "fund_axis_relation",
            data.shape[1] == len(output_wficn),
            f"data_funds={data.shape[1]}, wficn={len(output_wficn)}",
            checks,
        )
        _require_check(
            "data_dtype",
            data.dtype == reference["data_dtype"],
            f"actual={data.dtype}, expected={reference['data_dtype']}",
            checks,
        )
        _require_check(
            "names_dtype",
            names.dtype == output_names.dtype,
            f"actual={names.dtype}, expected={output_names.dtype}",
            checks,
        )
        _require_check(
            "date_dtype",
            output_dates.dtype == reference["date_dtype"],
            f"actual={output_dates.dtype}, expected={reference['date_dtype']}",
            checks,
        )

        source_wficn = np.asarray(source["wficn"])
        source_y = np.asarray(source["y"])
        source_x = np.asarray(source["X"])
        source_macro = np.asarray(source["X_macro"])
        _assert_array_equal("names_exact", names, output_names, checks)
        _assert_array_equal("wficn_exact", output_wficn, source_wficn, checks)
        _require_check(
            "wficn_dtype_preserved",
            output_wficn.dtype == source_wficn.dtype,
            f"actual={output_wficn.dtype}, expected={source_wficn.dtype}",
            checks,
        )
        _assert_array_equal(
            "date_exact",
            output_dates,
            inspection["dates_yyyymm"].astype(reference["date_dtype"]),
            checks,
        )
        reference_dates = np.asarray(reference["dates"])
        _require_check(
            "reference_date_prefix_available",
            len(output_dates) >= len(reference_dates),
            f"output={len(output_dates)}, reference={len(reference_dates)}",
            checks,
        )
        _assert_array_equal(
            "reference_date_prefix_exact",
            output_dates[: len(reference_dates)],
            reference_dates,
            checks,
        )

        source_y_mask = _missing_mask(source_y)
        output_y_mask = data[:, :, 0] == OUTPUT_SENTINEL
        _assert_array_equal(
            "y_missing_mask_exact", output_y_mask, source_y_mask, checks
        )
        _assert_array_equal(
            "y_nonmissing_values_exact",
            data[:, :, 0][~output_y_mask],
            source_y.astype(reference["data_dtype"], copy=False)[~source_y_mask],
            checks,
        )
        stats["y_missing_count"] = int(source_y_mask.sum())

        feature_rows: list[dict[str, Any]] = []
        x_missing_total = 0
        for output_index, source_index in enumerate(
            inspection["feature_source_indices"]
        ):
            output_channel = output_index + 1
            source_channel = source_x[:, :, source_index]
            output_channel_data = data[:, :, output_channel]
            source_mask = _missing_mask(source_channel)
            output_mask = output_channel_data == OUTPUT_SENTINEL
            official_name = str(output_names[output_index])
            source_name = str(np.asarray(source["feature_names"])[source_index])
            _assert_array_equal(
                f"feature_{official_name}_missing_mask_exact",
                output_mask,
                source_mask,
                checks,
            )
            _assert_array_equal(
                f"feature_{official_name}_nonmissing_values_exact",
                output_channel_data[~output_mask],
                source_channel.astype(reference["data_dtype"], copy=False)[
                    ~source_mask
                ],
                checks,
            )
            missing_count = int(source_mask.sum())
            x_missing_total += missing_count
            feature_rows.append(
                {
                    "output_channel": output_channel,
                    "output_name": official_name,
                    "source_index": source_index,
                    "source_name": source_name,
                    "missing_count": missing_count,
                }
            )
        stats["feature_mapping"] = feature_rows
        stats["X_missing_count"] = x_missing_total

        macro_rows: list[dict[str, Any]] = []
        for macro_offset, macro_name in enumerate(("Sentiment", "CFNAI")):
            output_channel = 60 + macro_offset
            source_index = inspection["macro_indices"][macro_name]
            source_values = source_macro[:, source_index]
            expected_values = _normalized_vector(
                source_values, reference["data_dtype"]
            )
            output_channel_data = data[:, :, output_channel]
            expected_broadcast = np.broadcast_to(
                expected_values[:, None], output_channel_data.shape
            )
            _assert_array_equal(
                f"{macro_name}_broadcast_values_exact",
                output_channel_data,
                expected_broadcast,
                checks,
            )
            unique_sizes = np.asarray(
                [
                    np.unique(output_channel_data[month_index]).size
                    for month_index in range(output_channel_data.shape[0])
                ],
                dtype=np.int64,
            )
            _require_check(
                f"{macro_name}_one_value_per_month",
                bool(np.all(unique_sizes == 1)),
                f"max_unique_values={int(unique_sizes.max())}",
                checks,
            )
            source_missing = _missing_mask(source_values)
            output_missing_months = output_channel_data[:, 0] == OUTPUT_SENTINEL
            _assert_array_equal(
                f"{macro_name}_missing_month_mask_exact",
                output_missing_months,
                source_missing,
                checks,
            )
            macro_rows.append(
                {
                    "name": macro_name,
                    "source_index": source_index,
                    "output_channel": output_channel,
                    "missing_months": int(source_missing.sum()),
                    "max_unique_values_per_month": int(unique_sizes.max()),
                }
            )
        stats["macro_mapping"] = macro_rows
        stats["output_shape"] = list(data.shape)
        stats["output_dtypes"] = {
            "data": str(data.dtype),
            "wficn": str(output_wficn.dtype),
            "names": str(names.dtype),
            "date": str(output_dates.dtype),
        }
    return checks, stats


def _render_check_table(checks: list[dict[str, Any]]) -> list[str]:
    lines = ["| Check | Result | Details |", "|---|---|---|"]
    for check in checks:
        details = str(check.get("details", "")).replace("|", "\\|").replace("\n", " ")
        result = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"| `{check['name']}` | {result} | {details} |")
    return lines


def _render_success_report(
    *,
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    input_sha_before: str,
    input_sha_after: str,
    reference_sha: str,
    output_sha: str,
    output_size: int,
    inspection: dict[str, Any],
    reference: dict[str, Any],
    checks: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    lines = [
        "# Dataset Validation Report",
        "",
        "**Dataset:** Kaniel-compatible extended layout",
        "",
        "**Status:** PASS",
        "",
        (
            "This file is not the official Kaniel dataset. It is an extended dataset "
            "whose channels follow the official Kaniel ordering."
        ),
        "",
        "## Files and SHA256",
        "",
        "| Role | Path | Size (bytes) | SHA256 |",
        "|---|---|---:|---|",
        f"| Input | `{input_path}` | {input_path.stat().st_size} | `{input_sha_after}` |",
        f"| Reference | `{reference_path}` | {reference_path.stat().st_size} | `{reference_sha}` |",
        f"| Output | `{output_path}` | {output_size} | `{output_sha}` |",
        "",
        f"- Input SHA256 before conversion: `{input_sha_before}`",
        f"- Input SHA256 after conversion: `{input_sha_after}`",
        f"- Input unchanged: **{input_sha_before == input_sha_after}**",
        "",
        "## Output Schema",
        "",
        "| Key | Shape | dtype | Meaning |",
        "|---|---:|---|---|",
        f"| `data` | `{tuple(stats['output_shape'])}` | `{stats['output_dtypes']['data']}` | y plus 61 feature channels |",
        f"| `wficn` | `({inspection['fund_count']},)` | `{stats['output_dtypes']['wficn']}` | Current WFICN axis, unchanged |",
        f"| `names` | `(61,)` | `{stats['output_dtypes']['names']}` | 59 core features, sentiment, CFNAI |",
        f"| `date` | `({inspection['time_count']},)` | `{stats['output_dtypes']['date']}` | Explicit YYYYMM label months |",
        "",
        "Channel relationship:",
        "",
        "```text",
        "data[:, :, 0]    = y",
        "data[:, :, 1:60] = 59 core X features",
        "data[:, :, 60]   = Sentiment",
        "data[:, :, 61]   = CFNAI",
        "data.shape[-1]   = len(names) + 1",
        "```",
        "",
        "## Input Integrity",
        "",
        f"- Input keys: `{inspection['keys']}`",
        f"- Dimensions: T={inspection['time_count']}, N={inspection['fund_count']}, F={inspection['feature_count']}",
        f"- Date range: {inspection['first_date']}–{inspection['last_date']}",
        f"- WFICN count: {inspection['wficn_count']}; dtype `{inspection['wficn_dtype']}`",
        f"- Macro channel names resolved from: `{inspection['macro_name_source']}`",
        "- Duplicate feature names: 0",
        "- Duplicate WFICN values: 0",
        "- No feature calculation, filling, interpolation, deletion, or non-missing value modification was performed.",
        "",
        "## Feature Mapping",
        "",
        "| data channel | output name | source index | source name | missing count |",
        "|---:|---|---:|---|---:|",
    ]
    for row in stats["feature_mapping"]:
        lines.append(
            f"| {row['output_channel']} | `{row['output_name']}` | "
            f"{row['source_index']} | `{row['source_name']}` | "
            f"{row['missing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Macro Broadcast",
            "",
            "| Name | source index | data channel | missing months | maximum unique values per month |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stats["macro_mapping"]:
        lines.append(
            f"| `{row['name']}` | {row['source_index']} | "
            f"{row['output_channel']} | {row['missing_months']} | "
            f"{row['max_unique_values_per_month']} |"
        )
    lines.extend(
        [
            "",
            "## Intentional Differences from the Official Dataset",
            "",
            "1. The identifier key is `wficn`, not `permno`.",
            "2. The current set and order of 6,207 WFICN values are retained.",
            "3. A CFNAI channel is appended after sentiment.",
            "4. The time range is extended through 2024.",
            "",
            "## Validation Checks",
            "",
            *_render_check_table(checks),
            "",
            "## Conclusion",
            "",
            (
                "All schema, axis, missing-mask, exact non-missing value, macro "
                "broadcast, date-prefix, and source-integrity checks passed. The "
                "output is approved as a Kaniel-compatible extended layout."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_failure_report(
    *,
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    reason: str,
    input_sha: str | None,
    reference_sha: str | None,
    output_existed_before: bool,
) -> str:
    publication_status = (
        "No new NPZ was published; the pre-existing output was preserved."
        if output_existed_before
        else "The final NPZ was not published."
    )
    return "\n".join(
        [
            "# Dataset Validation Report",
            "",
            "**Dataset:** Kaniel-compatible extended layout",
            "",
            "**Status:** FAIL",
            "",
            publication_status,
            "",
            "## Failure",
            "",
            f"`{reason}`",
            "",
            "## Files",
            "",
            f"- Input: `{input_path}`",
            f"- Reference: `{reference_path}`",
            f"- Intended output: `{output_path}`",
            f"- Input SHA256: `{input_sha or 'unavailable'}`",
            f"- Reference SHA256: `{reference_sha or 'unavailable'}`",
            f"- Report generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
            "",
        ]
    )


def _publish_output_and_report(
    *,
    temporary_npz: Path,
    target_path: Path,
    temporary_report: Path,
    report_path: Path,
) -> None:
    """Publish the validated NPZ and report with rollback of prior artifacts."""
    target_backup: Path | None = None
    report_backup: Path | None = None
    try:
        if target_path.exists():
            target_backup = _temporary_sibling(target_path, suffix=".backup")
            target_backup.unlink(missing_ok=True)
            os.replace(target_path, target_backup)
        if report_path.exists():
            report_backup = _temporary_sibling(report_path, suffix=".backup")
            report_backup.unlink(missing_ok=True)
            os.replace(report_path, report_backup)

        os.replace(temporary_npz, target_path)
        try:
            os.replace(temporary_report, report_path)
        except Exception:
            target_path.unlink(missing_ok=True)
            raise
    except Exception:
        if target_backup is not None and target_backup.exists():
            target_path.unlink(missing_ok=True)
            os.replace(target_backup, target_path)
            target_backup = None
        if report_backup is not None and report_backup.exists():
            report_path.unlink(missing_ok=True)
            os.replace(report_backup, report_path)
            report_backup = None
        raise
    finally:
        if target_backup is not None:
            target_backup.unlink(missing_ok=True)
        if report_backup is not None:
            report_backup.unlink(missing_ok=True)


def convert_layout(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
    expected_dimensions: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Convert an existing model-ready NPZ into the compatible extended layout."""
    source_path = Path(input_path).resolve()
    schema_path = Path(reference_path).resolve()
    target_path = Path(output_path).resolve()
    validation_report_path = Path(report_path).resolve()

    input_sha_before: str | None = None
    reference_sha: str | None = None
    temporary_data: Path | None = None
    temporary_npz: Path | None = None
    temporary_report: Path | None = None
    output_existed_before = target_path.exists()

    if validation_report_path in {source_path, schema_path, target_path}:
        raise LayoutAdapterError(
            "Report path must differ from input, reference, and output paths"
        )

    try:
        if not source_path.is_file():
            raise LayoutAdapterError(f"Input NPZ does not exist: {source_path}")
        if not schema_path.is_file():
            raise LayoutAdapterError(f"Reference NPZ does not exist: {schema_path}")
        if target_path.exists() and not overwrite:
            raise LayoutAdapterError(
                f"Output already exists; pass overwrite=True to replace it: {target_path}"
            )
        if source_path == target_path or schema_path == target_path:
            raise LayoutAdapterError("Output path must differ from input and reference")

        input_sha_before = file_sha256(source_path)
        reference_sha = file_sha256(schema_path)
        reference = _load_reference(schema_path)

        with np.load(source_path, allow_pickle=False) as source:
            inspection = inspect_input(
                source,
                reference["names"],
                expected_dimensions=expected_dimensions,
            )
            reference_dates = np.asarray(reference["dates"])
            converted_dates = inspection["dates_yyyymm"].astype(
                reference["date_dtype"], copy=False
            )
            if len(converted_dates) < len(reference_dates):
                raise LayoutAdapterError(
                    "Input date axis is shorter than the official reference"
                )
            try:
                np.testing.assert_array_equal(
                    converted_dates[: len(reference_dates)], reference_dates
                )
            except AssertionError as exc:
                raise LayoutAdapterError(
                    "The first reference-length input months do not match official dates"
                ) from exc

            output_names = np.concatenate(
                [
                    np.asarray(reference["names"], dtype=reference["names_dtype"]),
                    np.asarray(["CFNAI"], dtype=reference["names_dtype"]),
                ]
            )
            output_shape = (
                inspection["time_count"],
                inspection["fund_count"],
                len(output_names) + 1,
            )

            temporary_data = _temporary_sibling(target_path, suffix=".data.npy")
            temporary_data.unlink(missing_ok=True)
            data = np.lib.format.open_memmap(
                temporary_data,
                mode="w+",
                dtype=reference["data_dtype"],
                shape=output_shape,
                fortran_order=False,
            )
            source_y = np.asarray(source["y"])
            source_x = np.asarray(source["X"])
            source_macro = np.asarray(source["X_macro"])
            _copy_numeric_channel(source_y, data[:, :, 0])
            for output_index, source_index in enumerate(
                inspection["feature_source_indices"]
            ):
                _copy_numeric_channel(
                    source_x[:, :, source_index],
                    data[:, :, output_index + 1],
                )
            for macro_offset, macro_name in enumerate(("Sentiment", "CFNAI")):
                macro_index = inspection["macro_indices"][macro_name]
                macro_values = _normalized_vector(
                    source_macro[:, macro_index], reference["data_dtype"]
                )
                data[:, :, 60 + macro_offset] = macro_values[:, None]
            data.flush()
            del source_y, source_x, source_macro
            gc.collect()

            temporary_npz = _temporary_sibling(target_path, suffix=".npz.tmp")
            with temporary_npz.open("wb") as handle:
                np.savez(
                    handle,
                    data=data,
                    wficn=np.asarray(source["wficn"]),
                    names=output_names,
                    date=converted_dates,
                )
            del data
            gc.collect()
            temporary_data.unlink(missing_ok=True)
            temporary_data = None

            checks, stats = _validate_output(
                temporary_npz,
                source,
                inspection,
                reference,
                output_names,
            )

        input_sha_after = file_sha256(source_path)
        if input_sha_after != input_sha_before:
            raise LayoutAdapterError("Input SHA256 changed during conversion")
        output_sha = file_sha256(temporary_npz)
        output_size = temporary_npz.stat().st_size

        report = _render_success_report(
            input_path=source_path,
            reference_path=schema_path,
            output_path=target_path,
            input_sha_before=input_sha_before,
            input_sha_after=input_sha_after,
            reference_sha=reference_sha,
            output_sha=output_sha,
            output_size=output_size,
            inspection=inspection,
            reference=reference,
            checks=checks,
            stats=stats,
        )
        temporary_report = _temporary_sibling(
            validation_report_path, suffix=".report.tmp"
        )
        temporary_report.write_text(report, encoding="utf-8")
        result = {
            "status": "PASS",
            "input": str(source_path),
            "reference": str(schema_path),
            "output": str(target_path),
            "report": str(validation_report_path),
            "input_sha256": input_sha_after,
            "reference_sha256": reference_sha,
            "output_sha256": output_sha,
            "output_size_bytes": output_size,
            "schema": {
                "keys": list(OUTPUT_KEYS),
                "data_shape": stats["output_shape"],
                "dtypes": stats["output_dtypes"],
            },
            "checks_passed": len(checks),
        }
        _publish_output_and_report(
            temporary_npz=temporary_npz,
            target_path=target_path,
            temporary_report=temporary_report,
            report_path=validation_report_path,
        )
        temporary_npz = None
        temporary_report = None
        return result
    except Exception as exc:
        reason = str(exc)
        failure_report = _render_failure_report(
            input_path=source_path,
            reference_path=schema_path,
            output_path=target_path,
            reason=reason,
            input_sha=input_sha_before,
            reference_sha=reference_sha,
            output_existed_before=output_existed_before,
        )
        write_text_atomic(validation_report_path, failure_report)
        if isinstance(exc, LayoutAdapterError):
            raise
        raise LayoutAdapterError(reason) from exc
    finally:
        if temporary_data is not None:
            temporary_data.unlink(missing_ok=True)
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)
