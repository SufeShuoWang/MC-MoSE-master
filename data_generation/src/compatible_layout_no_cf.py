from __future__ import annotations

import gc
import hashlib
import os
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .atomic_write import write_text_atomic


LAYOUT_KEYS = ("data", "wficn", "names", "date")
CF_FEATURE_NAME = "CF"
SENTIMENT_FEATURE_NAME = "sentiment"
CFNAI_FEATURE_NAME = "CFNAI"
_ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_ZIP_LOCAL_SIGNATURE = 0x04034B50


class NoCFPostprocessError(RuntimeError):
    """Raised when the CF-only layout postprocessing cannot be completed safely."""


@dataclass(frozen=True)
class StoredNpyMember:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool
    payload_offset: int
    member_size: int


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_npz_crc(path: Path) -> int:
    """Read every ZIP member fully so raw memmaps cannot bypass CRC errors."""
    try:
        with zipfile.ZipFile(path) as archive:
            member_count = len(archive.infolist())
            bad_member = archive.testzip()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise NoCFPostprocessError(
            f"NPZ CRC verification failed for {path}: {exc}"
        ) from exc
    if bad_member is not None:
        raise NoCFPostprocessError(
            f"NPZ CRC verification failed for member {bad_member!r} in {path}"
        )
    return member_count


def locate_unique_feature(
    names: np.ndarray | Sequence[str],
    feature_name: str = CF_FEATURE_NAME,
) -> int:
    """Return the exact, unique names index for a feature."""
    values = np.asarray(names)
    if values.ndim != 1:
        raise NoCFPostprocessError(
            f"names must be one-dimensional; got shape {values.shape}"
        )
    if values.dtype.kind not in {"U", "S"}:
        raise NoCFPostprocessError(
            f"names must use a string dtype; got {values.dtype}"
        )
    string_values = values.astype(str)
    matches = np.flatnonzero(string_values == feature_name)
    if matches.size != 1:
        raise NoCFPostprocessError(
            f"Expected exactly one exact {feature_name!r} feature in names; "
            f"found {matches.size}"
        )
    return int(matches[0])


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise NoCFPostprocessError(f"Unsupported NPY header version: {version!r}")
    return (
        tuple(int(value) for value in shape),
        bool(fortran_order),
        np.dtype(dtype),
    )


def _inspect_stored_npy_member(npz_path: Path, key: str) -> StoredNpyMember:
    member_name = f"{key}.npy"
    try:
        with zipfile.ZipFile(npz_path) as archive:
            matches = [
                member for member in archive.infolist()
                if member.filename == member_name
            ]
            if len(matches) != 1:
                raise NoCFPostprocessError(
                    f"Expected exactly one {member_name!r} member; "
                    f"found {len(matches)}"
                )
            member = matches[0]
            if member.compress_type != zipfile.ZIP_STORED:
                raise NoCFPostprocessError(
                    f"{member_name} must be ZIP_STORED for safe out-of-core access"
                )
            if member.flag_bits & 0x1:
                raise NoCFPostprocessError(
                    f"{member_name} must not be encrypted"
                )

        with npz_path.open("rb") as stream:
            stream.seek(member.header_offset)
            local_header = stream.read(_ZIP_LOCAL_HEADER.size)
            if len(local_header) != _ZIP_LOCAL_HEADER.size:
                raise NoCFPostprocessError(
                    f"Truncated ZIP local header for {member_name}"
                )
            fields = _ZIP_LOCAL_HEADER.unpack(local_header)
            if fields[0] != _ZIP_LOCAL_SIGNATURE:
                raise NoCFPostprocessError(
                    f"Invalid ZIP local header signature for {member_name}"
                )
            filename_length = int(fields[-2])
            extra_length = int(fields[-1])
            npy_offset = (
                int(member.header_offset)
                + _ZIP_LOCAL_HEADER.size
                + filename_length
                + extra_length
            )
            stream.seek(npy_offset)
            shape, fortran_order, dtype = _read_npy_header(stream)
            payload_offset = stream.tell()

        payload_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        npy_header_size = payload_offset - npy_offset
        if npy_header_size + payload_size != member.file_size:
            raise NoCFPostprocessError(
                f"Unexpected NPY member size for {member_name}: "
                f"header={npy_header_size}, payload={payload_size}, "
                f"member={member.file_size}"
            )
        return StoredNpyMember(
            shape=shape,
            dtype=dtype,
            fortran_order=fortran_order,
            payload_offset=payload_offset,
            member_size=int(member.file_size),
        )
    except NoCFPostprocessError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise NoCFPostprocessError(
            f"Could not inspect {member_name} in {npz_path}: {exc}"
        ) from exc


def _open_stored_npy_memmap(
    npz_path: Path,
    key: str,
) -> tuple[np.memmap, StoredNpyMember]:
    member = _inspect_stored_npy_member(npz_path, key)
    order = "F" if member.fortran_order else "C"
    try:
        array = np.memmap(
            npz_path,
            mode="r",
            dtype=member.dtype,
            offset=member.payload_offset,
            shape=member.shape,
            order=order,
        )
    except (OSError, ValueError) as exc:
        raise NoCFPostprocessError(
            f"Could not memory-map {key}.npy in {npz_path}: {exc}"
        ) from exc
    return array, member


def _month_ordinal(value: int) -> int:
    year, month = divmod(int(value), 100)
    if year < 1000 or year > 9999 or month < 1 or month > 12:
        raise NoCFPostprocessError(f"Invalid YYYYMM date value: {value!r}")
    return year * 12 + month - 1


def _temporary_sibling(target: Path, *, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(handle)
    return Path(name)


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _inspect_input(
    input_path: Path,
    *,
    expected_input_shape: tuple[int, int, int] | None,
    expected_date_range: tuple[int, int] | None,
) -> dict[str, Any]:
    data_member = _inspect_stored_npy_member(input_path, "data")
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            keys = list(archive.files)
            if keys != list(LAYOUT_KEYS):
                raise NoCFPostprocessError(
                    f"Input keys/order must be {list(LAYOUT_KEYS)}; got {keys}"
                )
            names = np.asarray(archive["names"]).copy()
            wficn = np.asarray(archive["wficn"]).copy()
            dates = np.asarray(archive["date"]).copy()
    except NoCFPostprocessError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise NoCFPostprocessError(f"Could not read input NPZ metadata: {exc}") from exc

    if len(data_member.shape) != 3:
        raise NoCFPostprocessError(
            f"data must be three-dimensional; got {data_member.shape}"
        )
    if data_member.dtype.kind != "f":
        raise NoCFPostprocessError(
            f"data must have floating dtype; got {data_member.dtype}"
        )
    if names.ndim != 1:
        raise NoCFPostprocessError(
            f"names must be one-dimensional; got {names.shape}"
        )
    if wficn.ndim != 1:
        raise NoCFPostprocessError(
            f"wficn must be one-dimensional; got {wficn.shape}"
        )
    if dates.ndim != 1:
        raise NoCFPostprocessError(
            f"date must be one-dimensional; got {dates.shape}"
        )
    if dates.dtype.kind not in {"i", "u"}:
        raise NoCFPostprocessError(
            f"date must have integer dtype; got {dates.dtype}"
        )
    if data_member.shape[-1] != len(names) + 1:
        raise NoCFPostprocessError(
            "data channel count must equal len(names) + 1; "
            f"got channels={data_member.shape[-1]}, names={len(names)}"
        )
    if data_member.shape[0] != len(dates):
        raise NoCFPostprocessError(
            "Time-axis mismatch between data and date: "
            f"{data_member.shape[0]} != {len(dates)}"
        )
    if data_member.shape[1] != len(wficn):
        raise NoCFPostprocessError(
            "Fund-axis mismatch between data and wficn: "
            f"{data_member.shape[1]} != {len(wficn)}"
        )
    if expected_input_shape is not None and data_member.shape != tuple(
        expected_input_shape
    ):
        raise NoCFPostprocessError(
            f"Input data shape is {data_member.shape}; "
            f"expected {tuple(expected_input_shape)}"
        )

    name_values = names.astype(str)
    if len(set(name_values.tolist())) != len(name_values):
        raise NoCFPostprocessError("names contains duplicate values")
    wficn_values = wficn.astype(str)
    if len(set(wficn_values.tolist())) != len(wficn_values):
        raise NoCFPostprocessError("wficn contains duplicate values")

    cf_name_index = locate_unique_feature(names, CF_FEATURE_NAME)
    sentiment_name_index = locate_unique_feature(names, SENTIMENT_FEATURE_NAME)
    cfnai_name_index = locate_unique_feature(names, CFNAI_FEATURE_NAME)
    expected_macro_indices = (len(names) - 2, len(names) - 1)
    if (sentiment_name_index, cfnai_name_index) != expected_macro_indices:
        raise NoCFPostprocessError(
            "Input macro names must be the final two feature names in order "
            f"{(SENTIMENT_FEATURE_NAME, CFNAI_FEATURE_NAME)}; got indices "
            f"{(sentiment_name_index, cfnai_name_index)}"
        )
    if cf_name_index >= sentiment_name_index:
        raise NoCFPostprocessError(
            "CF must be a core feature before sentiment and CFNAI"
        )

    ordinals = np.asarray(
        [_month_ordinal(int(value)) for value in dates],
        dtype=np.int64,
    )
    if len(ordinals) > 1 and not np.all(np.diff(ordinals) == 1):
        raise NoCFPostprocessError("date must contain consecutive monthly YYYYMM values")
    if expected_date_range is not None:
        expected_start, expected_end = map(int, expected_date_range)
        if len(dates) == 0 or (
            int(dates[0]) != expected_start or int(dates[-1]) != expected_end
        ):
            raise NoCFPostprocessError(
                f"date range is "
                f"{None if len(dates) == 0 else (int(dates[0]), int(dates[-1]))}; "
                f"expected {(expected_start, expected_end)}"
            )

    output_names = np.delete(names, cf_name_index)
    return {
        "keys": list(LAYOUT_KEYS),
        "data_member": data_member,
        "input_shape": data_member.shape,
        "output_shape": (
            data_member.shape[0],
            data_member.shape[1],
            data_member.shape[2] - 1,
        ),
        "names": names,
        "output_names": output_names,
        "wficn": wficn,
        "date": dates,
        "cf_name_index": cf_name_index,
        "cf_data_channel": cf_name_index + 1,
        "sentiment_input_channel": sentiment_name_index + 1,
        "cfnai_input_channel": cfnai_name_index + 1,
        "sentiment_output_channel": locate_unique_feature(
            output_names, SENTIMENT_FEATURE_NAME
        )
        + 1,
        "cfnai_output_channel": locate_unique_feature(
            output_names, CFNAI_FEATURE_NAME
        )
        + 1,
        "retained_core_feature_count": len(output_names) - 2,
    }


def _copy_without_channel(
    source: np.ndarray,
    destination: np.ndarray,
    *,
    removed_channel: int,
    chunk_months: int,
) -> None:
    if chunk_months <= 0:
        raise NoCFPostprocessError("chunk_months must be positive")
    for start in range(0, source.shape[0], chunk_months):
        stop = min(start + chunk_months, source.shape[0])
        destination[start:stop, :, :removed_channel] = source[
            start:stop, :, :removed_channel
        ]
        destination[start:stop, :, removed_channel:] = source[
            start:stop, :, removed_channel + 1:
        ]


def _require_check(
    name: str,
    condition: bool,
    details: str,
    checks: list[dict[str, Any]],
) -> None:
    checks.append({"name": name, "passed": bool(condition), "details": details})
    if not condition:
        raise NoCFPostprocessError(
            f"Validation failed: {name}: {details}"
        )


def _assert_array_equal(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    checks: list[dict[str, Any]],
) -> None:
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
        checks.append(
            {"name": name, "passed": False, "details": str(exc)[:1200]}
        )
        raise NoCFPostprocessError(f"Validation failed: {name}") from exc
    checks.append(
        {
            "name": name,
            "passed": True,
            "details": f"shape={actual.shape}, dtype={actual.dtype}",
        }
    )


def _assert_chunked_slices_equal(
    name: str,
    output_data: np.ndarray,
    input_data: np.ndarray,
    *,
    output_channels: slice,
    input_channels: slice,
    chunk_months: int,
    checks: list[dict[str, Any]],
) -> None:
    try:
        for start in range(0, input_data.shape[0], chunk_months):
            stop = min(start + chunk_months, input_data.shape[0])
            np.testing.assert_array_equal(
                output_data[start:stop, :, output_channels],
                input_data[start:stop, :, input_channels],
            )
    except AssertionError as exc:
        checks.append(
            {"name": name, "passed": False, "details": str(exc)[:1200]}
        )
        raise NoCFPostprocessError(f"Validation failed: {name}") from exc
    checks.append(
        {
            "name": name,
            "passed": True,
            "details": (
                "All elements matched exactly in monthly chunks using "
                "np.testing.assert_array_equal"
            ),
        }
    )


def _validate_output(
    output_path: Path,
    input_data: np.ndarray,
    inspection: dict[str, Any],
    *,
    chunk_months: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    output_data: np.memmap | None = None
    try:
        output_crc_member_count = _verify_npz_crc(output_path)
        _require_check(
            "output_zip_crc",
            True,
            f"All {output_crc_member_count} ZIP members passed CRC verification",
            checks,
        )
        output_data, output_member = _open_stored_npy_memmap(output_path, "data")
        try:
            with np.load(output_path, allow_pickle=False) as archive:
                output_keys = list(archive.files)
                output_names = np.asarray(archive["names"]).copy()
                output_wficn = np.asarray(archive["wficn"]).copy()
                output_dates = np.asarray(archive["date"]).copy()
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise NoCFPostprocessError(
                f"Could not reload output metadata: {exc}"
            ) from exc

        _require_check(
            "output_key_order",
            output_keys == list(LAYOUT_KEYS),
            f"actual={output_keys}, expected={list(LAYOUT_KEYS)}",
            checks,
        )
        _require_check(
            "output_data_shape",
            output_member.shape == inspection["output_shape"],
            f"actual={output_member.shape}, expected={inspection['output_shape']}",
            checks,
        )
        _require_check(
            "output_names_shape",
            output_names.shape == (len(inspection["output_names"]),),
            (
                f"actual={output_names.shape}, "
                f"expected={(len(inspection['output_names']),)}"
            ),
            checks,
        )
        _require_check(
            "feature_channel_relation",
            output_member.shape[-1] == len(output_names) + 1,
            f"channels={output_member.shape[-1]}, names={len(output_names)}",
            checks,
        )
        _require_check(
            "data_dtype_preserved",
            output_member.dtype == inspection["data_member"].dtype,
            (
                f"actual={output_member.dtype}, "
                f"expected={inspection['data_member'].dtype}"
            ),
            checks,
        )
        _require_check(
            "data_order_preserved",
            output_member.fortran_order
            == inspection["data_member"].fortran_order,
            (
                f"actual_fortran={output_member.fortran_order}, "
                f"expected_fortran={inspection['data_member'].fortran_order}"
            ),
            checks,
        )
        _assert_array_equal(
            "names_exact_after_cf_removal",
            output_names,
            inspection["output_names"],
            checks,
        )
        _require_check(
            "names_dtype_preserved",
            output_names.dtype == inspection["names"].dtype,
            (
                f"actual={output_names.dtype}, "
                f"expected={inspection['names'].dtype}"
            ),
            checks,
        )
        _require_check(
            "cf_absent_after_removal",
            int(np.count_nonzero(output_names.astype(str) == CF_FEATURE_NAME)) == 0,
            f"names={output_names.tolist()}",
            checks,
        )
        _assert_array_equal(
            "wficn_exact",
            output_wficn,
            inspection["wficn"],
            checks,
        )
        _require_check(
            "wficn_dtype_preserved",
            output_wficn.dtype == inspection["wficn"].dtype,
            (
                f"actual={output_wficn.dtype}, "
                f"expected={inspection['wficn'].dtype}"
            ),
            checks,
        )
        _assert_array_equal(
            "date_exact",
            output_dates,
            inspection["date"],
            checks,
        )
        _require_check(
            "date_dtype_preserved",
            output_dates.dtype == inspection["date"].dtype,
            (
                f"actual={output_dates.dtype}, "
                f"expected={inspection['date'].dtype}"
            ),
            checks,
        )

        output_sentiment_channel = (
            locate_unique_feature(output_names, SENTIMENT_FEATURE_NAME) + 1
        )
        output_cfnai_channel = (
            locate_unique_feature(output_names, CFNAI_FEATURE_NAME) + 1
        )
        _require_check(
            "sentiment_output_channel",
            output_sentiment_channel
            == inspection["sentiment_output_channel"]
            == output_member.shape[-1] - 2,
            (
                f"actual={output_sentiment_channel}, "
                f"expected={inspection['sentiment_output_channel']}"
            ),
            checks,
        )
        _require_check(
            "cfnai_output_channel",
            output_cfnai_channel
            == inspection["cfnai_output_channel"]
            == output_member.shape[-1] - 1,
            (
                f"actual={output_cfnai_channel}, "
                f"expected={inspection['cfnai_output_channel']}"
            ),
            checks,
        )

        removed_channel = inspection["cf_data_channel"]
        _assert_chunked_slices_equal(
            "y_exact",
            output_data,
            input_data,
            output_channels=slice(0, 1),
            input_channels=slice(0, 1),
            chunk_months=chunk_months,
            checks=checks,
        )
        _assert_chunked_slices_equal(
            "retained_channels_before_cf_exact",
            output_data,
            input_data,
            output_channels=slice(1, removed_channel),
            input_channels=slice(1, removed_channel),
            chunk_months=chunk_months,
            checks=checks,
        )
        _assert_chunked_slices_equal(
            "retained_channels_after_cf_exact",
            output_data,
            input_data,
            output_channels=slice(removed_channel, None),
            input_channels=slice(removed_channel + 1, None),
            chunk_months=chunk_months,
            checks=checks,
        )
        _assert_chunked_slices_equal(
            "sentiment_exact",
            output_data,
            input_data,
            output_channels=slice(
                output_sentiment_channel, output_sentiment_channel + 1
            ),
            input_channels=slice(
                inspection["sentiment_input_channel"],
                inspection["sentiment_input_channel"] + 1,
            ),
            chunk_months=chunk_months,
            checks=checks,
        )
        _assert_chunked_slices_equal(
            "cfnai_exact",
            output_data,
            input_data,
            output_channels=slice(output_cfnai_channel, output_cfnai_channel + 1),
            input_channels=slice(
                inspection["cfnai_input_channel"],
                inspection["cfnai_input_channel"] + 1,
            ),
            chunk_months=chunk_months,
            checks=checks,
        )

        stats = {
            "output_shape": output_member.shape,
            "output_crc_member_count": output_crc_member_count,
            "output_dtypes": {
                "data": str(output_member.dtype),
                "wficn": str(output_wficn.dtype),
                "names": str(output_names.dtype),
                "date": str(output_dates.dtype),
            },
            "sentiment_output_channel": output_sentiment_channel,
            "cfnai_output_channel": output_cfnai_channel,
        }
        return checks, stats
    finally:
        _close_memmap(output_data)
        output_data = None
        gc.collect()


def _render_checks(checks: list[dict[str, Any]]) -> list[str]:
    lines = ["| Check | Result | Details |", "|---|---|---|"]
    for check in checks:
        details = str(check["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{check['name']}` | "
            f"{'PASS' if check['passed'] else 'FAIL'} | {details} |"
        )
    return lines


def _render_success_report(
    *,
    input_path: Path,
    output_path: Path,
    input_sha_before: str,
    input_sha_after: str,
    output_sha: str,
    input_size: int,
    output_size: int,
    input_crc_member_count: int,
    inspection: dict[str, Any],
    checks: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Dataset Validation Report - no CF",
            "",
            "**Status: PASS**",
            "",
            "This artifact is based on the validated **Kaniel-compatible extended "
            "layout** and removes exactly one core `CF` (Cash Flow) feature.",
            "It is a layout-only postprocessed artifact, not a recomputed dataset. "
            "No original data source was read and no feature was recalculated, "
            "filled, standardized, interpolated, or reordered.",
            "",
            "## Files and SHA256",
            "",
            "| File | Path | Size (bytes) | SHA256 |",
            "|---|---|---:|---|",
            f"| Input compatible layout | `{input_path}` | {input_size} | "
            f"`{input_sha_after}` |",
            f"| Output no-CF layout | `{output_path}` | {output_size} | "
            f"`{output_sha}` |",
            "",
            f"- Input SHA before processing: `{input_sha_before}`",
            f"- Input SHA after processing: `{input_sha_after}`",
            f"- Input unchanged: `{input_sha_before == input_sha_after}`",
            "",
            "## Archive integrity",
            "",
            f"- Input ZIP members passing CRC: `{input_crc_member_count}`",
            f"- Output ZIP members passing CRC: "
            f"`{stats['output_crc_member_count']}`",
            "",
            "## Removed feature",
            "",
            f"- Exact feature name: `{CF_FEATURE_NAME}`",
            f"- Input `names` index (0-based): "
            f"`{inspection['cf_name_index']}`",
            f"- Input `data` channel (0-based): "
            f"`{inspection['cf_data_channel']}`",
            "- `CFNAI`, `sentiment`, `y`, and every other core feature were retained.",
            "",
            "## Schema",
            "",
            "| Item | Input | Output |",
            "|---|---|---|",
            f"| Keys | `{inspection['keys']}` | `{list(LAYOUT_KEYS)}` |",
            f"| data shape | `{inspection['input_shape']}` | "
            f"`{stats['output_shape']}` |",
            f"| names shape | `{inspection['names'].shape}` | "
            f"`{inspection['output_names'].shape}` |",
            f"| data dtype | `{inspection['data_member'].dtype}` | "
            f"`{stats['output_dtypes']['data']}` |",
            f"| wficn | `{inspection['wficn'].shape}`, "
            f"`{inspection['wficn'].dtype}` | unchanged |",
            f"| date | `{inspection['date'].shape}`, "
            f"`{inspection['date'].dtype}` | unchanged |",
            f"| sentiment data channel | "
            f"`{inspection['sentiment_input_channel']}` | "
            f"`{stats['sentiment_output_channel']}` |",
            f"| CFNAI data channel | `{inspection['cfnai_input_channel']}` | "
            f"`{stats['cfnai_output_channel']}` |",
            "",
            "## Strict validation",
            "",
            "Every retained data element was compared with "
            "`np.testing.assert_array_equal()` in monthly chunks. This strictly "
            f"covers `y`, all {inspection['retained_core_feature_count']} retained "
            "core features, `sentiment`, `CFNAI`, and "
            "all existing missing-value sentinel positions.",
            "",
            *_render_checks(checks),
            "",
            f"- Checks passed: `{len(checks)}`",
            f"- Report generated UTC: "
            f"`{datetime.now(timezone.utc).isoformat()}`",
            "",
        ]
    )


def _render_failure_report(
    *,
    input_path: Path,
    output_path: Path,
    reason: str,
    input_sha: str | None,
    output_existed_before: bool,
) -> str:
    preservation = (
        "The pre-existing output was preserved."
        if output_existed_before
        else "No final output NPZ was published."
    )
    return "\n".join(
        [
            "# Dataset Validation Report - no CF",
            "",
            "**Status: FAIL**",
            "",
            "The independent CF-removal layout postprocessing did not complete.",
            preservation,
            "",
            f"- Input: `{input_path}`",
            f"- Intended output: `{output_path}`",
            f"- Input SHA256: `{input_sha or 'not available'}`",
            f"- Failure reason: `{reason}`",
            f"- Report generated UTC: "
            f"`{datetime.now(timezone.utc).isoformat()}`",
            "",
        ]
    )


def _publish_output_and_report(
    *,
    temporary_npz: Path,
    output_path: Path,
    temporary_report: Path,
    report_path: Path,
) -> None:
    output_backup: Path | None = None
    report_backup: Path | None = None
    new_output_published = False
    publish_succeeded = False
    try:
        if output_path.exists():
            output_backup = _temporary_sibling(output_path, suffix=".backup")
            output_backup.unlink(missing_ok=True)
            os.replace(output_path, output_backup)
        if report_path.exists():
            report_backup = _temporary_sibling(report_path, suffix=".backup")
            report_backup.unlink(missing_ok=True)
            os.replace(report_path, report_backup)

        os.replace(temporary_npz, output_path)
        new_output_published = True
        os.replace(temporary_report, report_path)
        publish_succeeded = True
    except Exception as publish_error:
        recovery_errors: list[str] = []
        if new_output_published:
            try:
                output_path.unlink(missing_ok=True)
            except OSError as exc:
                recovery_errors.append(
                    f"could not remove new output {output_path}: {exc}"
                )
        if output_backup is not None and output_backup.exists():
            try:
                output_path.unlink(missing_ok=True)
                os.replace(output_backup, output_path)
                output_backup = None
            except OSError as exc:
                recovery_errors.append(
                    f"could not restore output backup {output_backup}: {exc}"
                )
        if report_backup is not None and report_backup.exists():
            try:
                report_path.unlink(missing_ok=True)
                os.replace(report_backup, report_path)
                report_backup = None
            except OSError as exc:
                recovery_errors.append(
                    f"could not restore report backup {report_backup}: {exc}"
                )
        if recovery_errors:
            retained_backups = [
                str(path)
                for path in (output_backup, report_backup)
                if path is not None and path.exists()
            ]
            raise NoCFPostprocessError(
                f"Atomic publish failed: {publish_error}. Recovery errors: "
                f"{'; '.join(recovery_errors)}. Retained backups: "
                f"{retained_backups}"
            ) from publish_error
        raise
    finally:
        if publish_succeeded:
            if output_backup is not None:
                output_backup.unlink(missing_ok=True)
            if report_backup is not None:
                report_backup.unlink(missing_ok=True)


def remove_cf_from_compatible_layout(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
    expected_input_shape: tuple[int, int, int] | None = None,
    expected_date_range: tuple[int, int] | None = None,
    chunk_months: int = 8,
) -> dict[str, Any]:
    """Remove only the exact core CF channel from a compatible-layout NPZ."""
    source_path = Path(input_path).resolve()
    target_path = Path(output_path).resolve()
    validation_report_path = Path(report_path).resolve()

    if len({source_path, target_path, validation_report_path}) != 3:
        raise NoCFPostprocessError(
            "Input, output, and report paths must all be different"
        )

    input_sha_before: str | None = None
    temporary_data: Path | None = None
    temporary_npz: Path | None = None
    temporary_report: Path | None = None
    source_data: np.memmap | None = None
    destination_data: np.memmap | None = None
    output_existed_before = target_path.exists()

    try:
        if chunk_months <= 0:
            raise NoCFPostprocessError("chunk_months must be positive")
        if not source_path.is_file():
            raise NoCFPostprocessError(f"Input NPZ does not exist: {source_path}")
        if target_path.exists() and not overwrite:
            raise NoCFPostprocessError(
                f"Output already exists; pass overwrite=True to replace it: "
                f"{target_path}"
            )

        input_sha_before = file_sha256(source_path)
        input_crc_member_count = _verify_npz_crc(source_path)
        inspection = _inspect_input(
            source_path,
            expected_input_shape=expected_input_shape,
            expected_date_range=expected_date_range,
        )
        source_data, source_member = _open_stored_npy_memmap(source_path, "data")
        if (
            source_member.shape != inspection["input_shape"]
            or source_member.dtype != inspection["data_member"].dtype
        ):
            raise NoCFPostprocessError(
                "Input data header changed between inspection and mapping"
            )

        temporary_data = _temporary_sibling(target_path, suffix=".data.npy")
        temporary_data.unlink(missing_ok=True)
        destination_data = np.lib.format.open_memmap(
            temporary_data,
            mode="w+",
            dtype=source_member.dtype,
            shape=inspection["output_shape"],
            fortran_order=source_member.fortran_order,
        )
        _copy_without_channel(
            source_data,
            destination_data,
            removed_channel=inspection["cf_data_channel"],
            chunk_months=chunk_months,
        )
        destination_data.flush()

        temporary_npz = _temporary_sibling(target_path, suffix=".npz.tmp")
        with temporary_npz.open("wb") as handle:
            np.savez(
                handle,
                data=destination_data,
                wficn=inspection["wficn"],
                names=inspection["output_names"],
                date=inspection["date"],
            )

        _close_memmap(destination_data)
        destination_data = None
        gc.collect()
        temporary_data.unlink(missing_ok=True)
        temporary_data = None

        checks, stats = _validate_output(
            temporary_npz,
            source_data,
            inspection,
            chunk_months=chunk_months,
        )

        input_sha_after = file_sha256(source_path)
        if input_sha_after != input_sha_before:
            raise NoCFPostprocessError("Input SHA256 changed during postprocessing")
        output_sha = file_sha256(temporary_npz)
        input_size = source_path.stat().st_size
        output_size = temporary_npz.stat().st_size

        report = _render_success_report(
            input_path=source_path,
            output_path=target_path,
            input_sha_before=input_sha_before,
            input_sha_after=input_sha_after,
            output_sha=output_sha,
            input_size=input_size,
            output_size=output_size,
            input_crc_member_count=input_crc_member_count,
            inspection=inspection,
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
            "output": str(target_path),
            "report": str(validation_report_path),
            "removed_feature": CF_FEATURE_NAME,
            "removed_names_index": inspection["cf_name_index"],
            "removed_data_channel": inspection["cf_data_channel"],
            "input_sha256": input_sha_after,
            "output_sha256": output_sha,
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "schema": {
                "keys": list(LAYOUT_KEYS),
                "data_shape": stats["output_shape"],
                "names_shape": inspection["output_names"].shape,
                "dtypes": stats["output_dtypes"],
                "sentiment_channel": stats["sentiment_output_channel"],
                "cfnai_channel": stats["cfnai_output_channel"],
            },
            "checks_passed": len(checks),
        }
        _close_memmap(source_data)
        source_data = None
        gc.collect()
        _publish_output_and_report(
            temporary_npz=temporary_npz,
            output_path=target_path,
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
            output_path=target_path,
            reason=reason,
            input_sha=input_sha_before,
            output_existed_before=output_existed_before,
        )
        write_text_atomic(validation_report_path, failure_report)
        if isinstance(exc, NoCFPostprocessError):
            raise
        raise NoCFPostprocessError(reason) from exc
    finally:
        _close_memmap(destination_data)
        _close_memmap(source_data)
        destination_data = None
        source_data = None
        gc.collect()
        if temporary_data is not None:
            temporary_data.unlink(missing_ok=True)
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)
