"""General time, I/O, and numerical helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


def open_dataset_robust(path: str | Path, *, decode_times: bool = True) -> xr.Dataset:
    """Open a NetCDF file by trying the common engines in a safe order."""
    errors: list[str] = []
    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = {"decode_times": decode_times}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on local engines
            errors.append(f"{engine or 'auto'}: {exc}")
    joined = "\n".join(errors)
    raise OSError(f"Could not open NetCDF file {path}. Tried all supported engines:\n{joined}")


def water_year(value: pd.Timestamp | str) -> int:
    date = pd.Timestamp(value)
    return date.year + 1 if date.month >= 10 else date.year


def add_water_year(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result[date_column])
    result["water_year"] = np.where(dates.dt.month >= 10, dates.dt.year + 1, dates.dt.year).astype(int)
    return result


def water_year_bounds(wy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(wy - 1, 10, 1), pd.Timestamp(wy, 9, 30)


def expected_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")


def weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(data) & np.isfinite(w) & (w > 0)
    total_weight = float(np.nansum(w[w > 0]))
    valid_weight = float(np.nansum(w[valid]))
    if valid_weight <= 0:
        return math.nan, 0.0
    return float(np.nansum(data[valid] * w[valid]) / valid_weight), valid_weight / total_weight if total_weight else 0.0


def depth_to_volume_m3(depth_mm: np.ndarray, fractional_weights: np.ndarray, cell_area_m2: float) -> float:
    values = np.asarray(depth_mm, dtype=float)
    weights = np.asarray(fractional_weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.nansum(values[valid] * 1.0e-3 * weights[valid] * cell_area_m2))




def stable_hash(data: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-serializable metadata."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Return a deterministic SHA-256 hash of an array's shape, dtype, and bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()

def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def read_csv_dates(path: str | Path, date_columns: Iterable[str] = ("date",)) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def assert_units(actual: str | None, accepted: set[str], context: str) -> str:
    normalized = (actual or "").strip().lower().replace("**", "^")
    normalized = " ".join(normalized.split())
    accepted_normalized = {" ".join(item.strip().lower().split()) for item in accepted}
    if normalized not in accepted_normalized:
        raise ValueError(f"Unexpected units for {context}: {actual!r}; expected one of {sorted(accepted)}")
    return normalized


def nice_upper(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(value))
    scaled = value / exponent
    for candidate in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]:
        if scaled <= candidate:
            return candidate * exponent
    return 10 * exponent
