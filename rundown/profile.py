"""Column profiling: everything the margin viz needs, computed with vectorized pandas/numpy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMERIC = "numeric"
DATETIME = "datetime"
TIMEDELTA = "timedelta"
CATEGORICAL = "categorical"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    count: int
    nulls: int
    unique: int | None = None
    stats: dict = field(default_factory=dict)
    # histogram: (counts, edges) on the numeric axis (ns for datetimes/timedeltas)
    histogram: tuple[np.ndarray, np.ndarray] | None = None
    # top values: list of (label, count); `other` = count of non-null values not shown
    top: list[tuple[str, int]] = field(default_factory=list)
    other: int = 0
    sparkline: dict | None = None
    # right-align cells (numeric dtype, whatever the margin viz kind)
    align_right: bool = False

    @property
    def null_pct(self) -> float:
        return 100 * self.nulls / self.count if self.count else 0.0


def _numeric_axis(s: pd.Series) -> np.ndarray | None:
    """Values of `s` as float64 (ns for datetimes/timedeltas, 0/1 for bools), NaN for nulls."""
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.to_numpy("float64", na_value=np.nan)
    if pd.api.types.is_datetime64_any_dtype(s.dtype) or pd.api.types.is_timedelta64_dtype(s.dtype):
        out = s.to_numpy("datetime64[ns]" if pd.api.types.is_datetime64_any_dtype(s.dtype) else "timedelta64[ns]")
        ints = out.view("int64").astype("float64")
        ints[np.isnat(out)] = np.nan
        return ints
    if pd.api.types.is_numeric_dtype(s.dtype):
        return pd.to_numeric(s, errors="coerce").to_numpy("float64", na_value=np.nan)
    return None


def _kind(s: pd.Series, unique: int, discrete_threshold: int) -> str:
    dtype = s.dtype
    if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return CATEGORICAL
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return DATETIME
    if pd.api.types.is_timedelta64_dtype(dtype):
        return TIMEDELTA
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
        return CATEGORICAL if unique <= discrete_threshold else NUMERIC
    return CATEGORICAL


def _value_counts(s: pd.Series) -> pd.Series:
    try:
        return s.value_counts(dropna=True, sort=True)
    except TypeError:  # unhashable values, e.g. lists or dicts
        return s.dropna().astype(str).value_counts(sort=True)


def _histogram(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    lo, hi = finite.min(), finite.max()
    if lo == hi:
        return np.array([finite.size]), np.array([lo, hi])
    return np.histogram(finite, bins=bins, range=(lo, hi))


def sparkline_series(values: np.ndarray, points: int) -> dict | None:
    """Downsample `values` (in row order) to at most `points` buckets of (min, mean, max)."""
    n = values.size
    if n < 2 or not np.isfinite(values).any():
        return None
    values = np.where(np.isfinite(values), values, np.nan)
    if n <= points:
        return {"n": n, "mean": values, "min": values, "max": values, "bucketed": False}
    size = int(np.ceil(n / points))
    padded = np.full(size * int(np.ceil(n / size)), np.nan)
    padded[:n] = values
    buckets = padded.reshape(-1, size)
    with np.errstate(all="ignore"), _quiet_nan_warnings():
        return {
            "n": n,
            "mean": np.nanmean(buckets, axis=1),
            "min": np.nanmin(buckets, axis=1),
            "max": np.nanmax(buckets, axis=1),
            "bucketed": True,
        }


class _quiet_nan_warnings:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def profile_column(
    s: pd.Series,
    name: str,
    *,
    bins: int = 20,
    top_k: int = 5,
    discrete_threshold: int = 10,
    sparkline_points: int | None = None,
    sparkline_source: pd.Series | None = None,
) -> ColumnProfile:
    count = len(s)
    nulls = int(s.isna().sum())
    counts = _value_counts(s)
    unique = int(counts.size)
    kind = _kind(s, unique, discrete_threshold)
    prof = ColumnProfile(name=name, dtype=str(s.dtype), kind=kind, count=count, nulls=nulls, unique=unique)
    prof.align_right = pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)

    axis = _numeric_axis(s) if kind != CATEGORICAL or pd.api.types.is_numeric_dtype(s.dtype) else None
    if axis is not None and np.isfinite(axis).any():
        finite = axis[np.isfinite(axis)]
        prof.stats = {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "median": float(np.median(finite)),
        }
        tz = getattr(s.dtype, "tz", None)
        if tz is not None:
            prof.stats["tz"] = tz
        infs = int(np.isinf(axis).sum())
        if infs:
            prof.stats["inf"] = infs

    if kind == CATEGORICAL:
        shown = counts.iloc[:top_k]
        prof.top = [(_label(v), int(c)) for v, c in shown.items()]
        prof.other = int(counts.iloc[top_k:].sum())
    elif axis is not None:
        prof.histogram = _histogram(axis, bins)

    if sparkline_points:
        source = s if sparkline_source is None else sparkline_source
        values = _numeric_axis(source)
        if values is not None:
            prof.sparkline = sparkline_series(values, sparkline_points)
    return prof


def _label(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)
