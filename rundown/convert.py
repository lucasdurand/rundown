"""Turn non-pandas frames (polars, Arrow-compatible tables, ...) into pandas for profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_polars(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "polars"


def _polars_dtype(dtype) -> str:
    # polars' own short names (i64, str, datetime[μs, UTC], ...), as shown in its table headers
    short = getattr(dtype, "_string_repr", None)
    try:
        return short() if callable(short) else str(dtype)
    except Exception:  # pragma: no cover
        return str(dtype)


def _polars_series(s) -> pd.Series:
    dtype = s.dtype
    try:
        values = s.to_numpy()
    except Exception:
        values = np.array(s.to_list(), dtype=object)
    if dtype.is_integer() and s.null_count():
        out = pd.Series(values).astype("Int64" if dtype.is_signed_integer() else "UInt64")
    elif str(dtype) == "Boolean" and s.null_count():
        out = pd.Series(values, dtype="boolean")
    else:
        out = pd.Series(values)
    tz = getattr(dtype, "time_zone", None)
    if tz:
        out = out.dt.tz_localize("UTC").dt.tz_convert(tz)
    return out


def to_pandas(obj, max_columns: int | None = None) -> tuple[pd.DataFrame | pd.Series, str, dict[str, str]]:
    """Return (pandas frame/series, caption label, {column: dtype label overriding the pandas one})."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj, type(obj).__name__, {}

    if _is_polars(obj):
        kind = type(obj).__name__
        if kind == "LazyFrame":
            raise TypeError("rundown needs materialized data: call .collect() on the LazyFrame first")
        if kind == "Series":
            name = obj.name or "value"
            return _polars_series(obj).rename(name), "polars Series", {name: _polars_dtype(obj.dtype)}
        if kind == "DataFrame":
            cols = obj.columns if max_columns is None else obj.columns[:max_columns]
            df = pd.DataFrame({c: _polars_series(obj.get_column(c)) for c in cols})
            # keep the true width for the caption even though only `max_columns` were converted
            df.attrs["rundown_ncols"] = obj.width
            return df, "polars DataFrame", {c: _polars_dtype(obj.schema[c]) for c in cols}

    label = f"{type(obj).__module__.split('.')[0]} {type(obj).__name__}"
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas(), label, {}
    if hasattr(obj, "__arrow_c_stream__") and hasattr(pd.DataFrame, "from_arrow"):
        return pd.DataFrame.from_arrow(obj), label, {}
    if hasattr(obj, "__dataframe__"):
        return pd.api.interchange.from_dataframe(obj), label, {}
    raise TypeError(f"rundown can't summarize a {type(obj).__name__}; pass a pandas or polars DataFrame/Series")
