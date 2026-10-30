"""
Rundown: a quick, good-looking overview of a pandas DataFrame for notebooks.

    import rundown
    df.rundown()                  # margin viz: a histogram / top-values chart above every column
    df.rundown(sparklines=True)   # ...plus a sparkline of each column in row order

Works on pandas and polars DataFrames/Series (and anything with `to_pandas()` or the Arrow
PyCapsule interface). `rundown.rundown(df)` is the same thing as a plain function.

"Do you have that rundown ready for me, Jim?"
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .convert import to_pandas
from .profile import ColumnProfile, profile_column
from .render import render

try:
    __version__ = version("rundown")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["Rundown", "rundown", "ColumnProfile"]

SAMPLE_MODES = ("random", "head", "tail")

# srcdoc iframes share the parent's origin, so the page can resize its own frame to fit.
# Goes after the content so document.body exists when the ResizeObserver is attached.
_FIT_FRAME = (
    "<script>(function(){function fit(){var f=window.frameElement;"
    "if(f)f.style.height=Math.ceil(document.body.getBoundingClientRect().height)+'px';}"
    "addEventListener('load',fit);if(window.ResizeObserver)new ResizeObserver(fit).observe(document.body);})();</script>"
)


class Rundown:
    """The rendered rundown. Displays itself in Jupyter; use `to_html()` or `save()` elsewhere."""

    def __init__(self, html: str, profiles: list[ColumnProfile], shape: tuple[int, int]):
        self.html = html
        self.profiles = profiles
        self.shape = shape

    def _repr_html_(self) -> str:
        return self.html

    def to_html(self) -> str:
        return self.html

    def _page(self, head: str = "", tail: str = "") -> str:
        return f'<!doctype html><meta charset="utf-8"><title>rundown</title>{head}{self.html}{tail}'

    def save(self, path: str | Path) -> Path:
        """Write a standalone HTML page."""
        path = Path(path)
        path.write_text(self._page(), encoding="utf-8")
        return path

    def to_dash(self, **kwargs):
        """
        A Dash `html.Iframe` showing this rundown (sorting/filtering included), sized to fit its content.

        Keyword arguments are passed to `html.Iframe` (e.g. `id`, `style`). Requires `dash`.
        """
        from dash import html

        page = self._page("<style>body{margin:0;display:flow-root}</style>", _FIT_FRAME)
        style = {"width": "100%", "border": "0", "height": "520px", **kwargs.pop("style", {})}
        return html.Iframe(srcDoc=page, style=style, **kwargs)

    def __repr__(self) -> str:
        rows, cols = self.shape
        return f"<Rundown: {rows:,} rows \u00d7 {cols:,} columns>"


def _label(name) -> str:
    if isinstance(name, tuple):
        return " / ".join(str(part) for part in name if str(part) != "")
    return str(name)


def _columns(df: pd.DataFrame, max_columns: int | None) -> tuple[list[tuple[str, pd.Series]], int]:
    """(label, series) for index levels worth showing, then up to `max_columns` columns."""
    out = []
    index = df.index
    trivial = isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1 and index.name is None
    if not trivial:
        for i in range(index.nlevels):
            name = index.names[i]
            label = _label(name) if name is not None else ("index" if index.nlevels == 1 else f"level_{i}")
            out.append((label, pd.Series(index.get_level_values(i), name=label).reset_index(drop=True)))
    n_index = len(out)
    ncols = df.shape[1] if max_columns is None else min(df.shape[1], max_columns)
    for j in range(ncols):
        out.append((_label(df.columns[j]), df.iloc[:, j].reset_index(drop=True)))
    return out, n_index


def rundown(
    df,
    n: int = 10,
    *,
    sample: str = "random",
    sparklines: bool = False,
    max_columns: int | None = 50,
    max_height: str = "480px",
    max_chars: int = 40,
    bins: int = 20,
    top_k: int = 5,
    max_profile_rows: int | None = 1_000_000,
    sparkline_points: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> Rundown:
    """
    Summarize a DataFrame: a "margin viz" above each column, with a sample of rows underneath.

    Numeric and datetime columns get a histogram, low-cardinality or text columns get their top
    values, and every column shows null rate and a couple of key stats. The table has a sticky
    header, click-to-sort columns and a row filter.

    Parameters
    ----------
    df : pandas or polars DataFrame/Series (or anything with `to_pandas()` / `__arrow_c_stream__`).
    n : number of rows to show.
    sample : which rows to show: "random" (kept in their original order), "head" or "tail".
    sparklines : add a row of sparklines showing each numeric/datetime/bool column's values
        in row order (downsampled to `sparkline_points` min/mean/max buckets).
    max_columns : only render the first `max_columns` columns (None for all).
    max_height : CSS max height of the scrollable table.
    max_chars : truncate cell text longer than this (full text on hover).
    bins : histogram bins.
    top_k : top values shown for categorical/text columns.
    max_profile_rows : compute stats from a random sample of this many rows on bigger frames
        (None to always use every row). Sparklines always use every row.
    random_state : seed or numpy Generator for row/profile sampling.
    """
    if sample not in SAMPLE_MODES:
        raise ValueError(f"sample must be one of {SAMPLE_MODES}, got {sample!r}")
    df, kind, dtypes = to_pandas(df, max_columns)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else "value")
    rng = np.random.default_rng(random_state)
    nrows, ncols = df.shape[0], df.attrs.get("rundown_ncols", df.shape[1])
    columns, n_index = _columns(df, max_columns)

    profile_pos = None
    if max_profile_rows is not None and nrows > max_profile_rows:
        profile_pos = np.sort(rng.choice(nrows, size=max_profile_rows, replace=False))
    profiles = [
        profile_column(
            s if profile_pos is None else s.iloc[profile_pos],
            label,
            bins=bins,
            top_k=top_k,
            sparkline_points=sparkline_points if sparklines else None,
            sparkline_source=s,
        )
        for label, s in columns
    ]
    for prof in profiles:
        prof.dtype = dtypes.get(prof.name, prof.dtype)

    k = max(0, min(n, nrows))
    if sample == "head":
        pos = np.arange(k)
    elif sample == "tail":
        pos = np.arange(nrows - k, nrows)
    else:
        pos = np.sort(rng.choice(nrows, size=k, replace=False))
    rows = pd.DataFrame({j: s.iloc[pos].reset_index(drop=True) for j, (_, s) in enumerate(columns)})

    caption = f"{kind} \u00b7 <b>{nrows:,}</b> rows \u00d7 <b>{ncols:,}</b> columns"
    caption += f" \u00b7 showing {k:,} {sample if k < nrows else ''} rows".replace("  ", " ")
    if max_columns is not None and ncols > max_columns:
        caption += f" \u00b7 first {max_columns:,} columns"
    if profile_pos is not None:
        caption += f" \u00b7 stats from a {max_profile_rows:,}-row sample"

    html = render(
        profiles,
        rows,
        n_index=n_index,
        caption=caption,
        max_height=max_height,
        max_chars=max_chars,
        sparklines=sparklines,
    )
    return Rundown(html, profiles, (nrows, ncols))


pd.DataFrame.rundown = rundown
pd.Series.rundown = rundown

try:
    import polars as _pl
except ImportError:  # pragma: no cover
    pass
else:
    _pl.DataFrame.rundown = rundown
    _pl.Series.rundown = rundown
