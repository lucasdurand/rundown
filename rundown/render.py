"""HTML/SVG rendering for rundowns. No plotting libraries: every chart is a few hundred bytes of inline SVG."""

from __future__ import annotations

import html
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .profile import CATEGORICAL, DATETIME, NUMERIC, TIMEDELTA, ColumnProfile

VIZ_WIDTH = 120
HIST_HEIGHT = 36
SPARK_HEIGHT = 28

_ASSETS = Path(__file__).parent / "assets"


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


# ---------------------------------------------------------------- number / value formatting


def compact(x: float) -> str:
    """Short human-friendly number: 0.123, 29.7, 891, 12.3k, 4.56M."""
    if x is None or not math.isfinite(x):
        return str(x)
    a = abs(x)
    if a == 0:
        return "0"
    if a >= 1e15 or a < 1e-3:
        return f"{x:.2e}"
    for suffix, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if a >= div:
            s = f"{x / div:.3g}"
            return (f"{x / div:.0f}" if "e" in s else s) + suffix
    return f"{x:.3g}"


def axis_label(prof: ColumnProfile, v: float) -> str:
    """Format a value on a column's numeric axis (ns for datetimes/timedeltas)."""
    if prof.kind == DATETIME:
        ts = pd.Timestamp(int(v), tz="UTC" if "tz" in prof.stats else None)
        if "tz" in prof.stats:
            ts = ts.tz_convert(prof.stats["tz"])
        span = prof.stats["max"] - prof.stats["min"]
        if span >= 2 * 86400e9:
            return ts.strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%d %H:%M") if span >= 86400e9 else ts.strftime("%H:%M:%S")
    if prof.kind == TIMEDELTA:
        return duration(v)
    return compact(v)


def duration(ns: float) -> str:
    """Short duration: 4ns, 3.43ms, 2.5min, 1.2d."""
    sign, a = ("-" if ns < 0 else ""), abs(ns)
    for unit, size in (("d", 86400e9), ("h", 3600e9), ("min", 60e9), ("s", 1e9), ("ms", 1e6), ("\u00b5s", 1e3)):
        if a >= size:
            return f"{sign}{a / size:.3g}{unit}"
    return f"{sign}{a:.0f}ns"


def _float_formatter(values: np.ndarray):
    finite = np.abs(values[np.isfinite(values)])
    if not finite.size:
        return lambda x: f"{x}"
    if np.all(finite == np.round(finite)) and finite.max() < 1e15:
        return lambda x: f"{x:,.1f}"
    nonzero = finite[finite > 0]
    typical = float(np.median(nonzero)) if nonzero.size else 1.0
    if finite.max() >= 1e15 or typical < 1e-4:
        return lambda x: f"{x:.4g}"
    cap = min(max(3 - math.floor(math.log10(typical)), 1), 6)
    sample = values[np.isfinite(values)][:1000]
    decimals = next(d for d in range(1, cap + 1) if d == cap or np.allclose(np.round(sample, d), sample, rtol=0, atol=1e-9))
    return lambda x: f"{x:,.{decimals}f}"


def format_cells(s: pd.Series, max_chars: int) -> list[tuple[str, str]]:
    """Format the displayed values of one column. Returns (td attributes, inner html) per value."""
    isna = s.isna().to_numpy()
    out: list[tuple[str, str]] = []
    dtype = s.dtype
    if pd.api.types.is_bool_dtype(dtype):
        fmt, num = str, False
    elif pd.api.types.is_integer_dtype(dtype):
        fmt, num = (lambda x: f"{int(x):,}"), True
    elif pd.api.types.is_float_dtype(dtype):
        fmt, num = _float_formatter(s.to_numpy("float64", na_value=np.nan)), True
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        valid = s.dropna()
        dates_only = len(valid) > 0 and bool((valid == valid.dt.normalize()).all())
        fmt = (lambda x: x.strftime("%Y-%m-%d")) if dates_only else (lambda x: str(x))
        num = False
    elif pd.api.types.is_timedelta64_dtype(dtype):
        fmt, num = (lambda x: duration(x.value)), False
    else:
        fmt, num = str, False
    null_cls = "rd-null-cell rd-num" if num else "rd-null-cell"

    for value, null in zip(s.tolist(), isna):
        if null:
            label = "NaN" if value is None or isinstance(value, float) else str(value)
            if value is None:
                label = "None"
            out.append((f' class="{null_cls}" data-null', f'<span class="rd-null">{_esc(label)}</span>'))
            continue
        text = fmt(value)
        attrs = ""
        if num:
            attrs = f' class="rd-num" data-v="{_esc(float(value))}"'
        elif isinstance(value, pd.Timestamp):
            attrs = f' data-v="{value.value}"'
        elif isinstance(value, pd.Timedelta):
            attrs = f' data-v="{value.value}"'
        if len(text) > max_chars:
            attrs += f' title="{_esc(text)}"'
            text = text[: max_chars - 1] + "\u2026"
        out.append((attrs, _esc(text)))
    return out


# ---------------------------------------------------------------- margin viz


def histogram_svg(prof: ColumnProfile) -> str:
    counts, edges = prof.histogram
    peak = counts.max() or 1
    n = len(counts)
    bw = VIZ_WIDTH / n
    gap = 1 if bw > 3 else 0
    rects = []
    for i, c in enumerate(counts):
        h = 0 if c == 0 else max(1.5, (HIST_HEIGHT - 2) * c / peak)
        lo, hi = axis_label(prof, edges[i]), axis_label(prof, edges[i + 1])
        rects.append(
            f'<rect x="{i * bw:.2f}" y="{HIST_HEIGHT - h:.2f}" width="{max(bw - gap, 0.5):.2f}" height="{h:.2f}">'
            f"<title>{_esc(lo)} \u2013 {_esc(hi)}: {int(c):,}</title></rect>"
        )
    # an invisible full-height hit area makes tiny bars easy to hover
    return (
        f'<svg class="rd-hist" width="{VIZ_WIDTH}" height="{HIST_HEIGHT}" viewBox="0 0 {VIZ_WIDTH} {HIST_HEIGHT}" role="img">'
        + "".join(rects)
        + f'<line x1="0" x2="{VIZ_WIDTH}" y1="{HIST_HEIGHT - 0.5}" y2="{HIST_HEIGHT - 0.5}"/></svg>'
        + f'<div class="rd-axis"><span>{_esc(axis_label(prof, edges[0]))}</span>'
        + f"<span>{_esc(axis_label(prof, edges[-1]))}</span></div>"
    )


def bars_html(prof: ColumnProfile, max_chars: int) -> str:
    total = prof.count or 1
    peak = max([c for _, c in prof.top] + [1])
    rows = []
    for label, c in prof.top:
        short = label if len(label) <= max_chars else label[: max_chars - 1] + "\u2026"
        rows.append(
            f'<div class="rd-bar" title="{_esc(label)}: {c:,}">'
            f'<span class="rd-bar-fill" style="width:{100 * c / peak:.1f}%"></span>'
            f'<span class="rd-bar-label">{_esc(short)}</span><span class="rd-bar-pct">{100 * c / total:.1f}%</span></div>'
        )
    if prof.other:
        n_other = (prof.unique or 0) - len(prof.top)
        rows.append(
            f'<div class="rd-bar rd-bar-other"><span class="rd-bar-label">+{n_other:,} more</span>'
            f'<span class="rd-bar-pct">{100 * prof.other / total:.1f}%</span></div>'
        )
    return f'<div class="rd-bars">{"".join(rows)}</div>'


def _path(xs, ys) -> str:
    """SVG path through (x, y), breaking at NaNs."""
    parts, pen_down = [], False
    for x, y in zip(xs, ys):
        if not math.isfinite(y):
            pen_down = False
            continue
        parts.append(f"{'L' if pen_down else 'M'}{x:.1f},{y:.1f}")
        pen_down = True
    return "".join(parts)


def sparkline_svg(prof: ColumnProfile) -> str:
    sp = prof.sparkline
    lo, hi = np.nanmin(sp["min"]), np.nanmax(sp["max"])
    span = (hi - lo) or 1.0
    m = len(sp["mean"])
    xs = np.linspace(1, VIZ_WIDTH - 3, m)

    def y(v):
        return SPARK_HEIGHT - 3 - (np.asarray(v) - lo) / span * (SPARK_HEIGHT - 6)

    band = ""
    if sp["bucketed"]:
        top, bottom = y(sp["max"]), y(sp["min"])
        ok = np.isfinite(top)
        if ok.any():
            pts = [f"{x:.1f},{t:.1f}" for x, t in zip(xs[ok], top[ok])]
            pts += [f"{x:.1f},{b:.1f}" for x, b in zip(xs[ok][::-1], bottom[ok][::-1])]
            band = f'<polygon class="rd-spark-band" points="{" ".join(pts)}"/>'
    ym = y(sp["mean"])
    last = np.flatnonzero(np.isfinite(ym))
    dot = f'<circle cx="{xs[last[-1]]:.1f}" cy="{ym[last[-1]]:.1f}" r="1.8"/>' if last.size else ""
    how = f"{m} buckets (min/mean/max)" if sp["bucketed"] else "every value"
    title = f"{sp['n']:,} values in row order, {how}; range {axis_label(prof, lo)} \u2013 {axis_label(prof, hi)}"
    return (
        f'<svg class="rd-spark" width="{VIZ_WIDTH}" height="{SPARK_HEIGHT}" viewBox="0 0 {VIZ_WIDTH} {SPARK_HEIGHT}" role="img">'
        f"<title>{_esc(title)}</title>{band}<path d=\"{_path(xs, ym)}\"/>{dot}</svg>"
    )


def stats_html(prof: ColumnProfile) -> str:
    chips = []
    st = prof.stats
    if prof.kind == NUMERIC and st:
        chips.append(f'<span title="mean">\u03bc {compact(st["mean"])}</span>')
        chips.append(f'<span title="standard deviation">\u03c3 {compact(st["std"])}</span>')
    elif prof.kind in (DATETIME, TIMEDELTA) or prof.kind == CATEGORICAL:
        chips.append(f'<span title="distinct values">{prof.unique:,} unique</span>')
    if prof.nulls:
        chips.append(
            f'<span class="rd-warn" title="{prof.nulls:,} missing values">{prof.null_pct:.1f}% null</span>'
        )
    if st.get("inf"):
        chips.append(f'<span class="rd-warn" title="infinite values">{st["inf"]:,} inf</span>')
    return f'<div class="rd-stats">{"".join(chips)}</div>'


def margin_html(prof: ColumnProfile) -> str:
    if prof.histogram is not None:
        viz = histogram_svg(prof)
    elif prof.top:
        viz = bars_html(prof, max_chars=18)
    else:
        viz = '<div class="rd-empty">no values</div>'
    return viz + stats_html(prof)


# ---------------------------------------------------------------- page assembly


def render(
    profiles: list[ColumnProfile],
    rows: pd.DataFrame,
    *,
    n_index: int,
    caption: str,
    max_height: str,
    max_chars: int,
    sparklines: bool,
) -> str:
    rid = f"rd-{uuid.uuid4().hex[:10]}"
    names, margins, sparks = [], [], []
    for i, prof in enumerate(profiles):
        idx = " rd-idx" if i < n_index else ""
        num = " rd-num" if prof.kind == NUMERIC or prof.dtype.startswith(("int", "Int", "uint", "UInt", "float", "Float")) else ""
        names.append(
            f'<th class="rd-col{idx}{num}" data-col="{i}" title="Click to sort">'
            f'<div class="rd-name">{_esc(prof.name)}</div><div class="rd-dtype">{_esc(prof.dtype)}</div></th>'
        )
        margins.append(f'<th class="rd-viz{idx}">{margin_html(prof)}</th>')
        if sparklines:
            sparks.append(f'<th class="rd-viz rd-spark-cell{idx}">{sparkline_svg(prof) if prof.sparkline else ""}</th>')

    head = f'<tr class="rd-names">{"".join(names)}</tr><tr class="rd-margin">{"".join(margins)}</tr>'
    if sparklines:
        head += f'<tr class="rd-sparks">{"".join(sparks)}</tr>'

    cols = [format_cells(rows.iloc[:, j], max_chars) for j in range(rows.shape[1])]
    body = []
    for r in range(len(rows)):
        cells = []
        for j, col in enumerate(cols):
            attrs, inner = col[r]
            tag = "th" if j < n_index else "td"
            if j < n_index:
                attrs = attrs.replace('class="', 'class="rd-idx ') if 'class="' in attrs else attrs + ' class="rd-idx"'
            cells.append(f"<{tag}{attrs}>{inner}</{tag}>")
        body.append(f"<tr>{''.join(cells)}</tr>")

    css = (_ASSETS / "rundown.css").read_text()
    js = (_ASSETS / "rundown.js").read_text().replace("__RUNDOWN_ID__", rid)
    return (
        f'<div class="rundown" id="{rid}" style="--rd-max-height:{_esc(max_height)}"><style>{css}</style>'
        f'<div class="rd-caption"><span class="rd-shape">{caption}</span>'
        f'<input class="rd-filter" type="search" placeholder="Filter rows\u2026" aria-label="Filter rows"></div>'
        f'<div class="rd-scroll"><table class="rd-table"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
        f"<script>{js}</script></div>"
    )
