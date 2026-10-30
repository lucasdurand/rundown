import re

import numpy as np
import pandas as pd
import pytest

import rundown
from rundown.profile import CATEGORICAL, DATETIME, NUMERIC, TIMEDELTA, sparkline_series
from rundown.render import compact, duration


@pytest.fixture
def mixed():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "x": rng.standard_normal(n),
            "small_int": rng.integers(0, 3, n),
            "big_int": rng.integers(0, 10_000, n),
            "text": rng.choice(["apple", "banana", "cherry", "date", "elder", "fig", "grape"], n),
            "cat": pd.Categorical(rng.choice(["a", "b"], n)),
            "flag": rng.random(n) > 0.5,
            "when": pd.date_range("2024-01-01", periods=n, freq="h"),
            "dur": pd.to_timedelta(rng.exponential(5, n), unit="s"),
            "nullable": pd.array(np.where(rng.random(n) > 0.7, None, rng.integers(0, 50, n)), dtype="Int64"),
        }
    )


def test_methods_patched(mixed):
    assert isinstance(mixed.rundown(), rundown.Rundown)
    assert isinstance(mixed["x"].rundown(), rundown.Rundown)
    assert "<table" in mixed.rundown()._repr_html_()


def test_kinds(mixed):
    kinds = {p.name: p.kind for p in mixed.rundown().profiles}
    assert kinds == {
        "x": NUMERIC,
        "small_int": CATEGORICAL,
        "big_int": NUMERIC,
        "text": CATEGORICAL,
        "cat": CATEGORICAL,
        "flag": CATEGORICAL,
        "when": DATETIME,
        "dur": TIMEDELTA,
        "nullable": NUMERIC,
    }


def test_margin_viz(mixed):
    r = mixed.rundown()
    html = r.html
    by_name = {p.name: p for p in r.profiles}
    assert by_name["x"].histogram is not None and by_name["x"].histogram[0].sum() == len(mixed)
    assert len(by_name["text"].top) == 5 and by_name["text"].other > 0
    assert "+2 more" in html
    assert html.count('class="rd-hist"') == 5  # x, big_int, when, dur, nullable
    assert by_name["nullable"].nulls == mixed["nullable"].isna().sum()
    assert "% null" in html


def test_rows_and_sampling(mixed):
    def shown(r):
        return len(re.findall(r"<tbody>.*</tbody>", r.html, flags=re.S)[0].split("<tr>")) - 1

    assert shown(mixed.rundown(5)) == 5
    assert shown(mixed.rundown(1000)) == len(mixed)
    head = mixed.rundown(3, sample="head").html
    assert "2024-01-01 00:00:00" in head and "2024-01-01 02:00:00" in head
    assert "2024-01-09 07:00:00" in mixed.rundown(1, sample="tail").html
    a, b = mixed.rundown(random_state=1).html, mixed.rundown(random_state=1).html
    assert re.sub(r"rd-[0-9a-f]{10}", "", a) == re.sub(r"rd-[0-9a-f]{10}", "", b)
    with pytest.raises(ValueError):
        mixed.rundown(sample="middle")


def test_sparklines(mixed):
    assert 'class="rd-spark"' not in mixed.rundown().html
    r = mixed.rundown(sparklines=True)
    assert r.html.count('class="rd-spark"') == 7  # every column except the two text/categorical ones
    assert r.html.count("rd-spark-cell") == mixed.shape[1]


def test_sparkline_downsampling():
    values = np.arange(10_000, dtype=float)
    sp = sparkline_series(values, 100)
    assert sp["bucketed"] and len(sp["mean"]) == 100
    assert sp["min"][0] == 0 and sp["max"][-1] == 9_999
    small = sparkline_series(np.array([1.0, np.nan, 3.0]), 100)
    assert not small["bucketed"] and np.isnan(small["mean"][1])
    assert sparkline_series(np.array([np.nan, np.nan]), 100) is None


def test_index_and_multiindex_columns():
    df = pd.DataFrame(np.ones((4, 2)), columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")]))
    df.index = pd.Index(list("wxyz"), name="key")
    r = df.rundown()
    assert [p.name for p in r.profiles] == ["key", "a / x", "a / y"]
    assert '<th class="rd-idx"' in r.html
    assert '<th class="rd-idx"' not in pd.DataFrame({"a": [1, 2]}).rundown().html


def test_escaping_and_truncation():
    df = pd.DataFrame({"<b>col</b>": ["<script>alert(1)</script>", "x" * 100]})
    html = df.rundown().html
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "&lt;b&gt;col&lt;/b&gt;" in html
    assert "x" * 39 + "\u2026" in html and f'title="{"x" * 100}"' in html


def test_awkward_values():
    df = pd.DataFrame(
        {
            "inf": [1.0, np.inf, np.nan, 2.0],
            "lists": [[1], [2], None, [1]],
            "tz": pd.date_range("2024-01-01", periods=4, tz="America/Toronto"),
            "allnull": [np.nan] * 4,
            "dup": 1,
        }
    )
    df["dup2"] = 2
    df.columns = ["inf", "lists", "tz", "allnull", "dup", "dup"]
    r = df.rundown(sparklines=True)
    assert "1 inf" in r.html
    assert "America/Toronto" in r.html
    pd.DataFrame({"empty": pd.Series([], dtype=float)}).rundown()


def test_large_frame_profile_sampling():
    df = pd.DataFrame({"x": np.arange(50_000)})
    r = df.rundown(max_profile_rows=10_000, sparklines=True)
    assert "10,000-row sample" in r.html
    assert r.profiles[0].count == 10_000
    assert r.profiles[0].sparkline["n"] == 50_000


def test_max_columns():
    df = pd.DataFrame(np.zeros((3, 80)))
    r = df.rundown(max_columns=10)
    assert len(r.profiles) == 10 and "first 10 columns" in r.html


def test_save(tmp_path, mixed):
    path = mixed.rundown().save(tmp_path / "out.html")
    assert path.read_text().startswith("<!doctype html>")


def test_formatting():
    assert compact(0) == "0"
    assert compact(29.6991) == "29.7"
    assert compact(12_345) == "12.3k"
    assert compact(4_560_000) == "4.56M"
    assert duration(4) == "4ns"
    assert duration(3_430_000) == "3.43ms"
    assert duration(90e9) == "1.5min"
    html = pd.DataFrame({"f": [40.0, 34.5], "g": [27.7208, 7.25], "i": [1000, 2]}).rundown(sample="head").html
    assert ">40.0<" in html and ">34.5<" in html
    assert ">27.72<" in html and ">7.25<" in html
    assert ">1,000<" in html


def test_polars():
    pl = pytest.importorskip("polars")
    import datetime as dt

    n = 50
    df = pl.DataFrame(
        {
            "x": np.linspace(0, 1, n),
            "i": pl.Series([None if k % 7 == 0 else k for k in range(n)], dtype=pl.Int64),
            "s": [["a", "b", "c"][k % 3] for k in range(n)],
            "c": pl.Series([["u", "v"][k % 2] for k in range(n)], dtype=pl.Categorical),
            "b": pl.Series([None if k % 5 == 0 else k % 2 == 0 for k in range(n)], dtype=pl.Boolean),
            "t": pl.datetime_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1) + dt.timedelta(hours=n - 1),
                                   "1h", time_zone="America/Toronto", eager=True),
            "d": pl.Series([dt.timedelta(seconds=k) for k in range(n)]),
            "l": [[k, k + 1] for k in range(n)],
        }
    )
    r = df.rundown(sparklines=True, random_state=0)
    assert r.shape == (n, 8)
    assert "polars DataFrame" in r.html
    kinds = {p.name: p.kind for p in r.profiles}
    assert kinds["x"] == NUMERIC and kinds["s"] == CATEGORICAL and kinds["c"] == CATEGORICAL
    assert kinds["t"] == DATETIME and kinds["d"] == TIMEDELTA
    nulls = {p.name: p.nulls for p in r.profiles}
    assert nulls["i"] == 8 and nulls["b"] == 10
    dtypes = {p.name: p.dtype for p in r.profiles}
    assert dtypes["x"] == "f64" and dtypes["i"] == "i64" and dtypes["s"] == "str"
    assert "America/Toronto" in dtypes["t"]
    assert r.profiles[[p.name for p in r.profiles].index("t")].stats["tz"] is not None
    # integers with nulls stay integers in the rows (no "3.0")
    ih = pl.DataFrame({"i": pl.Series([1, None, 3])}).rundown(sample="head").html.split("<tbody")[1]
    assert ">3<" in ih and ">3.0<" not in ih
    assert df.shape == (n, 8)  # input untouched
    assert pl.Series("v", [1, 2, 3]).rundown().shape == (3, 1)
    assert "polars Series" in rundown.rundown(pl.Series([1.0, None])).html
    assert rundown.rundown(df, max_columns=3).html.count('class="rd-col') == 3
    assert "first 3 columns" in rundown.rundown(df, max_columns=3).html
    with pytest.raises(TypeError, match="collect"):
        rundown.rundown(df.lazy())


def test_unsupported_type():
    with pytest.raises(TypeError):
        rundown.rundown([1, 2, 3])


def test_to_dash(mixed):
    pytest.importorskip("dash")
    frame = mixed.rundown().to_dash(id="rd", style={"height": "300px"})
    assert type(frame).__name__ == "Iframe" and frame.id == "rd"
    assert frame.style["height"] == "300px" and frame.style["width"] == "100%"
    assert frame.srcDoc.startswith("<!doctype html>") and "frameElement" in frame.srcDoc
    assert 'class="rundown' in frame.srcDoc
    assert frame.srcDoc.index('class="rundown') < frame.srcDoc.index("ResizeObserver")
