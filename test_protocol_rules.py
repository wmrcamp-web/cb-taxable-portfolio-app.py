"""Offline regressions for the CB rules shared with the tax-exempt protocol."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import cb_taxable_portfolio_app as app


AS_OF = dt.date(2026, 11, 6)
SPECS = {
    "SHADOW": (0.001, [0.3, 0.1]),
    "HIGH_RETURN": (0.004, [0.59, 0.1]),
    "STRONG_LOADING": (0.002, [0.95, 0.1]),
    "BELOW": (0.008, [0.589999, -0.589999]),
    "NEGATIVE": (0.003, [0.1, -0.59]),
    "FILL": (0.0015, [0.1, 0.7]),
    "CAPPED": (0.001, [0.1, 0.8]),
}


@pytest.fixture
def inputs(monkeypatch):
    weeks = np.arange(119)
    spy = np.where(weeks % 2, -0.01 - 0.003 * np.sin(weeks), 0.015)
    prices = {"SPY": 100 * np.cumprod(1 + np.r_[0, spy])}
    for ticker, (drift, _) in SPECS.items():
        beta = np.where(weeks < 67, 1.8, 0.8)
        prices[ticker] = 100 * np.cumprod(1 + np.r_[0, beta * spy + drift])
    closes = pd.DataFrame(prices, index=pd.date_range(end=AS_OF, periods=120, freq="W-FRI"))
    caps = {t: (20 - i) * 1e9 for i, t in enumerate(SPECS)}
    seen = {"warnings": [], "fits": [], "caps": caps, "closes": closes}
    monkeypatch.setattr(app, "fetch_sp500_tickers", lambda: pd.DataFrame({
        "Symbol": list(SPECS), "GICS Sector": ["Technology"] * len(SPECS),
    }))
    monkeypatch.setattr(app, "get_sorted_by_market_cap", lambda tickers: (list(tickers), caps))

    def download(tickers, start, end):
        seen["download"] = (tickers, start, end)
        return closes.reindex(columns=tickers)

    class Analyzer:
        def __init__(self, n_factors, rotation, method):
            assert (n_factors, rotation, method) == (2, "varimax", "minres")

        def fit(self, matrix):
            seen["fits"].append(matrix.copy())
            self.loadings_ = np.array([SPECS[t][1] for t in matrix.columns])

    monkeypatch.setattr(app, "_download_close", download)
    monkeypatch.setattr(app, "FactorAnalyzer", Analyzer)
    monkeypatch.setattr(app.st, "warning", seen["warnings"].append)
    return seen


def run(gate=app.BENCHMARK_GATE, required=(), weights=None, target=6, universe=150):
    return app._run_pipeline(AS_OF, gate, universe, 2, 0.4, target, list(required), weights or {})


def test_default_gate_and_factor_analysis_use_same_52_week_window(inputs):
    benchmark, metrics, _, portfolio, _ = run()
    prices = inputs["closes"]
    expected = prices.iloc[-1] / prices.iloc[-53] - 1
    assert benchmark == pytest.approx(expected["SPY"])
    assert set(metrics.index) == set(expected.drop("SPY")[expected.drop("SPY") >= benchmark].index)
    pd.testing.assert_frame_equal(
        inputs["fits"][0], prices.iloc[-53:].pct_change(fill_method=None).iloc[1:][list(SPECS)]
    )
    assert inputs["download"][1:] == (AS_OF - dt.timedelta(days=742), AS_OF)
    assert inputs["download"][0][-1] == "SPY"
    assert "^GSPC" not in inputs["download"][0]
    assert portfolio.index.tolist() == [
        "SHADOW", "HIGH_RETURN", "NEGATIVE", "STRONG_LOADING", "FILL", "CAPPED",
    ]


def test_downside_beta_uses_104_returns_not_only_factor_window(inputs):
    _, metrics, _, _, _ = run(None)
    closes = inputs["closes"]
    returns = closes.iloc[-105:].pct_change(fill_method=None).iloc[1:]
    short = closes.iloc[-53:].pct_change(fill_method=None).iloc[1:]
    expected = app.calculate_downside_beta(returns["HIGH_RETURN"], returns["SPY"])
    assert metrics.loc["HIGH_RETURN", "Downside_Beta"] == pytest.approx(expected)
    assert expected != pytest.approx(
        app.calculate_downside_beta(short["HIGH_RETURN"], short["SPY"])
    )


def test_factor_matrix_matches_canonical_column_major_layout(inputs):
    returns = inputs["closes"].iloc[-53:].pct_change(fill_method=None).iloc[1:]
    matrix = app.validate_returns_matrix(returns[list(SPECS)])
    pd.testing.assert_frame_equal(matrix, returns[list(SPECS)])
    assert matrix.to_numpy().flags.f_contiguous


@pytest.mark.parametrize("down_weeks", [11, 12])
def test_downside_beta_requires_12_paired_spy_down_weeks(down_weeks):
    benchmark = pd.Series(np.linspace(-0.04, -0.01, down_weeks))
    result = app.calculate_downside_beta(benchmark * 0.7, benchmark)
    if down_weeks == 11:
        assert np.isnan(result)
    else:
        assert result == pytest.approx(0.7)
        asset = benchmark * 0.7
        asset.iloc[0] = np.nan
        assert np.isnan(app.calculate_downside_beta(asset, benchmark))


def test_custom_gate_off_and_retained_holding_allocation(inputs):
    _, _, _, portfolio, _ = run(10.0, ["SHADOW", "HIGH_RETURN"], {"SHADOW": 20, "HIGH_RETURN": 30})
    assert portfolio.index.tolist() == ["SHADOW", "HIGH_RETURN"]
    assert portfolio["Weight (%)"].tolist() == [20, 30]
    _, _, _, portfolio, _ = run(None, ["STRONG_LOADING"], {"STRONG_LOADING": 20})
    assert portfolio.index[0] == "STRONG_LOADING"
    assert portfolio.loc["STRONG_LOADING", "Weight (%)"] == 20
    assert portfolio.loc[~portfolio["Taxable_Holding"], "Weight (%)"].tolist() == [16.0] * 5
    assert portfolio["Dominant_Factor"].value_counts().max() <= 3
    assert "BELOW" not in portfolio.index


@pytest.mark.parametrize("raw", [None, 0, -1, float("nan"), float("inf"), "bad", True])
def test_unavailable_caps_warn_rank_last_and_do_not_drop(inputs, raw):
    inputs["caps"]["SHADOW"] = raw
    _, metrics, _, _, _ = run(None)
    assert metrics.loc["SHADOW", "Market_Cap"] == 0
    assert inputs["download"][0][-2:] == ["SHADOW", "SPY"]
    assert any("assigned zero and ranked last" in w for w in inputs["warnings"])


def test_all_missing_caps_keep_constituent_order(inputs):
    inputs["caps"].clear()
    _, metrics, _, _, _ = run(None)
    assert inputs["download"][0] == list(SPECS) + ["SPY"]
    assert (metrics["Market_Cap"] == 0).all()
    assert any("constituent order" in w for w in inputs["warnings"])


def test_fills_before_slicing_and_keeps_benchmark_failure_explicit(inputs):
    closes = inputs["closes"]
    closes.loc[closes.index[-53], "HIGH_RETURN"] = np.nan
    result = run()[1]
    filled = closes.ffill()
    assert result.loc["HIGH_RETURN", "T12M_Return"] == pytest.approx(
        filled["HIGH_RETURN"].iloc[-1] / filled["HIGH_RETURN"].iloc[-53] - 1
    )
    closes["SPY"] = np.nan
    with pytest.raises(RuntimeError, match="benchmark prices"):
        run()


def test_download_requests_adjusted_weekly_prices_with_exclusive_end(monkeypatch):
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            {("Close", "SPY"): [100.0, 110.0]},
            index=pd.date_range("2026-10-23", periods=2, freq="W-FRI"),
        )

    monkeypatch.setattr(app.yf, "download", download)
    result = app._download_close(["SPY"], AS_OF - dt.timedelta(days=742), AS_OF)
    assert seen["interval"] == "1wk" and seen["auto_adjust"] is True
    assert seen["end"] == AS_OF
    assert result["SPY"].tolist() == [100, 110]


def test_dashboard_defaults_to_in_run_spy_gate_without_separate_fetch(inputs, monkeypatch):
    monkeypatch.setattr(app, "get_benchmark_return",
                        lambda date: pytest.fail("No separate sidebar benchmark fetch"))
    monkeypatch.setattr(app, "_get_holdings_csv_default", lambda: "")
    result = run()
    calls = []

    def pipeline(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(app, "_run_pipeline", pipeline)
    page = AppTest.from_string("import cb_taxable_portfolio_app as app\napp.main()").run(timeout=20)
    assert not page.exception
    assert page.sidebar.radio[0].value == "SPY trailing return (default)"
    next(b for b in page.button if b.label == "Run protocol").click().run(timeout=20)
    assert not page.exception and not page.error
    assert calls[0][1] == app.BENCHMARK_GATE
    assert any(m.label == "S&P 500 (SPY) 52-week trailing return" for m in page.metric)
