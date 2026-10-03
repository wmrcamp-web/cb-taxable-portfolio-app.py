from __future__ import annotations

import datetime as dt
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from factor_analyzer import FactorAnalyzer


LT_TAX_RATE = 0.239
ST_TAX_RATE = 0.40
BENCHMARK = "^GSPC"
BANNED_TICKERS = {
    "XLE", "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO",
    "GLD", "IAU", "NEM", "GOLD", "AEM",
    "BITO", "IBIT", "GBTC", "MSTR", "COIN", "GOOG",
}


def _download_close(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    downloaded = yf.download(
        tickers=tickers,
        start=start,
        end=end + dt.timedelta(days=1),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no price data. Please retry later.")
    if isinstance(downloaded.columns, pd.MultiIndex):
        close = downloaded.xs("Close", axis=1, level=0, drop_level=True)
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].copy()
        close.columns = tickers[:1]
    else:
        raise RuntimeError("Yahoo Finance response did not contain closing prices.")
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.columns = [str(column) for column in close.columns]
    return close.sort_index().apply(pd.to_numeric, errors="coerce")


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_return(as_of_date: dt.date) -> float:
    prices = _download_close([BENCHMARK], as_of_date - dt.timedelta(days=400), as_of_date)
    values = prices[BENCHMARK].dropna()
    if len(values) < 2:
        raise RuntimeError("Not enough S&P 500 price history for the selected date.")
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_tickers() -> pd.DataFrame:
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "CB-Taxable-Portfolio/1.0 (Streamlit app)"},
        timeout=20,
    )
    response.raise_for_status()
    for table in pd.read_html(io.StringIO(response.text)):
        if {"Symbol", "GICS Sector"}.issubset(table.columns):
            result = table[["Symbol", "GICS Sector"]].dropna().copy()
            result["Symbol"] = result["Symbol"].astype(str).str.replace(".", "-", regex=False)
            return result
    raise RuntimeError("Could not find the S&P 500 constituent table on Wikipedia.")


def _get_market_cap(ticker: str) -> tuple[str, float]:
    try:
        fast_info: Any = yf.Ticker(ticker).fast_info
        market_cap = fast_info.get("market_cap") or fast_info.get("marketCap") or 0
        return ticker, float(market_cap)
    except Exception:
        return ticker, 0.0


@st.cache_data(ttl=86400, show_spinner=False)
def get_sorted_by_market_cap(tickers: tuple[str, ...]) -> tuple[list[str], dict[str, float]]:
    caps: dict[str, float] = {}
    if not tickers:
        return [], caps
    with ThreadPoolExecutor(max_workers=min(12, len(tickers))) as executor:
        futures = [executor.submit(_get_market_cap, ticker) for ticker in tickers]
        for future in as_completed(futures):
            ticker, market_cap = future.result()
            caps[ticker] = market_cap
    return sorted(tickers, key=lambda ticker: caps.get(ticker, 0.0), reverse=True), caps


def calculate_downside_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat(
        [asset_returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    downside = aligned.loc[aligned["benchmark"] < 0]
    variance = downside["benchmark"].var(ddof=1)
    if len(downside) < 2 or not np.isfinite(variance) or variance == 0:
        return float("nan")
    return float(downside["asset"].cov(downside["benchmark"]) / variance)


def validate_returns_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    clean = clean.loc[:, clean.nunique(dropna=True) > 1]
    if clean.empty or clean.shape[0] < 3 or clean.shape[1] < 2:
        raise ValueError("At least two non-constant tickers and three complete return days are required.")
    return clean


def _run_pipeline(
    as_of_date: dt.date,
    gate_threshold: float | None,
    universe_size: int,
    n_factors: int,
    shadow_cutoff: float,
    target_size: int,
    must_go_in: list[str],
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    constituents = fetch_sp500_tickers()
    constituents = constituents.loc[constituents["GICS Sector"] != "Energy"].copy()
    sector_by_ticker = dict(zip(constituents["Symbol"], constituents["GICS Sector"]))
    eligible = [ticker for ticker in sector_by_ticker if ticker not in BANNED_TICKERS]
    required = list(dict.fromkeys(ticker for ticker in must_go_in if ticker in sector_by_ticker))

    with st.spinner("Checking S&P 500 market capitalizations..."):
        sorted_tickers, market_caps = get_sorted_by_market_cap(tuple(eligible))
    tickers = list(dict.fromkeys(required + sorted_tickers[:universe_size] + [BENCHMARK]))
    with st.spinner("Downloading historical prices..."):
        prices = _download_close(tickers, as_of_date - dt.timedelta(days=730), as_of_date)
    prices = prices.tail(253).dropna(axis=1, how="any")
    if BENCHMARK not in prices:
        raise RuntimeError("Yahoo Finance did not return S&P 500 benchmark prices.")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise RuntimeError("Not enough price history to calculate returns.")

    total_returns = prices.iloc[-1].div(prices.iloc[0]).sub(1.0)
    benchmark_return = float(total_returns[BENCHMARK])
    benchmark_daily = returns[BENCHMARK]
    stock_tickers = [ticker for ticker in prices.columns if ticker != BENCHMARK]
    survivors = [
        ticker for ticker in stock_tickers
        if ticker in required or gate_threshold is None or total_returns[ticker] >= gate_threshold
    ]
    if not survivors:
        empty = pd.DataFrame()
        return benchmark_return, empty, empty, empty, empty

    returns_matrix = validate_returns_matrix(returns[survivors])
    survivors = list(returns_matrix.columns)
    factor_count = min(n_factors, len(survivors), returns_matrix.shape[0] - 1)
    if factor_count < 2:
        raise RuntimeError("At least two qualifying stocks are needed for factor analysis.")
    with st.spinner(f"Running {factor_count}-factor Varimax analysis..."):
        analyzer = FactorAnalyzer(n_factors=factor_count, rotation="varimax", method="minres")
        analyzer.fit(returns_matrix)
        loadings = pd.DataFrame(
            analyzer.loadings_,
            index=survivors,
            columns=[f"Factor_{index + 1}" for index in range(factor_count)],
        )

    max_loading = loadings.abs().max(axis=1)
    dominant = loadings.abs().idxmax(axis=1)
    metrics = pd.DataFrame(index=survivors)
    metrics.index.name = "Ticker"
    metrics["T12M_Return"] = total_returns.reindex(survivors)
    metrics["Market_Cap"] = pd.Series(market_caps).reindex(survivors).fillna(0.0)
    metrics["Downside_Beta"] = {
        ticker: calculate_downside_beta(returns[ticker], benchmark_daily) for ticker in survivors
    }
    metrics["Max_Factor_Loading"] = max_loading
    metrics["Dominant_Factor"] = dominant
    metrics["Sector"] = pd.Series(sector_by_ticker).reindex(survivors).fillna("Unknown")
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["Downside_Beta"])
    loadings = loadings.reindex(metrics.index)
    if metrics.empty:
        return benchmark_return, metrics, loadings, pd.DataFrame(), pd.DataFrame()

    shadow = metrics.loc[metrics["Max_Factor_Loading"] < shadow_cutoff].sort_values(
        "Downside_Beta"
    ).index.tolist()
    factor_labels: dict[str, str] = {}
    for factor in loadings.columns:
        factor_tickers = metrics.index[metrics["Dominant_Factor"] == factor]
        sectors = metrics.loc[factor_tickers, "Sector"].value_counts()
        sector = sectors.index[0] if not sectors.empty else "Mixed"
        factor_labels[factor] = f"{factor}\n({sector})"
    loadings = loadings.rename(columns=factor_labels)
    metrics["Dominant_Factor"] = metrics["Dominant_Factor"].map(factor_labels)

    factor_counts = {label: 0 for label in factor_labels.values()}
    chosen: list[str] = []
    for ticker in required:
        if ticker in metrics.index and len(chosen) < target_size:
            chosen.append(ticker)
            factor = metrics.loc[ticker, "Dominant_Factor"]
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    def add_candidate(ticker: str) -> None:
        factor = metrics.loc[ticker, "Dominant_Factor"]
        if ticker not in chosen and factor_counts.get(factor, 0) < 3 and len(chosen) < target_size:
            chosen.append(ticker)
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    for ticker in shadow:
        add_candidate(ticker)
    absolute_loadings = loadings.abs()
    for factor in loadings.columns:
        if len(chosen) >= target_size:
            break
        for ticker in absolute_loadings[factor].sort_values(ascending=False).index:
            if absolute_loadings.loc[ticker, factor] >= 0.5:
                before = len(chosen)
                add_candidate(ticker)
                if len(chosen) > before:
                    break
    if len(chosen) < target_size:
        remaining = metrics.loc[~metrics.index.isin(chosen)].sort_values("Downside_Beta")
        for ticker in remaining.index:
            if ticker in shadow or metrics.loc[ticker, "Max_Factor_Loading"] >= 0.5:
                add_candidate(ticker)

    portfolio = metrics.loc[chosen].copy() if chosen else pd.DataFrame()
    if not portfolio.empty:
        portfolio["Weight (%)"] = round(100.0 / len(portfolio), 2)
        portfolio["In_Shadow_List"] = portfolio.index.isin(shadow)
    metrics = metrics.sort_values("Market_Cap", ascending=False)
    return benchmark_return, metrics, loadings, portfolio, metrics


def _join_split_header(csv_text: str, required: list[str]) -> str:
    # Pasted headers are sometimes wrapped onto several lines; rejoin them.
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    for count in range(1, min(4, len(lines)) + 1):
        fields = {field.strip().upper() for field in "".join(lines[:count]).split(",")}
        if set(required).issubset(fields):
            return "\n".join(["".join(lines[:count]), *lines[count:]])
    return "\n".join(lines)


def _parse_holdings(csv_text: str) -> tuple[pd.DataFrame, list[str]]:
    required = ["TICKER", "ALLOCATION_PCT", "GAIN_PCT", "TERM"]
    if not csv_text.strip():
        return pd.DataFrame(columns=required), []
    csv_text = _join_split_header(csv_text, required)
    holdings = pd.read_csv(io.StringIO(csv_text))
    holdings.columns = [str(column).strip().upper() for column in holdings.columns]
    if not set(required).issubset(holdings.columns):
        raise ValueError("CSV must include TICKER, ALLOCATION_PCT, GAIN_PCT, and TERM columns.")
    holdings = holdings[required].dropna(how="all").copy()
    holdings["TICKER"] = holdings["TICKER"].astype("string").str.strip().str.upper()
    holdings["TERM"] = holdings["TERM"].astype("string").str.strip().str.upper()
    for column in ("ALLOCATION_PCT", "GAIN_PCT"):
        holdings[column] = pd.to_numeric(holdings[column], errors="coerce")
    holdings = holdings.dropna(subset=required)
    valid = (
        holdings["TICKER"].ne("")
        & holdings["ALLOCATION_PCT"].between(0, 100)
        & holdings["GAIN_PCT"].ge(-100)
        & holdings["TERM"].isin(["LT", "ST"])
    )
    holdings = holdings.loc[valid].head(50).copy()
    holdings["WEIGHTED_GAIN"] = holdings["ALLOCATION_PCT"].div(100) * holdings["GAIN_PCT"].div(100)
    rates = holdings["TERM"].map({"LT": LT_TAX_RATE, "ST": ST_TAX_RATE})
    holdings["TAX_DRAG"] = holdings["WEIGHTED_GAIN"] * rates
    return holdings, holdings["TICKER"].tolist()


def main() -> None:
    st.set_page_config(
        page_title="CB Taxable Portfolio Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CB Taxable Portfolio Engine")
    st.caption("S&P 500 screening, return gates, downside beta, factor diversification, and estimated tax drag.")
    sidebar = st.sidebar
    sidebar.header("Protocol configuration")
    as_of_date = sidebar.date_input("As-of date", value=dt.date.today())
    try:
        benchmark_default = get_benchmark_return(as_of_date)
        sidebar.metric("S&P 500 trailing return", f"{benchmark_default:.2%}")
    except Exception as error:
        benchmark_default = None
        sidebar.warning(f"Benchmark data unavailable: {error}")

    enforce_gate = sidebar.toggle("Enforce minimum trailing return", value=True)
    if enforce_gate:
        default_threshold = benchmark_default * 100 if benchmark_default is not None else 0.0
        threshold = sidebar.number_input(
            "Minimum trailing return (%)", value=float(default_threshold), step=1.0, format="%.2f"
        ) / 100.0
    else:
        threshold = None
    universe_size = sidebar.number_input(
        "Top S&P 500 by market cap", min_value=50, max_value=250, value=150, step=5
    )
    n_factors = sidebar.number_input("Varimax factors", min_value=3, max_value=12, value=8)
    shadow_cutoff = sidebar.slider(
        "Shadow-list max loading cutoff", min_value=0.20, max_value=0.60, value=0.40, step=0.05
    )
    target_size = sidebar.number_input("Target portfolio size", min_value=10, max_value=30, value=16)
    sidebar.markdown("#### Taxable holdings")
    sidebar.caption(
        "One holding per line: ticker, % of portfolio, unrealized gain %, "
        "and LT (long-term) or ST (short-term). Up to 50 rows."
    )
    holdings_csv = sidebar.text_area(
        "Holdings (CSV)",
        value="TICKER,ALLOCATION_PCT,GAIN_PCT,TERM\nAAPL,15,25,LT",
        height=170,
        help="Keep the header row. Replace the AAPL example with your holdings, or clear the box to skip taxable holdings.",
    )
    try:
        holdings, must_go_in = _parse_holdings(holdings_csv)
        if not holdings.empty:
            tax_a, tax_b = sidebar.columns(2)
            tax_a.metric("Allocated", f"{holdings['ALLOCATION_PCT'].sum():.1f}%")
            tax_b.metric("Estimated tax drag", f"{holdings['TAX_DRAG'].sum():.2%}")
            st.subheader("Taxable holdings")
            st.dataframe(
                holdings.style.format(
                    {
                        "ALLOCATION_PCT": "{:.2f}%", "GAIN_PCT": "{:.2f}%",
                        "WEIGHTED_GAIN": "{:.2%}", "TAX_DRAG": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
    except (ValueError, pd.errors.ParserError) as error:
        sidebar.error(str(error))
        holdings, must_go_in = pd.DataFrame(), []

    with sidebar.expander("Selection constraints"):
        st.markdown(
            "- Excludes energy, gold, bitcoin-related tickers, and GOOG\n"
            "- Equal-weight allocation\n"
            "- Maximum three holdings per dominant factor\n"
            "- Taxable holdings are prioritized when eligible"
        )
    st.caption(
        "Tax drag is an estimate using fixed 23.9% long-term and 40% short-term rates. "
        "It excludes state taxes and individual circumstances; it is not tax advice."
    )
    run = sidebar.button("Run protocol screen", type="primary") or st.button("Run protocol", type="primary")
    if not run:
        st.info("Set the screening parameters, then run the protocol to calculate a portfolio.")
        return

    try:
        benchmark, metrics, loadings, portfolio, survivors = _run_pipeline(
            as_of_date, threshold, int(universe_size), int(n_factors),
            float(shadow_cutoff), int(target_size), must_go_in,
        )
    except Exception as error:
        st.error(f"Screening failed: {error}")
        st.info("Check the network connection and retry. Yahoo Finance or Wikipedia may rate-limit requests.")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("S&P 500 trailing return", f"{benchmark:.2%}")
    col_b.metric("Screen survivors", len(metrics))
    col_c.metric("Portfolio holdings", len(portfolio))
    if portfolio.empty:
        st.warning("No portfolio was generated. Try relaxing the return gate or changing the date.")
        return
    st.subheader(f"Portfolio allocation as of {as_of_date}")
    st.dataframe(
        portfolio[
            ["Weight (%)", "In_Shadow_List", "Dominant_Factor", "Downside_Beta",
             "Max_Factor_Loading", "Sector", "T12M_Return"]
        ].style.format(
            {
                "Weight (%)": "{:.2f}%", "Downside_Beta": "{:.3f}",
                "Max_Factor_Loading": "{:.3f}", "T12M_Return": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    st.metric("Portfolio average trailing return", f"{portfolio['T12M_Return'].mean():.2%}")
    st.download_button(
        "Download portfolio CSV", data=portfolio.to_csv().encode("utf-8"),
        file_name=f"taxable_portfolio_{as_of_date}.csv", mime="text/csv",
    )
    st.subheader("Varimax factor loadings")
    st.dataframe(loadings.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.subheader("Surviving stocks, ranked by market cap")
    st.dataframe(
        survivors.style.format(
            {"T12M_Return": "{:.2%}", "Market_Cap": "${:,.0f}", "Downside_Beta": "{:.3f}"}
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()