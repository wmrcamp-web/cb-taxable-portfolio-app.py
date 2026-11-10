from __future__ import annotations

import datetime as dt
import io
import logging
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
BENCHMARK = "SPY"
BENCHMARK_GATE = "spy_t12m"
T12M_WEEKLY_CLOSES = 53
DOWNSIDE_BETA_WEEKS = 104
PRICE_HISTORY_DAYS = (DOWNSIDE_BETA_WEEKS + 2) * 7
NON_SHADOW_MIN_LOADING = 0.59
logger = logging.getLogger(__name__)
DEFAULT_HOLDINGS_CSV = "TICKER,ALLOCATION_PCT,GAIN_PCT,TERM\nAAPL,15,25,LT"
BANNED_TICKERS = {
    "XLE", "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO",
    "GLD", "IAU", "NEM", "GOLD", "AEM",
    "BITO", "IBIT", "GBTC", "MSTR", "COIN", "GOOG",
}

HOW_IT_WORKS_MD = """
## How It Works

The CB Taxable Portfolio Engine applies a multi-objective optimization framework to construct diversified, tax-aware equity portfolios from S&P 500 constituents.

**Universe Selection & Return Gating**

The screening process begins with the current S&P 500 list and market caps, even for an earlier as-of date. It excludes Energy-sector companies and specified oil, gold, cryptocurrency, and duplicate-share tickers. Missing or invalid market caps are assigned zero, warned about, and ranked last; if all are unavailable, constituent order determines the universe. The default return gate compares stocks with SPY over the same 52 weekly returns (53 weekly closes). You can choose a custom threshold or turn the gate off. Missing SPY data stops the run, rather than silently substituting a zero threshold.

**Factor-Based Diversification**

The engine downloads weekly prices and performs factor analysis on the last 52 weekly returns using Varimax rotation. Non-shadow candidates require an absolute loading of at least 0.59 on the factor being filled and are prioritized by highest trailing return, then loading, downside beta, and ticker to break ties. New selections respect a limit of three holdings per dominant factor; retained holdings count toward that limit but are not sold automatically if they already exceed it.

**Downside Risk & Defensive Positioning**

The engine calculates downside beta against SPY using up to 104 weekly returns, requiring at least 12 paired weeks when SPY fell. Stocks below the configured shadow-list loading cutoff receive first priority, ordered by lowest downside beta. Missing historical data may make a stock ineligible; historical risk measures are not predictions.

**Tax-Aware Allocation**

Existing taxable holdings retain their current weights and bypass the return gate. They count toward the target size; new picks split the remaining allocation equally. Estimated tax drag is displayed using allocation, unrealized gains, and fixed tax rates (23.9% long-term, 40% short-term); it is not used to optimize stock selection or recommend sales.

The result is a diversified portfolio balanced across return quality, systematic risk concentration, downside resilience, and tax efficiency.
"""

HOW_IT_WORKS_DISCLAIMER_HTML = """
<div style="border: 2px solid #b00020; border-radius: 10px; padding: 14px 16px; background: rgba(176,0,32,0.05); color: #b00020; font-weight: 700; line-height: 1.5; margin-top: 1.5rem;">
<p>IMPORTANT: RESEARCH TOOL ONLY—NOT INVESTMENT ADVICE. THE CB TAX-EXEMPT PROTOCOL, ITS ANALYSIS, PORTFOLIO RESULTS, AND RELATED MATERIALS ARE PROVIDED FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. THEY ARE NOT INVESTMENT, FINANCIAL, TAX, OR LEGAL ADVICE, AND ARE NOT A RECOMMENDATION OR SOLICITATION TO BUY, SELL, OR HOLD ANY SECURITY. INVESTING INVOLVES RISK, INCLUDING THE POSSIBLE LOSS OF PRINCIPAL. PAST PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS. CONSULT A QUALIFIED FINANCIAL PROFESSIONAL WHO UNDERSTANDS YOUR CIRCUMSTANCES BEFORE MAKING ANY INVESTMENT DECISION.</p>
<p style="margin-bottom: 0;">AI-GENERATED CONTENT AND LIMITATIONS. THE APP AND ITS CODE WERE DEVELOPED WITH THE ASSISTANCE OF ARTIFICIAL INTELLIGENCE AND PROVIDED IN GOOD FAITH. THEY MAY CONTAIN ERRORS, OMISSIONS, INACCURATE OR FABRICATED (“HALLUCINATED”) INFORMATION, OR OTHER DEFECTS. DATA, CALCULATIONS, AND RESULTS MAY BE INCOMPLETE OR INCORRECT AND HAVE NOT BEEN INDEPENDENTLY VERIFIED. REVIEW AND VERIFY INFORMATION BEFORE RELYING ON IT FOR ANY PURPOSE.</p>
</div>
"""


def _get_holdings_csv_default() -> str:
    try:
        configured = st.secrets.get("HOLDINGS_CSV")
    except FileNotFoundError:
        configured = None
    return configured if isinstance(configured, str) and configured.strip() else DEFAULT_HOLDINGS_CSV


def _download_close(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    downloaded = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1wk",
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
    prices = _download_close(
        [BENCHMARK], as_of_date - dt.timedelta(days=PRICE_HISTORY_DAYS), as_of_date
    )
    values = prices[BENCHMARK].ffill().iloc[-T12M_WEEKLY_CLOSES:].dropna()
    if len(values) < 2:
        raise RuntimeError("Not enough S&P 500 price history for the selected date.")
    result = float(values.iloc[-1] / values.iloc[0] - 1.0)
    if not np.isfinite(result):
        raise RuntimeError("SPY trailing return is not finite.")
    return result


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


def _positive_market_cap(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return 0.0
    return cap if np.isfinite(cap) and cap > 0 else 0.0


def _get_market_cap(ticker: str) -> tuple[str, float]:
    try:
        fast_info: Any = yf.Ticker(ticker).fast_info
        market_cap = fast_info.get("market_cap") or fast_info.get("marketCap") or 0
        cap = _positive_market_cap(market_cap)
        if not cap:
            logger.warning("Market cap unavailable for %s; assigned zero and ranked last", ticker)
        return ticker, cap
    except Exception as error:
        logger.warning("Market cap unavailable for %s (%s); assigned zero and ranked last",
                       ticker, type(error).__name__)
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
    if len(downside) < 12 or not np.isfinite(variance) or variance == 0:
        return float("nan")
    return float(downside["asset"].cov(downside["benchmark"]) / variance)


def validate_returns_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    clean = clean.loc[:, clean.nunique(dropna=True) > 1]
    if clean.empty or clean.shape[0] < 3 or clean.shape[1] < 2:
        raise ValueError("At least two non-constant tickers and three complete return weeks are required.")
    # Match the tax-exempt engine's column-major input to avoid numerical solver drift.
    return pd.DataFrame(
        np.asfortranarray(clean.to_numpy(dtype=float)),
        index=clean.index, columns=clean.columns, copy=False,
    )


def _run_pipeline(
    as_of_date: dt.date,
    gate_threshold: float | str | None,
    universe_size: int,
    n_factors: int,
    shadow_cutoff: float,
    target_size: int,
    must_go_in: list[str],
    held_weights: dict[str, float],
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(gate_threshold, str):
        if gate_threshold != BENCHMARK_GATE:
            raise ValueError(f"Unknown performance gate {gate_threshold!r}")
    elif gate_threshold is not None and not np.isfinite(gate_threshold):
        raise ValueError("Performance gate threshold must be finite.")
    constituents = fetch_sp500_tickers()
    constituents = constituents.loc[constituents["GICS Sector"] != "Energy"].copy()
    sector_by_ticker = dict(zip(constituents["Symbol"], constituents["GICS Sector"]))
    eligible = [ticker for ticker in sector_by_ticker if ticker not in BANNED_TICKERS]
    required = list(dict.fromkeys(must_go_in))

    with st.spinner("Checking S&P 500 market capitalizations..."):
        _, fetched_caps = get_sorted_by_market_cap(tuple(eligible))
    market_caps = {t: _positive_market_cap(fetched_caps.get(t)) for t in eligible}
    sorted_tickers = sorted(eligible, key=lambda ticker: market_caps[ticker], reverse=True)
    no_cap = sorted(t for t in eligible if not market_caps.get(t))
    if no_cap:
        st.warning(
            f"Market caps unavailable for {len(no_cap)} tickers; assigned zero and ranked last: "
            f"{', '.join(no_cap)}. Market-cap ranking may be incomplete; "
            "if all caps are unavailable, constituent order determines the universe."
        )
    selected = sorted_tickers if universe_size <= 0 else sorted_tickers[:universe_size]
    tickers = list(dict.fromkeys(required + selected + [BENCHMARK]))
    with st.spinner("Downloading historical prices..."):
        history = _download_close(
            tickers, as_of_date - dt.timedelta(days=PRICE_HISTORY_DAYS), as_of_date
        ).ffill()
    prices = history.iloc[-T12M_WEEKLY_CLOSES:].dropna(axis=1, how="any")
    dropped = sorted(set(tickers) - set(prices.columns))
    if dropped:
        st.warning(
            f"{len(dropped)} of {len(tickers)} tickers were dropped for missing or incomplete "
            f"price history: {', '.join(dropped)}"
        )
    if BENCHMARK not in prices or len(prices) < 2:
        raise RuntimeError("Yahoo Finance did not return S&P 500 benchmark prices.")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise RuntimeError("Not enough price history to calculate returns.")

    total_returns = prices.iloc[-1].div(prices.iloc[0]).sub(1.0)
    benchmark_return = float(total_returns[BENCHMARK])
    if not np.isfinite(benchmark_return):
        raise RuntimeError("SPY trailing return is not finite.")
    if gate_threshold == BENCHMARK_GATE:
        gate_threshold = benchmark_return
    downside_returns = history.iloc[-(DOWNSIDE_BETA_WEEKS + 1):].pct_change(
        fill_method=None
    ).dropna(how="all")
    benchmark_weekly = downside_returns[BENCHMARK]
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
    with st.spinner(f"Running {factor_count}-factor analysis..."):
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
        ticker: calculate_downside_beta(downside_returns[ticker], benchmark_weekly)
        for ticker in survivors
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
    held_weights = {t: w for t, w in held_weights.items() if t in required}
    held_in_screen = [t for t in required if t in metrics.index]
    limit = len(held_in_screen) + max(target_size - len(required), 0)
    for ticker in held_in_screen:
        if len(chosen) < limit:
            chosen.append(ticker)
            factor = metrics.loc[ticker, "Dominant_Factor"]
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    def add_candidate(ticker: str) -> None:
        factor = metrics.loc[ticker, "Dominant_Factor"]
        if ticker not in chosen and factor_counts.get(factor, 0) < 3 and len(chosen) < limit:
            chosen.append(ticker)
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

    for ticker in shadow:
        add_candidate(ticker)
    absolute_loadings = loadings.abs()
    non_shadow = metrics.loc[~metrics.index.isin(shadow)]
    ranked_non_shadow = non_shadow.assign(_selection_ticker=non_shadow.index).sort_values(
        by=["T12M_Return", "Max_Factor_Loading", "Downside_Beta", "_selection_ticker"],
        ascending=[False, False, True, True], na_position="last",
    )
    for factor in loadings.columns:
        if len(chosen) >= limit:
            break
        for ticker in ranked_non_shadow.index:
            if absolute_loadings.loc[ticker, factor] >= NON_SHADOW_MIN_LOADING:
                before = len(chosen)
                add_candidate(ticker)
                if len(chosen) > before:
                    break
    if len(chosen) < limit:
        for ticker in shadow + ranked_non_shadow.index.tolist():
            if (ticker in shadow
                    or absolute_loadings.loc[ticker].ge(NON_SHADOW_MIN_LOADING).any()):
                add_candidate(ticker)

    portfolio = metrics.reindex(list(dict.fromkeys(required + chosen))).copy() if (chosen or required) else pd.DataFrame()
    if not portfolio.empty:
        new_picks = [t for t in portfolio.index if t not in held_weights]
        remaining = max(100.0 - sum(held_weights.values()), 0.0)
        portfolio["Weight (%)"] = [
            held_weights[t] if t in held_weights else round(remaining / len(new_picks), 2)
            for t in portfolio.index
        ]
        portfolio["In_Shadow_List"] = portfolio.index.isin(shadow)
        portfolio["Taxable_Holding"] = portfolio.index.isin(held_weights)
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
    holdings["TICKER"] = holdings["TICKER"].astype("string").str.strip().str.upper().str.replace(".", "-", regex=False)
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


def _show_how_it_works() -> None:
    st.session_state.show_how_it_works = True


def _show_app() -> None:
    st.session_state.show_how_it_works = False


def main() -> None:
    st.set_page_config(
        page_title="CB Taxable Portfolio Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] textarea {
            font-size: 0.75rem;
        }
        [data-testid="stSidebar"] div.stButton > button p {
            font-size: 1.15rem;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "show_how_it_works" not in st.session_state:
        st.session_state.show_how_it_works = False

    st.title("CB Taxable Portfolio Engine")
    st.markdown(
        """
        <div style="border: 2px solid #b00020; border-radius: 10px; padding: 14px 16px; background: rgba(176,0,32,0.05); color: #b00020; font-weight: 700; line-height: 1.5; margin-top: 0.25rem; margin-bottom: 0.75rem;">
            <strong>IMPORTANT: RESEARCH TOOL ONLY — NOT INVESTMENT ADVICE.</strong><br>
            Provided for educational and research purposes only; not a recommendation or solicitation to buy, sell, or hold any security.<br>
            Investing involves risk, including possible loss of principal; past performance does not guarantee future results.<br>
            Content was developed with AI assistance and may contain errors — verify before relying on it, and consult a qualified financial professional before investing.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("S&P 500 screening, return gates, downside beta, factor diversification, and estimated tax drag.")
    sidebar = st.sidebar

    # Navigation buttons: callbacks update state before the rerun, so one click always works.
    nav1, nav2 = sidebar.columns(2)
    nav1.button("📘 How It Works", use_container_width=True, key="how_it_works_btn", on_click=_show_how_it_works)
    nav2.button("🏠 APP", use_container_width=True, key="back_to_app_btn", on_click=_show_app)

    if st.session_state.show_how_it_works:
        st.markdown("---")
        st.markdown(HOW_IT_WORKS_MD)
        st.markdown(HOW_IT_WORKS_DISCLAIMER_HTML, unsafe_allow_html=True)
        return

    sidebar.header("Protocol configuration")
    as_of_date = sidebar.date_input("As-of date", value=dt.date.today())
    enforce_gate = sidebar.toggle("Enforce minimum trailing return", value=True)
    if enforce_gate:
        gate_source = sidebar.radio(
            "Gate threshold source", ["SPY trailing return (default)", "Custom threshold"]
        )
        if gate_source == "SPY trailing return (default)":
            threshold = BENCHMARK_GATE
            sidebar.caption("SPY's return over the same 52 weekly returns as each stock.")
        else:
            threshold = sidebar.number_input(
                "Minimum trailing return (%)", value=0.0, step=1.0, format="%.2f"
            ) / 100.0
    else:
        threshold = None
    full_universe = sidebar.checkbox(
        "Screen the full S&P 500", value=False, help="Slower: downloads prices for every eligible constituent."
    )
    universe_size = sidebar.number_input(
        "Top S&P 500 by market cap", min_value=50, max_value=250, value=150, step=5,
        disabled=full_universe,
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
        value=_get_holdings_csv_default(),
        height=170,
        help="Keep the header row. Update the preloaded holdings as needed, or clear the box to skip taxable holdings.",
    )
    try:
        holdings, must_go_in = _parse_holdings(holdings_csv)
        if not holdings.empty:
            tax_a, tax_b = sidebar.columns(2)
            tax_a.metric("Allocated", f"{holdings['ALLOCATION_PCT'].sum():.1f}%")
            tax_b.metric("Estimated tax drag", f"{holdings['TAX_DRAG'].sum():.2%}")
    except (ValueError, pd.errors.ParserError) as error:
        sidebar.error(str(error))
        holdings, must_go_in = pd.DataFrame(), []

    with sidebar.expander("Selection constraints"):
        st.markdown(
            "- Excludes energy, gold, bitcoin-related tickers, and GOOG\n"
            "- Taxable holdings keep their allocation and count toward the target size; new picks split the remaining % equally\n"
            "- Maximum three holdings per dominant factor\n"
            "- Non-shadow picks require absolute loading >= 0.59; highest trailing return first\n"
            "- Taxable holdings are prioritized when eligible"
        )
    st.caption(
        "Tax drag is an estimate using fixed 23.9% long-term and 40% short-term rates. "
        "It excludes state taxes and individual circumstances; it is not tax advice."
    )
    run = st.button("Run protocol", type="primary")
    if not run:
        st.info("Set the screening parameters, then run the protocol to calculate a portfolio.")
        return

    try:
        benchmark, metrics, loadings, portfolio, survivors = _run_pipeline(
            as_of_date, threshold, 0 if full_universe else int(universe_size), int(n_factors),
            float(shadow_cutoff), int(target_size), must_go_in,
            dict(zip(holdings["TICKER"], holdings["ALLOCATION_PCT"])) if not holdings.empty else {},
        )
    except Exception as error:
        st.error(f"Screening failed: {error}")
        st.info("Check the network connection and retry. Yahoo Finance or Wikipedia may rate-limit requests.")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("S&P 500 (SPY) 52-week trailing return", f"{benchmark:.2%}")
    col_b.metric("Screen survivors", len(metrics))
    col_c.metric("Portfolio holdings", len(portfolio))
    if portfolio.empty:
        st.warning("No portfolio was generated. Try relaxing the return gate or changing the date.")
        return
    st.subheader(f"Portfolio allocation as of {as_of_date}")
    st.dataframe(
        portfolio[
            ["Weight (%)", "Taxable_Holding", "In_Shadow_List", "Dominant_Factor", "Downside_Beta",
             "Max_Factor_Loading", "Sector", "T12M_Return"]
        ].style.format(
            {
                "Weight (%)": "{:.2f}%", "Downside_Beta": "{:.3f}",
                "Max_Factor_Loading": "{:.3f}", "T12M_Return": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    if not holdings.empty:
        st.markdown("**Taxable holdings**")
        st.dataframe(
            holdings.assign(IN_PORTFOLIO=holdings["TICKER"].isin(portfolio.index)).style.format(
                {
                    "ALLOCATION_PCT": "{:.2f}%", "GAIN_PCT": "{:.2f}%",
                    "WEIGHTED_GAIN": "{:.2%}", "TAX_DRAG": "{:.2%}",
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
    if not holdings.empty:
        st.markdown("**Taxable holdings factor analysis**")
        held = holdings["TICKER"].tolist()
        held_loadings = loadings.reindex(held).dropna(how="all")
        if held_loadings.empty:
            st.caption("None of your taxable holdings survived the screen, so no factor loadings are available.")
        else:
            held_view = held_loadings.join(
                metrics.reindex(held_loadings.index)[["Dominant_Factor", "Max_Factor_Loading"]]
            )
            st.dataframe(
                held_view.style.format({"Max_Factor_Loading": "{:.3f}"}).background_gradient(
                    cmap="Blues", subset=list(held_loadings.columns)
                ),
                use_container_width=True,
            )
            missing = [t for t in held if t not in held_loadings.index]
            if missing:
                st.caption(f"Not in factor analysis (screened out): {', '.join(missing)}")
    st.subheader("Surviving stocks, ranked by market cap")
    st.dataframe(
        survivors.style.format(
            {"T12M_Return": "{:.2%}", "Market_Cap": "${:,.0f}", "Downside_Beta": "{:.3f}"}
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
