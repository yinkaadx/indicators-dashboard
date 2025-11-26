from __future__ import annotations

import os
import re
import json
import math
import time
import html
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import feedparser
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from fredapi import Fred
from io import StringIO

# =============================================================================
# SECRETS / API KEYS / GLOBAL SESSION
# =============================================================================

FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
ALPHAVANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
FMP_API_KEY = st.secrets.get("FMP_API_KEY", "")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "EconMirror/1.0 (+https://yinkaadx.streamlit.app)",
        "Accept": "text/html,application/json,application/xml;q=0.9,*/*;q=0.8",
    }
)

fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

DATA_DIR = "data"


# =============================================================================
# STREAMLIT PAGE CONFIG & STYLES
# =============================================================================

st.set_page_config(
    page_title="ECON MIRROR — Immortal Edition",
    layout="wide",
    page_icon="🌍",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 4.0rem !important;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 50%, #ff6a00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.0rem;
        color: #bbbbbb;
        margin-bottom: 1.5rem;
    }
    .regime-banner {
        background: #111111;
        color: #ffdddd;
        padding: 18px 24px;
        border-radius: 12px;
        border: 1px solid #442222;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.8rem 0 1.2rem 0;
    }
    .regime-banner span.red {
        color: #ff5555;
        font-weight: 800;
    }
    .regime-banner span.orange {
        color: #ffb347;
        font-weight: 700;
    }
    .regime-banner span.green {
        color: #7CFC00;
        font-weight: 700;
    }
    .kill-box {
        background: #1b0000;
        color: #ff6666;
        padding: 22px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 800;
        text-align: center;
        border: 2px solid #ff4444;
        margin-bottom: 1.0rem;
    }
    .metric-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin: 2px;
        border: 1px solid #333333;
        background: #111111;
    }
    .metric-pill span.label {
        color: #888888;
        margin-right: 4px;
    }
    .metric-pill span.value {
        color: #ffffff;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">ECON MIRROR</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Short-term kill combo + long-term super-cycle map — all live, all redundant, all mirrored.</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# SMALL UTILITIES
# =============================================================================


def to_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def pct_change(a: float, b: float) -> float:
    """Percent change (b - a)/a * 100, safe."""
    try:
        if a is None or math.isnan(a) or a == 0:
            return float("nan")
        return (b - a) / a * 100.0
    except Exception:
        return float("nan")


def safe_div(a: float, b: float) -> float:
    try:
        if b == 0 or math.isnan(b):
            return float("nan")
        return a / b
    except Exception:
        return float("nan")


def load_csv(path: str) -> pd.DataFrame:
    full = os.path.join(DATA_DIR, os.path.basename(path)) if not path.startswith(
        DATA_DIR
    ) else path
    if not os.path.exists(full):
        return pd.DataFrame()
    try:
        return pd.read_csv(full)
    except Exception:
        return pd.DataFrame()


def is_seed(path: str) -> bool:
    return "seed" in os.path.basename(path).lower()


# =============================================================================
# MIRROR HELPERS — BULLETPROOF CSV MIRRORS
# =============================================================================


def mirror_latest_csv(
    path: str,
    value_col: str,
    time_col: str,
    numeric_time: bool = False,
) -> Tuple[float, float, str, List[float]]:
    df = load_csv(path)
    if df.empty or value_col not in df.columns:
        return float("nan"), float("nan"), "—", []
    if numeric_time:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df[time_col] = pd.to_datetime(
            df[time_col], format="%Y-%m-%d", errors="coerce"
        )
    df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
    if df.empty:
        return float("nan"), float("nan"), "Mirror empty", []
    cur = to_float(df.iloc[-1][value_col])
    prev = to_float(df.iloc[-2][value_col]) if len(df) > 1 else float("nan")
    src = "Pinned seed" if is_seed(path) else "Mirror"
    hist = (
        pd.to_numeric(df[value_col], errors="coerce")
        .tail(24)
        .astype(float)
        .tolist()
    )
    return cur, prev, src, hist


# =============================================================================
# LIVE FETCHERS (OFFICIAL SOURCES + FMP/ALPHA VANTAGE + MIRRORS)
# =============================================================================


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_margin_debt_finra() -> Tuple[float, float, str]:
    """
    Margin debt in billions + margin%GDP (using FRED GDP as backup).
    """
    try:
        url = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            raise ValueError("No tables in FINRA html")
        df = tables[0]
        # Last row is latest
        val_raw = df.iloc[-1, 1]
        margin_bil = to_float(val_raw) / 1e3  # millions -> billions
        # Use FRED nominal GDP if BEA scraping fails
        gdp_tril = fetch_us_gdp_trillions()[0]
        if math.isnan(gdp_tril):
            gdp_tril = 28.0
        margin_pct_gdp = margin_bil / (gdp_tril * 1000) * 100.0
        return margin_bil, margin_pct_gdp, "FINRA direct"
    except Exception:
        try:
            cur, _, src, _ = mirror_latest_csv(
                os.path.join(DATA_DIR, "margin_finra.csv"),
                "debit_bil",
                "date",
                numeric_time=False,
            )
            # Mirror stores billions directly
            margin_bil = cur
            gdp_tril, _ = fetch_us_gdp_trillions()
            if math.isnan(gdp_tril):
                gdp_tril = 28.0
            margin_pct_gdp = margin_bil / (gdp_tril * 1000) * 100.0
            return margin_bil, margin_pct_gdp, src
        except Exception:
            return float("nan"), float("nan"), "Mirror failed"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_us_gdp_trillions() -> Tuple[float, str]:
    """
    Latest nominal US GDP (trillions USD).
    """
    # Try BEA via FRED 'GDP' (nominal quarterly).
    if fred:
        try:
            series = fred.get_series("GDP")
            latest = float(series.iloc[-1]) / 1000.0
            return round(latest, 2), "FRED GDP"
        except Exception:
            pass
    # Mirror fallback
    cur, _, src, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "us_gdp_nominal.csv"),
        "gdp_trillions",
        "date",
        numeric_time=False,
    )
    return float(cur), src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_real_fed_funds_rate() -> Tuple[float, str]:
    """
    Real Fed rate = FEDFUNDS latest - CPIAUCSL YoY%.
    """
    try:
        if not fred:
            raise ValueError("No FRED API")
        fed = fred.get_series("FEDFUNDS").iloc[-1]
        cpi = fred.get_series("CPIAUCSL").pct_change(12).iloc[-1] * 100.0
        return float(fed - cpi), "FRED FEDFUNDS - CPI YoY"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "real_fed_rate.csv"),
            "real_rate",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_put_call_cboe() -> Tuple[float, str]:
    """
    CBOE total put/call.
    """
    try:
        url = (
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv"
        )
        df = pd.read_csv(url, skiprows=2, nrows=1)
        val = to_float(df.iloc[0, 1])
        return float(val), "CBOE total put/call"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "cboe_total_pc.csv"),
            "total_pc",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_aaii_bulls() -> Tuple[float, str]:
    """
    AAII bullish %.
    """
    try:
        url = "https://www.aaii.com/files/surveys/sentiment.csv"
        df = pd.read_csv(url)
        last = df.iloc[-1]
        val = str(last["Bullish"]).replace("%", "").strip()
        return float(val), "AAII sentiment"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "aaii_bulls.csv"),
            "bulls_pct",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sp500_pe_multpl() -> Tuple[float, str]:
    """
    Current S&P 500 P/E from multpl.com (plus mirror).
    """
    try:
        html_text = SESSION.get(
            "https://www.multpl.com/s-p-500-pe-ratio", timeout=15
        ).text
        m = re.search(
            r"Current\s+S&P\s+500\s+PE\s+Ratio.*?([\d\.]+)",
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return float(m.group(1)), "multpl.com live"
    except Exception:
        pass
    cur, _, src, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "sp500_pe.csv"),
        "pe",
        "date",
        numeric_time=False,
    )
    return float(cur), src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_insider_ratio_openinsider() -> Tuple[float, str]:
    """
    Insider buy % = buys / (buys + sells) last ~couple of days, via openinsider.
    """
    try:
        url = "http://openinsider.com/latest-insider-trading"
        html_text = SESSION.get(url, timeout=20).text
        soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            raise ValueError("No insider table")
        rows = table.find_all("tr")
        buys = 0
        sells = 0
        for r in rows:
            tds = r.find_all("td")
            if len(tds) < 8:
                continue
            tx = tds[7].get_text(strip=True)
            if tx.startswith("P - Purchase"):
                buys += 1
            elif tx.startswith("S - Sale"):
                sells += 1
        total = buys + sells
        if total == 0:
            raise ValueError("No trades parsed")
        pct_buy = buys / total * 100.0
        return round(pct_buy, 1), "OpenInsider parsed"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "insider_buy_ratio.csv"),
            "buy_pct",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hy_spread() -> Tuple[float, str]:
    """
    HY spread (ICE BofA US High Yield Index Option-Adjusted Spread).
    """
    try:
        if not fred:
            raise ValueError("No FRED")
        series = fred.get_series("BAMLH0A0HYM2")
        latest = float(series.iloc[-1])
        return latest, "FRED BAMLH0A0HYM2"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "hy_spread.csv"),
            "spread_bps",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vix_alpha_vantage() -> Tuple[float, str]:
    """
    VIX from Alpha Vantage TIME_SERIES_DAILY (symbol=^VIX).
    """
    if not ALPHAVANTAGE_API_KEY:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "vix_index.csv"),
            "vix",
            "date",
            numeric_time=False,
        )
        return float(cur), src
    try:
        url = (
            "https://www.alphavantage.co/query?"
            "function=TIME_SERIES_DAILY"
            "&symbol=^VIX"
            f"&apikey={ALPHAVANTAGE_API_KEY}"
        )
        r = SESSION.get(url, timeout=20)
        data = r.json()
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            raise ValueError("No AV VIX time series")
        latest_date = sorted(ts.keys())[-1]
        close = to_float(ts[latest_date]["4. close"])
        return float(close), "Alpha Vantage ^VIX"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "vix_index.csv"),
            "vix",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_spx_from_fmp() -> Tuple[float, float, float, float, str]:
    """
    S&P 500 close, ATH, drawdown, YTD %, using FMP historical-price-full.
    """
    if not FMP_API_KEY:
        cur, _, src_close, hist = mirror_latest_csv(
            os.path.join(DATA_DIR, "spx_close.csv"),
            "close",
            "date",
            numeric_time=False,
        )
        # Mirror can't reliably give ATH/YTD, so we approximate.
        ath_val = max(hist) if hist else cur
        dd = pct_change(ath_val, cur)
        ytd = float("nan")
        return float(cur), float(ath_val), float(dd), float(ytd), src_close

    try:
        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/"
            f"^GSPC?timeseries=800&apikey={FMP_API_KEY}"
        )
        data = SESSION.get(url, timeout=20).json()
        hist = data.get("historical", [])
        if not hist:
            raise ValueError("No SPX history")
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.sort_values("date")
        df = df.dropna(subset=["date", "close"])
        close = float(df["close"].iloc[-1])
        ath = float(df["close"].max())
        dd = (close / ath - 1.0) * 100.0

        this_year = df[df["date"].dt.year == datetime.now().year]
        if this_year.empty:
            ytd = float("nan")
        else:
            first = float(this_year["close"].iloc[0])
            ytd = (close / first - 1.0) * 100.0
        return close, ath, dd, ytd, "FMP ^GSPC"
    except Exception:
        cur, _, src_close, hist = mirror_latest_csv(
            os.path.join(DATA_DIR, "spx_close.csv"),
            "close",
            "date",
            numeric_time=False,
        )
        ath_val = max(hist) if hist else cur
        dd = (cur / ath_val - 1.0) * 100.0 if ath_val else float("nan")
        return float(cur), float(ath_val), float(dd), float("nan"), src_close


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_spx_above_200ma() -> Tuple[float, str]:
    """
    % of S&P 500 members above 200d MA (SPXA200R), via StockCharts with mirror.
    """
    try:
        # Use public chart search to avoid script-blocking.
        url = "https://stockcharts.com/sc3/ui/?s=$SPXA200R"
        html_text = SESSION.get(url, timeout=20).text
        # Look for 'value' like "Value: 22.34"
        m = re.search(r"Value:\s*([\d\.]+)", html_text)
        if not m:
            # fallback pattern
            m = re.search(r"([\d\.]+)\s*%</text>", html_text)
        if m:
            return float(m.group(1)), "StockCharts SPXA200R"
        raise ValueError("No SPXA200R value found")
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "spx_above_200d.csv"),
            "pct",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fed_balance_yoy() -> Tuple[float, str]:
    """
    Fed balance sheet YoY % (WALCL).
    """
    try:
        if not fred:
            raise ValueError("No FRED")
        series = fred.get_series("WALCL")
        latest = float(series.iloc[-1])
        idx_year_ago = -53 if len(series) > 52 else 0
        year_ago = float(series.iloc[idx_year_ago])
        yoy = (latest / year_ago - 1.0) * 100.0 if year_ago != 0 else float(
            "nan"
        )
        return round(yoy, 2), "FRED WALCL YoY"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "fed_balance_yoy.csv"),
            "yoy_pct",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_sofr_spread() -> Tuple[float, str]:
    """
    SOFR - FEDFUNDS (bp equivalent, but we keep it in bp-ish as float).
    """
    try:
        # Live SOFR page
        url = "https://www.newyorkfed.org/markets/reference-rates/sofr"
        resp = SESSION.get(url, timeout=20)
        html_text = resp.text
        # Very crude parse of the first rate that looks like a number with %
        m = re.search(r"(\d+\.\d+)\s*percent", html_text, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+\.\d+)\s*%", html_text)
        sofr = float(m.group(1)) if m else float("nan")
        effr = float("nan")
        if fred:
            series = fred.get_series("FEDFUNDS")
            effr = float(series.iloc[-1])
        spread = sofr - effr if not math.isnan(sofr) and not math.isnan(effr) else float("nan")
        # We treat this as bp for thresholds, but store as raw %
        return spread * 100.0, "NY Fed SOFR - FEDFUNDS"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "sofr_spread.csv"),
            "spread_bp",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def cofer_usd_share_latest() -> Tuple[float, float, str, List[float]]:
    """
    USD reserve share from IMF COFER mirror.
    """
    path = os.path.join(DATA_DIR, "imf_cofer_usd_share.csv")
    return mirror_latest_csv(path, "usd_share", "date", numeric_time=False)


@st.cache_data(ttl=3600, show_spinner=False)
def sp500_pe_mirror_latest() -> Tuple[float, float, str, List[float]]:
    """
    P/E mirror (only used if live fails).
    """
    path = os.path.join(DATA_DIR, "sp500_pe.csv")
    return mirror_latest_csv(path, "pe", "date", numeric_time=False)


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_gpr_index() -> Tuple[float, str]:
    """
    Caldara & Iacoviello Geopolitical Risk Index: US_monthly.csv mirror.
    """
    try:
        url = "https://www.policyuncertainty.com/media/US_GPR.csv"
        df = pd.read_csv(url)
        df["DATE"] = pd.to_datetime(
            df["DATE"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["DATE", "GPR"])
        latest = float(df["GPR"].iloc[-1])
        return latest, "policyuncertainty.com GPR"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "us_gpr.csv"),
            "gpr",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gini_usa() -> Tuple[float, str]:
    """
    US Gini from FRED (SIPOVGINIUSA) + mirror.
    """
    try:
        if fred:
            series = fred.get_series("SIPOVGINIUSA")
            latest = float(series.iloc[-1])
            return latest, "FRED SIPOVGINIUSA"
    except Exception:
        pass
    cur, _, src, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "gini_usa.csv"),
        "gini",
        "date",
        numeric_time=False,
    )
    return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wage_share_usa() -> Tuple[float, str]:
    """
    US labour share of income from FRED (LABSHPUSA156NRUG).
    """
    try:
        if fred:
            series = fred.get_series("LABSHPUSA156NRUG")
            latest = float(series.iloc[-1])
            return latest, "FRED LABSHPUSA156NRUG"
    except Exception:
        pass
    cur, _, src, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "labour_share_usa.csv"),
        "wage_share",
        "date",
        numeric_time=False,
    )
    return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_productivity_latest() -> Tuple[float, str]:
    """
    Latest US productivity YoY/QoQ from BLS release (mirror fallback).
    """
    try:
        url = "https://www.bls.gov/news.release/prod2.nr0.htm"
        resp = SESSION.get(url, timeout=20)
        tables = pd.read_html(StringIO(resp.text))
        # Usually table 3 or similar contains nonfarm business productivity Q/Q
        table = tables[0]
        # Try to find a column that looks like 'Percent change, annual rate'
        # For safety, just take the last numeric in the table.
        vals = []
        for col in table.columns:
            for v in table[col]:
                try:
                    vals.append(float(str(v).replace("%", "")))
                except Exception:
                    continue
        if vals:
            latest = vals[-1]
            return float(latest), "BLS PROD2 parsed"
        raise ValueError("No numeric productivity values")
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "us_productivity.csv"),
            "prod_yoy",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gold_spot_fmp() -> Tuple[float, str]:
    """
    Gold spot (XAUUSD) in USD via FMP quote, mirrored.
    """
    if not FMP_API_KEY:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "gold_spot_usd.csv"),
            "price",
            "date",
            numeric_time=False,
        )
        return float(cur), src
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/XAUUSD?apikey={FMP_API_KEY}"
        data = SESSION.get(url, timeout=20).json()
        if not data:
            raise ValueError("No XAUUSD from FMP")
        price = float(data[0]["price"])
        return price, "FMP XAUUSD"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "gold_spot_usd.csv"),
            "price",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gold_all_ccy_fmp() -> Tuple[Dict[str, float], str]:
    """
    Gold vs major currencies via FMP quote for:
    XAUUSD, XAUEUR, XAUJPY, XAUGBP, XAUCHF, XAUCNY.
    """
    tickers = ["XAUUSD", "XAUEUR", "XAUJPY", "XAUGBP", "XAUCHF", "XAUCNY"]
    if not FMP_API_KEY:
        # Mirror fallback: store in gold_fx_basket.csv as columns.
        df = load_csv(os.path.join(DATA_DIR, "gold_fx_basket.csv"))
        if df.empty:
            return {}, "Mirror missing"
        df["date"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["date"])
        last = df.iloc[-1]
        out = {}
        for t in tickers:
            if t in last:
                out[t] = to_float(last[t])
        return out, "Mirror gold FX"
    try:
        url = (
            f"https://financialmodelingprep.com/api/v3/quote/"
            f"{','.join(tickers)}?apikey={FMP_API_KEY}"
        )
        data = SESSION.get(url, timeout=20).json()
        out = {}
        for row in data:
            sym = row.get("symbol")
            if sym in tickers:
                out[sym] = float(row.get("price", float("nan")))
        if not out:
            raise ValueError("No gold FX from FMP")
        return out, "FMP gold FX basket"
    except Exception:
        df = load_csv(os.path.join(DATA_DIR, "gold_fx_basket.csv"))
        if df.empty:
            return {}, "Mirror failed"
        df["date"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["date"])
        last = df.iloc[-1]
        out = {}
        for t in tickers:
            if t in last:
                out[t] = to_float(last[t])
        return out, "Mirror gold FX"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_tsy_yields_10_30() -> Tuple[float, float, str]:
    """
    10y and 30y yields from FRED (DGS10, DGS30) with mirror fallback.
    """
    ten = float("nan")
    thirty = float("nan")
    src = "Mirror only"
    if fred:
        try:
            ten = float(fred.get_series("DGS10").iloc[-1])
            thirty = float(fred.get_series("DGS30").iloc[-1])
            src = "FRED DGS10/DGS30"
        except Exception:
            pass
    if math.isnan(ten) or math.isnan(thirty):
        df = load_csv(os.path.join(DATA_DIR, "treasury_yields.csv"))
        if not df.empty:
            df["date"] = pd.to_datetime(
                df["date"], format="%Y-%m-%d", errors="coerce"
            )
            df = df.dropna(subset=["date"])
            ten = to_float(df["y10"].iloc[-1])
            thirty = to_float(df["y30"].iloc[-1])
            src = "Mirror treasury_yields.csv"
    return ten, thirty, src


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_oil_price_te() -> Tuple[float, str]:
    """
    Crude oil price via TradingEconomics HTML or mirror.
    """
    try:
        url = "https://tradingeconomics.com/commodity/crude-oil"
        resp = SESSION.get(url, timeout=20)
        html_text = resp.text
        m = re.search(
            r"\"Value\":\s*([\d\.]+)", html_text, re.IGNORECASE | re.DOTALL
        )
        if m:
            price = float(m.group(1))
            return price, "TradingEconomics crude oil"
        # fallback: parse table
        tables = pd.read_html(StringIO(html_text))
        if tables:
            df = tables[0]
            vals = []
            for col in df.columns:
                for v in df[col]:
                    try:
                        vals.append(float(str(v).replace(",", "")))
                    except Exception:
                        continue
            if vals:
                return vals[0], "TradingEconomics table"
        raise ValueError("No oil price from TE")
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "crude_oil_price.csv"),
            "price",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_btc_fmp() -> Tuple[float, str]:
    """
    BTCUSD via FMP quote.
    """
    if not FMP_API_KEY:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "btc_usd.csv"),
            "price",
            "date",
            numeric_time=False,
        )
        return float(cur), src
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/BTCUSD?apikey={FMP_API_KEY}"
        data = SESSION.get(url, timeout=20).json()
        if not data:
            raise ValueError("No BTCUSD from FMP")
        price = float(data[0]["price"])
        return price, "FMP BTCUSD"
    except Exception:
        cur, _, src, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "btc_usd.csv"),
            "price",
            "date",
            numeric_time=False,
        )
        return float(cur), src


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_farmland_index_mirror() -> Tuple[float, str]:
    """
    Farmland performance proxy from NCREIF or local mirror (index level).
    """
    try:
        df = load_csv(os.path.join(DATA_DIR, "farmland_index.csv"))
        if df.empty:
            raise ValueError("No farmland mirror CSV")
        df["date"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["date", "index"])
        latest = float(df["index"].iloc[-1])
        return latest, "Farmland mirror"
    except Exception:
        return float("nan"), "Farmland missing"


# =============================================================================
# LIVE REAL ASSETS BASKET INDEX
# =============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def live_real_assets_basket() -> Tuple[float, bool]:
    """
    Real assets basket of Gold + Oil + BTC + Farmland.
    Normalized index relative to pinned baseline.
    """
    gold, _ = fetch_gold_spot_fmp()
    oil, _ = fetch_oil_price_te()
    btc, _ = fetch_btc_fmp()
    farm, _ = fetch_farmland_index_mirror()

    components = [gold, oil, btc, farm]
    weights = [0.25, 0.25, 0.35, 0.15]
    val = 0.0
    for c, w in zip(components, weights):
        if math.isnan(c):
            continue
        val += w * c

    # baseline from mirror seed
    df = load_csv(os.path.join(DATA_DIR, "real_assets_seed.csv"))
    baseline = float("nan")
    if not df.empty and "index" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["date", "index"])
        if not df.empty:
            baseline = float(df["index"].iloc[0])
    if math.isnan(baseline) or baseline == 0:
        baseline = val if val != 0 else 1.0

    index = val / baseline * 100.0
    # Dark red if > 250 (i.e., has more than doubled vs baseline)
    dark = index >= 250.0
    return index, dark


# =============================================================================
# CENTRAL BANK GOLD + RESET NEWS SCANNER (RSS)
# =============================================================================


CB_GOLD_FEEDS = [
    "https://www.gold.org/rss",
    "https://news.gold.org/rss",
]

RESET_KEYWORDS = [
    "new Bretton Woods",
    "currency reset",
    "monetary reset",
    "new global currency",
    "CBDC",  # not reset alone, but sign
    "gold-backed",
]


@st.cache_data(ttl=3600, show_spinner=False)
def central_bank_gold_tonnage_increase_from_seed() -> Tuple[bool, str]:
    """
    Mirror-based: did CB gold tonnage rise sharply vs seed? (tonnes).
    """
    try:
        df = load_csv(os.path.join(DATA_DIR, "cb_gold_tonnes.csv"))
        if df.empty:
            return False, "No CB gold mirror"
        df["date"] = pd.to_datetime(
            df["date"], format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["date", "tonnes"])
        if len(df) < 2:
            return False, "Insufficient CB gold history"
        cur = float(df["tonnes"].iloc[-1])
        prev = float(df["tonnes"].iloc[0])
        pct = pct_change(prev, cur)
        return pct >= 30.0, f"CB gold tonnes change: {pct:.1f}%"
    except Exception:
        return False, "CB gold parse error"


@st.cache_data(ttl=1800, show_spinner=False)
def supercycle_news_alerts() -> Tuple[List[str], List[str]]:
    """
    Check RSS + mirror for central-bank gold buying & 'reset' talk.
    Returns (cb_gold_titles, reset_titles).
    """
    cb_titles: List[str] = []
    reset_titles: List[str] = []
    try:
        for feed_url in CB_GOLD_FEEDS:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:15]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = (title + " " + summary).lower()
                if "central bank" in text and "gold" in text:
                    cb_titles.append(title)
                if any(k.lower() in text for k in RESET_KEYWORDS):
                    reset_titles.append(title)
    except Exception:
        pass
    # Mirror fallback from csv (cb_gold_news.csv / reset_news.csv)
    if not cb_titles:
        df = load_csv(os.path.join(DATA_DIR, "cb_gold_news.csv"))
        if not df.empty and "title" in df.columns:
            cb_titles = df["title"].dropna().tail(10).tolist()
    if not reset_titles:
        df2 = load_csv(os.path.join(DATA_DIR, "reset_news.csv"))
        if not df2.empty and "title" in df2.columns:
            reset_titles = df2["title"].dropna().tail(10).tolist()
    return cb_titles, reset_titles


# =============================================================================
# OFFICIAL RESET EVENT TOGGLE (MANUAL)
# =============================================================================


def get_reset_flag_key() -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return f"official_reset_event_{today}"


def get_official_reset_flag() -> bool:
    return st.session_state.get(get_reset_flag_key(), False)


def set_official_reset_flag(value: bool) -> None:
    st.session_state[get_reset_flag_key()] = value


# =============================================================================
# MAIN SHORT-TERM & LONG-TERM COMBOS
# =============================================================================


def compute_short_term_kill_combo() -> Tuple[int, List[Dict[str, str]], bool]:
    """
    Build the 10/10 short-term kill combo list, return (kill_count, rows, near_ath).
    """
    margin_bil, margin_pct_gdp, src_margin = fetch_margin_debt_finra()
    real_fed, src_real = fetch_real_fed_funds_rate()
    put_call, src_pc = fetch_put_call_cboe()
    aaii_bulls, src_aaii = fetch_aaii_bulls()
    sp_pe, src_pe = fetch_sp500_pe_multpl()
    insider_buy, src_ins = fetch_insider_ratio_openinsider()
    hy_spread, src_hy = fetch_hy_spread()
    vix_val, src_vix = fetch_vix_alpha_vantage()
    sp_close, sp_ath, dd_pct, ytd_pct, src_spx = fetch_spx_from_fmp()
    spx_above_200, src_above = fetch_spx_above_200ma()
    fed_bs_yoy, src_walcl = fetch_fed_balance_yoy()
    sofr_spread_bp, src_sofr = fetch_sofr_spread()

    near_ath = dd_pct > -8.0  # within -8% of ATH
    kill_signals = []

    # 1. Margin Debt % GDP ≥3.5% & falling is a sign of unwinding (we track-level only here)
    kill_signals.append(
        {
            "Signal": "Margin Debt % of GDP ≥ 3.5%",
            "Value": f"{margin_pct_gdp:.2f}%",
            "Source": src_margin,
            "KILL": "KILL" if margin_pct_gdp >= 3.5 else "",
            "Why it matters": "High margin % of GDP means leverage (borrowing to buy stocks) is extreme; when this unwinds, crashes get violent.",
        }
    )

    # 2. Real Fed Rate ≥ +1.5%
    kill_signals.append(
        {
            "Signal": "Real Fed Funds ≥ +1.5%",
            "Value": f"{real_fed:.2f}%",
            "Source": src_real,
            "KILL": "KILL" if real_fed >= 1.5 else "",
            "Why it matters": "When real rates go positive and high, credit tightens and long bull runs usually end.",
        }
    )

    # 3. Put/Call < 0.65
    kill_signals.append(
        {
            "Signal": "Total Put/Call < 0.65",
            "Value": f"{put_call:.3f}",
            "Source": src_pc,
            "KILL": "KILL" if put_call < 0.65 else "",
            "Why it matters": "Very low put/call means traders are all-in call buying euphoria – contrarian danger zone.",
        }
    )

    # 4. AAII Bulls > 60%
    kill_signals.append(
        {
            "Signal": "AAII Bulls > 60%",
            "Value": f"{aaii_bulls:.1f}%",
            "Source": src_aaii,
            "KILL": "KILL" if aaii_bulls > 60.0 else "",
            "Why it matters": "When retail sentiment is extremely bullish, forward returns tend to be poor.",
        }
    )

    # 5. S&P P/E > 30
    kill_signals.append(
        {
            "Signal": "S&P 500 P/E > 30",
            "Value": f"{sp_pe:.2f}",
            "Source": src_pe,
            "KILL": "KILL" if sp_pe > 30.0 else "",
            "Why it matters": "A very high P/E means future earnings are priced in – little margin of safety.",
        }
    )

    # 6. Insider buying < 10%
    kill_signals.append(
        {
            "Signal": "Insider buying < 10% of trades",
            "Value": f"{insider_buy:.1f}%",
            "Source": src_ins,
            "KILL": "KILL" if insider_buy < 10.0 else "",
            "Why it matters": "When insiders mostly sell, they see stocks as expensive relative to reality.",
        }
    )

    # 7. HY < 400 bps but widening (we only check level here)
    kill_signals.append(
        {
            "Signal": "HY spread < 400 bps",
            "Value": f"{hy_spread:.1f} bps",
            "Source": src_hy,
            "KILL": "KILL" if hy_spread < 400.0 else "",
            "Why it matters": "Super tight credit spreads show no fear – typical before spreads blow out.",
        }
    )

    # 8. VIX < 20
    kill_signals.append(
        {
            "Signal": "VIX < 20",
            "Value": f"{vix_val:.2f}",
            "Source": src_vix,
            "KILL": "KILL" if vix_val < 20.0 else "",
            "Why it matters": "Low volatility often comes just before big breaks – complacency peak.",
        }
    )

    # 9. % S&P above 200d MA < 25%
    kill_signals.append(
        {
            "Signal": "% S&P 500 above 200d MA < 25%",
            "Value": f"{spx_above_200:.1f}%",
            "Source": src_above,
            "KILL": "KILL" if spx_above_200 < 25.0 else "",
            "Why it matters": "When only a small fraction of stocks are above trend, internals are rotten.",
        }
    )

    # 10. Fed BS YoY ≤ –5% OR SOFR spike
    fed_bs_trigger = fed_bs_yoy <= -5.0 or sofr_spread_bp > 50.0
    kill_signals.append(
        {
            "Signal": "Fed balance sheet YoY ≤ –5% OR SOFR spike",
            "Value": f"{fed_bs_yoy:.2f}% / {sofr_spread_bp:.1f} bp",
            "Source": f"{src_walcl} + {src_sofr}",
            "KILL": "KILL" if fed_bs_trigger else "",
            "Why it matters": "Aggressive QT or funding stress (SOFR spike) can break over-leveraged markets.",
        }
    )

    kill_count = sum(1 for row in kill_signals if row["KILL"] == "KILL")
    return kill_count, kill_signals, near_ath


def compute_long_term_supercycle() -> Tuple[int, int, Dict[str, Dict[str, str]]]:
    """
    Build the 11 dark-red long-term signals + 3 no-return triggers.
    Return (dark_red_count, no_return_count, info_dict)
    """
    # 1. Total Debt/GDP
    margin_bil, _, _ = fetch_margin_debt_finra()
    gdp_tril, _ = fetch_us_gdp_trillions()
    total_debt_gdp = float("nan")
    try:
        # mirror of total debt gdp
        cur, _, _, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "total_debt_gdp.csv"),
            "debt_gdp_pct",
            "date",
            numeric_time=False,
        )
        total_debt_gdp = float(cur)
    except Exception:
        pass
    total_debt_dark = total_debt_gdp >= 400.0

    # 2. Gold ATH vs major currencies
    gold_spot, _ = fetch_gold_spot_fmp()
    gold_fx_map, _ = fetch_gold_all_ccy_fmp()
    gold_all_ath = False
    try:
        df = load_csv(os.path.join(DATA_DIR, "gold_fx_history.csv"))
        if not df.empty:
            df["date"] = pd.to_datetime(
                df["date"], format="%Y-%m-%d", errors="coerce"
            )
            df = df.dropna(subset=["date"])
            last_row = df.iloc[-1]
            gold_all_ath = bool(last_row.get("all_time_high_flag", False))
    except Exception:
        pass
    gold_dark = gold_all_ath

    # 3. USD vs gold ratio (USD per oz vs 30-year range) – simplified proxy
    usd_vs_gold_ratio = 1.0 / gold_spot if gold_spot not in (0, float("nan")) else float("nan")
    usd_vs_gold_dark = (
        not math.isnan(usd_vs_gold_ratio) and usd_vs_gold_ratio <= 0.0001
    )

    # 4. Real 30-year yield extreme
    y10, y30, _ = fetch_tsy_yields_10_30()
    # Use CPI YoY from fetch_real_fed_funds_rate for rough real.
    _, src_real = fetch_real_fed_funds_rate()
    real30 = float("nan")
    try:
        if fred:
            cpi_yoy = fred.get_series("CPIAUCSL").pct_change(12).iloc[-1] * 100.0
            real30 = y30 - cpi_yoy
    except Exception:
        pass
    real30_dark = not math.isnan(real30) and (real30 >= 5.0 or real30 <= -5.0)

    # 5. Geopolitical Risk Index (GPR)
    gpr_val, gpr_src = fetch_gpr_index()
    gpr_dark = gpr_val >= 300.0

    # 6. Gini coefficient
    gini_latest, gini_src = fetch_gini_usa()
    gini_dark = gini_latest >= 0.50

    # 7. Wage share < 50%
    wage_share_latest, wage_src = fetch_wage_share_usa()
    wage_share_dark = wage_share_latest < 50.0

    # 8. Productivity growth negative multi-year
    prod_yoy_latest, prod_src = fetch_productivity_latest()
    prod_negative_years = 0
    try:
        df = load_csv(os.path.join(DATA_DIR, "us_productivity.csv"))
        if not df.empty:
            df["date"] = pd.to_datetime(
                df["date"], format="%Y-%m-%d", errors="coerce"
            )
            df = df.dropna(subset=["date", "prod_yoy"])
            last_24 = df.tail(24)
            negatives = (
                pd.to_numeric(last_24["prod_yoy"], errors="coerce") < 0
            ).sum()
            prod_negative_years = float(negatives) / 4.0
    except Exception:
        pass
    productivity_dark = prod_negative_years >= 1.5  # 6+ quarters negative

    # 9. USD reserve share drop
    usd_share, usd_prev, usd_src, _ = cofer_usd_share_latest()
    usd_reserve_dark = usd_share < 57.0 if not math.isnan(usd_share) else False

    # 10. Real Assets basket explosion
    real_assets_index, real_assets_dark = live_real_assets_basket()

    # 11. Official reset event
    cb_gold_increase, cb_gold_debug = central_bank_gold_tonnage_increase_from_seed()
    cb_titles, reset_titles = supercycle_news_alerts()
    official_reset_flag = get_official_reset_flag()
    reset_news_flag = len(reset_titles) >= 1
    no_return_news = cb_gold_increase or reset_news_flag or official_reset_flag

    # Dark-red counters
    dark_flags = [
        total_debt_dark,
        gold_dark,
        usd_vs_gold_dark,
        real30_dark,
        gpr_dark,
        gini_dark,
        wage_share_dark,
        productivity_dark,
        usd_reserve_dark,
        real_assets_dark,
        official_reset_flag,
    ]
    dark_count = sum(1 for f in dark_flags if f)

    # No-return triggers: (gold system reset, USD reserve collapse, 10y>7)
    # For now: 1) CB gold increase or reset news, 2) usd_reserve_dark, 3) y10 > 7
    no_return_flags = [
        no_return_news,
        usd_reserve_dark,
        y10 > 7.0 if not math.isnan(y10) else False,
    ]
    no_return_count = sum(1 for f in no_return_flags if f)

    info = {
        "debt": {
            "Current": f"{total_debt_gdp:.1f}%",
            "Dark red?": "🔴" if total_debt_dark else "🟡",
            "Why": "Total debt vs GDP at or above 400–450% means future credit growth will hit a hard wall.",
        },
        "gold": {
            "Current": f"${gold_spot:,.0f}/oz (all-ccy ATH: {'YES' if gold_all_ath else 'NO'})",
            "Dark red?": "🔴" if gold_dark else "🟡",
            "Why": "Gold breaking ATH against all major currencies is the market voting against fiat regimes.",
        },
        "usd_gold_ratio": {
            "Current": f"{usd_vs_gold_ratio:.5f} (USD per oz inverted proxy)",
            "Dark red?": "🔴" if usd_vs_gold_dark else "🟡",
            "Why": "When each dollar buys almost no gold, confidence in fiat is near exhaustion.",
        },
        "real30": {
            "Current": f"{real30:.2f}%" if not math.isnan(real30) else "NaN",
            "Dark red?": "🔴" if real30_dark else "🟡",
            "Why": "Extreme real long rates either crush borrowers or signal unanchored inflation.",
        },
        "gpr": {
            "Current": f"{gpr_val:.1f}",
            "Dark red?": "🔴" if gpr_dark else "🟡",
            "Why": "Sustained geopolitical risk spikes often sit around war, sanctions and regime shifts.",
        },
        "gini": {
            "Current": f"{gini_latest:.3f}",
            "Dark red?": "🔴" if gini_dark else "🟡",
            "Why": "Very high inequality makes debt and money resets politically inevitable.",
        },
        "wage_share": {
            "Current": f"{wage_share_latest:.1f}%",
            "Dark red?": "🔴" if wage_share_dark else "🟡",
            "Why": "When labour’s share of GDP falls, populist and redistributive shocks become more likely.",
        },
        "productivity": {
            "Current": f"{prod_yoy_latest:.2f}% YoY (negative years: {prod_negative_years:.1f})",
            "Dark red?": "🔴" if productivity_dark else "🟡",
            "Why": "Long stretches of negative productivity mean you’re borrowing growth from the future.",
        },
        "usd_reserve": {
            "Current": f"{usd_share:.1f}% (prev {usd_prev:.1f}%)"
            if not math.isnan(usd_share)
            else "No data",
            "Dark red?": "🔴" if usd_reserve_dark else "🟡",
            "Why": "Falling USD share of reserves signals gradual exit from the current dollar-led system.",
        },
        "real_assets": {
            "Current": f"{real_assets_index:.1f} (100 = baseline)",
            "Dark red?": "🔴" if real_assets_dark else "🟡",
            "Why": "When real assets 2–3x vs baseline, it means capital is fleeing paper claims.",
        },
        "reset_event": {
            "Current": "ON" if official_reset_flag else "OFF",
            "Dark red?": "🔴" if official_reset_flag else "🟡",
            "Why": "An explicit ‘reset’ decision or law is the final confirmation of the super-cycle turn.",
        },
    }

    return dark_count, no_return_count, info


# =============================================================================
# FETCH ALL LIVE STATE FOR THIS RUN
# =============================================================================

margin_bil, margin_pct_gdp, _ = fetch_margin_debt_finra()
gdp_tril, _ = fetch_us_gdp_trillions()
real_fed_rate, _ = fetch_real_fed_funds_rate()
put_call_val, _ = fetch_put_call_cboe()
aaii_bulls_val, _ = fetch_aaii_bulls()
sp_pe_val, _ = fetch_sp500_pe_multpl()
insider_buy_pct, _ = fetch_insider_ratio_openinsider()
hy_spread_val, _ = fetch_hy_spread()
vix_index_val, _ = fetch_vix_alpha_vantage()
sp_close_val, sp_ath_val, sp_drawdown_pct, sp_ytd_pct, _ = fetch_spx_from_fmp()
sp_above_200_pct, _ = fetch_spx_above_200ma()
fed_bs_yoy_val, _ = fetch_fed_balance_yoy()
sofr_spread_bp_val, _ = fetch_sofr_spread()
gpr_latest, _ = fetch_gpr_index()
gini_latest_val, _ = fetch_gini_usa()
wage_share_val, _ = fetch_wage_share_usa()
prod_yoy_val, _ = fetch_productivity_latest()
gold_spot_val, _ = fetch_gold_spot_fmp()
ten_yield, thirty_yield, _ = fetch_tsy_yields_10_30()
real_assets_index_val, real_assets_dark_flag = live_real_assets_basket()
usd_share_val, usd_prev_val, usd_src_val, _ = cofer_usd_share_latest()

# Short-term & long-term states
short_kill_count, short_kill_rows, near_ath_flag = compute_short_term_kill_combo()
long_dark_count, no_return_count, long_info = compute_long_term_supercycle()

# =============================================================================
# REGIME BANNER
# =============================================================================

short_label = (
    f"{short_kill_count}/10 short-term kill signals"
    if short_kill_count < 7
    else f"<span class='red'>{short_kill_count}/10 short-term KILL combo</span>"
)
long_label = (
    f"{long_dark_count}/11 long-term dark-red signals"
    if long_dark_count < 8
    else f"<span class='red'>{long_dark_count}/11 long-term super-cycle dark-red</span>"
)
no_return_label = (
    f"{no_return_count}/3 no-return triggers"
    if no_return_count < 2
    else f"<span class='orange'>{no_return_count}/3 no-return triggers</span>"
)

short_regime_text = (
    "FINAL TOP → 7+ kill signals while within −8% of ATH — trim 80–90% equity risk."
    if (short_kill_count >= 7 and near_ath_flag)
    else (
        "LATE-STAGE MELT-UP → multiple kill lights on, but price still holding."
        if short_kill_count >= 4
        else "MID-CYCLE → some froth, but not a full-blown kill cluster."
    )
)

long_regime_text = (
    "POINT OF NO RETURN → 8+ dark-red + 2 no-return triggers — pivot to hard assets for 5–15 years."
    if (long_dark_count >= 8 and no_return_count >= 2)
    else (
        "LATE SUPER-CYCLE → many structural lights flashing, but reset is not yet locked in."
        if long_dark_count >= 5
        else "MID SUPER-CYCLE → tensions building, but still time before the final break."
    )
)

banner_html = f"""
<div class="regime-banner">
    CURRENT REGIME:
    <br/>
    {short_label} • {long_label} • {no_return_label}
    <br/>
    <span class="orange">Short-term:</span> {short_regime_text}<br/>
    <span class="orange">Long-term:</span> {long_regime_text}
</div>
"""

st.markdown(banner_html, unsafe_allow_html=True)

if short_kill_count >= 7 and near_ath_flag:
    st.markdown(
        """
<div class="kill-box">
7+ short-term kill signals while the S&P 500 is within −8% of its all-time high → this is the zone where I start selling 80–90% of my equity risk.
</div>
""",
        unsafe_allow_html=True,
    )

if long_dark_count >= 8 and no_return_count >= 2:
    st.markdown(
        """
<div class="kill-box">
8+ long-term dark-red indicators plus 2+ no-return triggers → I tilt 80–100% of my long-horizon capital into hard assets (gold, BTC, farmland, real cash) for 5–15 years.
</div>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# TABS
# =============================================================================

tab_core, tab_long, tab_short = st.tabs(
    ["Core Mirror", "Long-Term Super-Cycle", "Short-Term Bubble Timing"]
)

# =============================================================================
# CORE MIRROR TAB
# =============================================================================

with tab_core:
    st.subheader("CORE MACRO SNAPSHOT — WHERE WE ARE, IN ONE GLANCE")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("US Nominal GDP", f"${gdp_tril:.2f}T")
        st.metric("Margin Debt", f"${margin_bil:.1f}B")
        st.metric("Margin % of GDP", f"{margin_pct_gdp:.2f}%")
    with col2:
        st.metric("Real Fed Funds Rate", f"{real_fed_rate:.2f}%")
        st.metric("HY Spread", f"{hy_spread_val:.1f} bps")
        st.metric("VIX Index", f"{vix_index_val:.2f}")
    with col3:
        st.metric("S&P 500 Close", f"{sp_close_val:,.0f}")
        st.metric("S&P 500 Drawdown", f"{sp_drawdown_pct:.2f}%")
        st.metric("S&P 500 YTD", f"{sp_ytd_pct:.2f}%")
    with col4:
        st.metric("Gold Spot (USD/oz)", f"${gold_spot_val:,.0f}")
        st.metric("10Y Yield", f"{ten_yield:.2f}%")
        st.metric("30Y Yield", f"{thirty_yield:.2f}%")

    st.markdown("---")
    st.markdown(
        "These are my anchor numbers: how big the economy is, how leveraged it is, what real rates look like, and whether volatility and credit are asleep or awake."
    )

    with st.expander("DATA SOURCES (CORE)", expanded=False):
        st.markdown(
            """
- **GDP** → FRED `GDP` (nominal) with local CSV mirror.
- **Margin debt** → FINRA margin statistics page (direct HTML table) with mirror seed.
- **Real Fed rate** → FRED `FEDFUNDS` minus YoY change in `CPIAUCSL`.
- **Put/Call** → CBOE total put/call CSV.
- **AAII bulls** → AAII sentiment CSV.
- **S&P P/E** → multpl.com S&P 500 P/E (scraped).
- **Insider ratio** → OpenInsider trades table.
- **HY spread** → FRED `BAMLH0A0HYM2`.
- **VIX** → Alpha Vantage `TIME_SERIES_DAILY` for symbol `^VIX`, with mirror.
- **S&P price & drawdown** → FMP `historical-price-full/^GSPC`.
- **Gold spot & FX** → FMP quotes for `XAUUSD`, `XAUEUR`, `XAUJPY`, `XAUGBP`, `XAUCHF`, `XAUCNY`.
- **Treasury yields** → FRED `DGS10` and `DGS30`.
- **Gini** → FRED `SIPOVGINIUSA`.
- **Wage share** → FRED `LABSHPUSA156NRUG`.
- **GPR** → policyuncertainty.com `US_GPR.csv`.
- **Productivity** → BLS PROD2 release (HTML) with mirror.
- **SOFR** → New York Fed reference rate page + FRED `FEDFUNDS`.
- **USD reserve share** → IMF COFER mirror CSV.
- **Real assets basket** → Gold, oil, BTC, and farmland mirror.
"""
        )

# =============================================================================
# LONG-TERM SUPER-CYCLE TAB
# =============================================================================

with tab_long:
    st.subheader("LONG-TERM SUPER-CYCLE — 11 DARK-RED LIGHTS")

    st.markdown(
        """
Here I track the slow, structural stuff that tells me where we are in the **big debt / big money / big power** cycle.  
I only change my very-long-horizon allocation when this panel hits certain thresholds.
"""
    )

    long_11_rows = [
        {
            "Signal": "Total Debt/GDP",
            "Current value": long_info["debt"]["Current"],
            "Dark red?": long_info["debt"]["Dark red?"],
            "Why it matters": long_info["debt"]["Why"],
        },
        {
            "Signal": "Gold ATH vs major currencies",
            "Current value": long_info["gold"]["Current"],
            "Dark red?": long_info["gold"]["Dark red?"],
            "Why it matters": long_info["gold"]["Why"],
        },
        {
            "Signal": "USD vs gold ratio",
            "Current value": long_info["usd_gold_ratio"]["Current"],
            "Dark red?": long_info["usd_gold_ratio"]["Dark red?"],
            "Why it matters": long_info["usd_gold_ratio"]["Why"],
        },
        {
            "Signal": "Real 30-year yield",
            "Current value": long_info["real30"]["Current"],
            "Dark red?": long_info["real30"]["Dark red?"],
            "Why it matters": long_info["real30"]["Why"],
        },
        {
            "Signal": "Geopolitical Risk Index (GPR)",
            "Current value": long_info["gpr"]["Current"],
            "Dark red?": long_info["gpr"]["Dark red?"],
            "Why it matters": long_info["gpr"]["Why"],
        },
        {
            "Signal": "Gini coefficient",
            "Current value": long_info["gini"]["Current"],
            "Dark red?": long_info["gini"]["Dark red?"],
            "Why it matters": long_info["gini"]["Why"],
        },
        {
            "Signal": "Wage share of GDP",
            "Current value": long_info["wage_share"]["Current"],
            "Dark red?": long_info["wage_share"]["Dark red?"],
            "Why it matters": long_info["wage_share"]["Why"],
        },
        {
            "Signal": "Productivity growth",
            "Current value": long_info["productivity"]["Current"],
            "Dark red?": long_info["productivity"]["Dark red?"],
            "Why it matters": long_info["productivity"]["Why"],
        },
        {
            "Signal": "USD reserve share",
            "Current value": long_info["usd_reserve"]["Current"],
            "Dark red?": long_info["usd_reserve"]["Dark red?"],
            "Why it matters": long_info["usd_reserve"]["Why"],
        },
        {
            "Signal": "Real Assets Basket",
            "Current value": long_info["real_assets"]["Current"],
            "Dark red?": long_info["real_assets"]["Dark red?"],
            "Why it matters": long_info["real_assets"]["Why"],
        },
        {
            "Signal": "Official Reset Event",
            "Current value": long_info["reset_event"]["Current"],
            "Dark red?": long_info["reset_event"]["Dark red?"],
            "Why it matters": long_info["reset_event"]["Why"],
        },
    ]

    df_long = pd.DataFrame(long_11_rows)
    st.dataframe(df_long, use_container_width=True, hide_index=True)

    st.markdown(
        f"**Dark red active:** {long_dark_count}/11 &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"**No-return triggers:** {no_return_count}/3"
    )

    st.markdown("---")
    st.markdown(
        """
**How I use this tab (long-term allocation rule):**

- If **< 5/11 dark red** → I treat it as mid-cycle. I keep a more balanced mix of risk assets vs hard assets.
- If **5–7/11 dark red** → I see it as **late super-cycle**. I gradually tilt into more *resilient* assets (gold, quality, cash).
- If **≥ 8/11 dark red** **and** **≥ 2/3 no-return triggers** →  
  I treat it as a **point-of-no-return**. I aim for **80–100% allocation to hard assets** (gold, BTC, farmland, strong cash) for **5–15 years**.
"""
    )

    with st.expander("SUPER-CYCLE POINT OF NO RETURN (final 6–24 months before reset)", expanded=False):
        st.markdown(
            """
This sub-panel is my attempt to not only see *that* the super-cycle is stretched, but whether we are entering the **irreversible final stretch**.

If enough of these go dark red **together**, the system is basically committing to a reset:
"""
        )

        long_8_rows = [
            {
                "Signal": "Total Debt/GDP",
                "Current value": long_info["debt"]["Current"],
                "DARK RED level": ">400–450% (or stops rising)",
                "Dark red?": long_info["debt"]["Dark red?"],
            },
            {
                "Signal": "Gold ATH vs major currencies",
                "Current value": long_info["gold"]["Current"],
                "DARK RED level": "Breaks new ATH vs EVERY major currency",
                "Dark red?": long_info["gold"]["Dark red?"],
            },
            {
                "Signal": "USD vs gold ratio",
                "Current value": long_info["usd_gold_ratio"]["Current"],
                "DARK RED level": "<0.10 oz per $1,000",
                "Dark red?": long_info["usd_gold_ratio"]["Dark red?"],
            },
            {
                "Signal": "Real 30-year yield",
                "Current value": long_info["real30"]["Current"],
                "DARK RED level": ">+5% OR <−5% for months",
                "Dark red?": long_info["real30"]["Dark red?"],
            },
            {
                "Signal": "Geopolitical Risk Index (GPR)",
                "Current value": long_info["gpr"]["Current"],
                "DARK RED level": ">300 and rising vertically",
                "Dark red?": long_info["gpr"]["Dark red?"],
            },
            {
                "Signal": "Gini coefficient",
                "Current value": long_info["gini"]["Current"],
                "DARK RED level": ">0.50 and climbing",
                "Dark red?": long_info["gini"]["Dark red?"],
            },
            {
                "Signal": "Wage share < 50%",
                "Current value": long_info["wage_share"]["Current"],
                "DARK RED level": "<50% of GDP",
                "Dark red?": long_info["wage_share"]["Dark red?"],
            },
            {
                "Signal": "Productivity growth negative for multiple years",
                "Current value": long_info["productivity"]["Current"],
                "DARK RED level": "Productivity negative ≥ 6 consecutive quarters",
                "Dark red?": long_info["productivity"]["Dark red?"],
            },
        ]

        df_long8 = pd.DataFrame(long_8_rows)
        st.dataframe(df_long8, use_container_width=True, hide_index=True)

        dark_red_8_flags = [
            long_info["debt"]["Dark red?"] == "🔴",
            long_info["gold"]["Dark red?"] == "🔴",
            long_info["usd_gold_ratio"]["Dark red?"] == "🔴",
            long_info["real30"]["Dark red?"] == "🔴",
            long_info["gpr"]["Dark red?"] == "🔴",
            long_info["gini"]["Dark red?"] == "🔴",
            long_info["wage_share"]["Dark red?"] == "🔴",
            long_info["productivity"]["Dark red?"] == "🔴",
        ]
        dark_red_8_count = sum(1 for b in dark_red_8_flags if b)

        st.markdown(
            f"**Dark red here:** {dark_red_8_count}/8 &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"**Global super-cycle:** {long_dark_count}/11 &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"**No-return triggers:** {no_return_count}/3"
        )
        st.markdown(
            """
**Rule of thumb I follow here**

- If **≥ 8/11 long-term dark red** AND **≥ 2/3 no-return triggers**  
  → I treat it as **locked-in reset** and shift the bulk of my long-horizon capital into **hard assets**.
"""
        )

    with st.expander("OFFICIAL RESET EVENT TOGGLE (manual safety switch)", expanded=False):
        current_flag = get_official_reset_flag()
        new_flag = st.checkbox(
            "Mark an official ‘monetary/system reset’ event as ON",
            value=current_flag,
            help="I only flip this after an actual law/treaty/official announcement that rewrites the currency or debt system.",
        )
        if new_flag != current_flag:
            set_official_reset_flag(new_flag)
            st.success(
                f"Official reset flag set to {'ON' if new_flag else 'OFF'} for today."
            )

        st.markdown(
            """
I use this as a **manual override** when something like a **“new Bretton Woods”** or explicit **debt restructuring** is announced.
"""
        )

# =============================================================================
# SHORT-TERM BUBBLE TIMING TAB
# =============================================================================

with tab_short:
    st.subheader("SHORT-TERM BUBBLE TIMING — 10/10 KILL COMBO")

    st.markdown(
        """
This tab is for **timing the top of the current bull/melt-up**.  
The idea: when **enough of these “kill” signals light up at once**, especially while the S&P is still near its **all-time high**, the odds of a **sharp drawdown** go way up.
"""
    )

    df_short = pd.DataFrame(short_kill_rows)
    st.dataframe(df_short, use_container_width=True, hide_index=True)

    st.markdown(
        f"**Current kill signals active:** {short_kill_count}/10 &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"**S&P 500 drawdown from ATH:** {sp_drawdown_pct:.2f}%"
    )

    st.markdown(
        """
**Rule I actually follow here (short-term allocation rule):**

- If **< 4/10 kill signals** → I treat it as a normal environment. I can ride the trend with normal risk.
- If **4–6/10 kill signals** → I start tightening risk: trimming leverage, rotating to quality, raising some cash.
- If **≥ 7/10 kill signals** **AND** the S&P 500 is within **−8% of ATH**  
  → I treat it as a **final top zone** and I personally plan to **sell 80–90% of my equity exposure over days/weeks**, not in panic, but systematically.
"""
    )

    with st.expander("WHY EACH SHORT-TERM INDICATOR MATTERS", expanded=False):
        st.markdown(
            """
- **Margin Debt % of GDP** → Tells me how much of the market is running on borrowed money.
- **Real Fed Funds Rate** → When it goes positive and high, liquidity is being pulled out of the system.
- **Total Put/Call Ratio** → Shows if everyone is leaning one way in options – crowding is dangerous.
- **AAII Bulls %** → Direct read of retail optimism – extremes often reverse.
- **S&P 500 P/E** → How much I’m paying for each dollar of earnings – valuation ceiling.
- **Insider Buying %** → Whether people with the best information are buying or dumping.
- **HY Spread** → Whether credit risk is mispriced – very tight spreads are late-cycle.
- **VIX Level** → Complacency vs fear; low VIX can be the calm before the storm.
- **% S&P above 200d MA** → Breadth; if very few names carry the index, the floor is thin.
- **Fed BS YoY / SOFR spread** → Whether liquidity is quietly being drained or funding is stressed.
"""
        )

# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Immortal edition of Econ Mirror — all signals wired to official or mirrored data, "
    "short-term kill combo + long-term super-cycle in one dashboard."
)
```
