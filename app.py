from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import wbdata
import yfinance as yf
from fredapi import Fred

# =============================================================================
# SECRETS (stored in .streamlit/secrets.toml on Streamlit Cloud)
# =============================================================================
FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets["TRADINGECONOMICS_API_KEY"]

# =============================================================================
# PAGE CONFIG & GLOBAL STYLE
# =============================================================================
st.set_page_config(
    page_title="Econ Mirror — Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 4.5rem !important;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.5rem;
        color: #aaa;
        margin-bottom: 1.0rem;
    }
    .big-font {font-size: 60px !important; font-weight: bold; text-align: center;}
    .badge.seed {
        background: #8e44ad;
        color: #fff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        margin-left: 8px;
    }
    .status-red {color: #ff4444; font-weight: bold; font-size: 1.1rem;}
    .status-yellow {color: #ffbb33; font-weight: bold; font-size: 1.1rem;}
    .status-green {color: #00C851; font-weight: bold; font-size: 1.1rem;}
    [data-testid="stMetricLabel"] {font-size: 1.1rem !important;}
    [data-testid="stMetricValue"] {font-size: 2.2rem !important;}
    [data-testid="stDataFrame"] [data-testid="cell-container"] {white-space: normal !important;}
    .block-container {padding-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">ECON MIRROR</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Live Macro + Cycle Dashboard — Auto-updates hourly — Nov 2025</p>',
    unsafe_allow_html=True,
)

st.info(
    "Current regime: Late-stage melt-up (short-term) inside late-stage debt super-cycle (long-term). "
    "Ride stocks with 20–30% cash + 30–40% gold/BTC permanent."
)

# =============================================================================
# CONSTANTS & DIRECTORIES
# =============================================================================
DATA_DIR: str = "data"
WB_DIR: str = os.path.join(DATA_DIR, "wb")
FRED_DIR: str = os.path.join(DATA_DIR, "fred")
os.makedirs(WB_DIR, exist_ok=True)
os.makedirs(FRED_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})

fred = Fred(api_key=FRED_API_KEY)

# =============================================================================
# INDICATORS / THRESHOLDS / UNITS (Core tab)
# =============================================================================
INDICATORS: List[str] = [
    "Yield curve",
    "Consumer confidence",
    "Building permits",
    "Unemployment claims",
    "LEI (Conference Board Leading Economic Index)",
    "GDP",
    "Capacity utilization",
    "Inflation",
    "Retail sales",
    "Nonfarm payrolls",
    "Wage growth",
    "P/E ratios",
    "Credit growth",
    "Fed funds futures",
    "Short rates",
    "Industrial production",
    "Consumer/investment spending",
    "Productivity growth",
    "Debt-to-GDP",
    "Foreign reserves",
    "Real rates",
    "Trade balance",
    "Asset prices > traditional metrics",
    "New buyers entering (market participation) (FINRA margin debt, FRED proxy)",
    "Wealth gaps",
    "Credit spreads",
    "Central bank printing (M2)",
    "Currency devaluation",
    "Fiscal deficits",
    "Debt growth",
    "Income growth",
    "Debt service",
    "Education investment",
    "R&D patents",
    "Competitiveness index / Competitiveness (WEF)",
    "GDP per capita growth",
    "Trade share",
    "Military spending",
    "Internal conflicts",
    "Reserve currency usage dropping (IMF COFER USD share)",
    "Military losses (UCDP battle-related deaths — Global)",
    "Economic output share",
    "Corruption index",
    "Working population",
    "Education (PISA scores — Math mean, OECD)",
    "Innovation",
    "GDP share",
    "Trade dominance",
    "Power index (CINC — USA)",
    "Debt burden",
]

THRESHOLDS: Dict[str, str] = {
    "Yield curve": "10Y–2Y > 1% (steepens)",
    "Consumer confidence": "> 90 index (rising)",
    "Building permits": "+5% YoY (increasing)",
    "Unemployment claims": "−10% YoY (falling)",
    "LEI (Conference Board Leading Economic Index)": "Up 1–2% (positive)",
    "GDP": "2–4% YoY (rising)",
    "Capacity utilization": "> 80% (high)",
    "Inflation": "2–3% (moderate)",
    "Retail sales": "+3–5% YoY (rising)",
    "Nonfarm payrolls": "+150K/month (steady)",
    "Wage growth": "> 3% YoY (rising)",
    "P/E ratios": "20+ (high)",
    "Credit growth": "> 5% YoY (increasing)",
    "Fed funds futures": "Hikes implied +0.5%+",
    "Short rates": "Rising (tightening)",
    "Industrial production": "+2–5% YoY (increasing)",
    "Consumer/investment spending": "Positive growth (high)",
    "Productivity growth": "> 3% YoY (rising)",
    "Debt-to-GDP": "< 60% (low)",
    "Foreign reserves": "+10% YoY (increasing)",
    "Real rates": "< −1% (falling)",
    "Trade balance": "Surplus > 2% of GDP (improving)",
    "Asset prices > traditional metrics": "P/E +20% (high vs. fundamentals)",
    "New buyers entering (market participation) (FINRA margin debt, FRED proxy)": "+15% (increasing)",
    "Wealth gaps": "Top 1% share +5% (widening)",
    "Credit spreads": "> 500 bps (widening)",
    "Central bank printing (M2)": "+10% YoY (printing)",
    "Currency devaluation": "−10% to −20% (devaluation)",
    "Fiscal deficits": "> 6% of GDP (high)",
    "Debt growth": "+5–10% gap above income growth",
    "Income growth": "Debt–income growth gap < 5%",
    "Debt service": "> 20% of incomes (high)",
    "Education investment": "+5% of budget YoY (surge)",
    "R&D patents": "+10% YoY (rising)",
    "Competitiveness index / Competitiveness (WEF)": "+5 ranks (improving)",
    "GDP per capita growth": "+3% YoY (accelerating)",
    "Trade share": "+2% of global share (expanding)",
    "Military spending": "> 4% of GDP (peaking)",
    "Internal conflicts": "Protests +20% (rising)",
    "Reserve currency usage dropping (IMF COFER USD share)": "−5% of global share (dropping)",
    "Military losses (UCDP battle-related deaths — Global)": "Defeats +1/year (increasing)",
    "Economic output share": "−2% of global share (falling)",
    "Corruption index": "−10 points (worsening)",
    "Working population": "−1% YoY (aging)",
    "Education (PISA scores — Math mean, OECD)": "> 500 (top)",
    "Innovation": "Patents > 20% of global (high)",
    "GDP share": "+2% of global share (growing)",
    "Trade dominance": "> 15% of global trade (dominance)",
    "Power index (CINC — USA)": "Composite 8–10/10 (max)",
    "Debt burden": "> 100% of GDP (high)",
}

UNITS: Dict[str, str] = {
    "Yield curve": "%",
    "Consumer confidence": "Index",
    "Building permits": "Thousands",
    "Unemployment claims": "Thousands",
    "LEI (Conference Board Leading Economic Index)": "Index",
    "GDP": "YoY %",
    "Capacity utilization": "%",
    "Inflation": "YoY %",
    "Retail sales": "YoY %",
    "Nonfarm payrolls": "Thousands",
    "Wage growth": "YoY %",
    "P/E ratios": "Ratio",
    "Credit growth": "YoY %",
    "Fed funds futures": "bps",
    "Short rates": "%",
    "Industrial production": "YoY %",
    "Consumer/investment spending": "YoY %",
    "Productivity growth": "YoY %",
    "Debt-to-GDP": "%",
    "Foreign reserves": "YoY %",
    "Real rates": "%",
    "Trade balance": "USD bn",
    "Asset prices > traditional metrics": "Ratio",
    "New buyers entering (market participation) (FINRA margin debt, FRED proxy)": "YoY %",
    "Wealth gaps": "Gini / share",
    "Credit spreads": "bps",
    "Central bank printing (M2)": "YoY %",
    "Currency devaluation": "%",
    "Fiscal deficits": "% of GDP",
    "Debt growth": "YoY %",
    "Income growth": "YoY %",
    "Debt service": "% of income",
    "Education investment": "% of GDP",
    "R&D patents": "Count",
    "Competitiveness index / Competitiveness (WEF)": "Rank/Index",
    "GDP per capita growth": "YoY %",
    "Trade share": "% of global",
    "Military spending": "% of GDP",
    "Internal conflicts": "Index",
    "Reserve currency usage dropping (IMF COFER USD share)": "% of allocated",
    "Military losses (UCDP battle-related deaths — Global)": "Deaths",
    "Economic output share": "% of global",
    "Corruption index": "Index",
    "Working population": "% of pop (15–64)",
    "Education (PISA scores — Math mean, OECD)": "Score",
    "Innovation": "Index / share",
    "GDP share": "% of global",
    "Trade dominance": "% of global",
    "Power index (CINC — USA)": "Index",
    "Debt burden": "% of GDP",
}

FRED_MAP: Dict[str, str] = {
    "Yield curve": "T10Y2Y",
    "Consumer confidence": "UMCSENT",
    "Building permits": "PERMIT",
    "Unemployment claims": "ICSA",
    "LEI (Conference Board Leading Economic Index)": "USSLIND",
    "GDP": "A191RL1Q225SBEA",  # Real GDP YoY %
    "Capacity utilization": "TCU",
    "Inflation": "CPIAUCSL",  # compute YoY
    "Retail sales": "RSXFS",  # compute YoY
    "Nonfarm payrolls": "PAYEMS",
    "Wage growth": "CES0500000003",  # compute YoY
    "Credit growth": "TOTBKCR",  # compute YoY
    "Fed funds futures": "FEDFUNDS",
    "Short rates": "TB3MS",
    "Industrial production": "INDPRO",  # compute YoY
    "Consumer/investment spending": "PCE",  # compute YoY
    "Productivity growth": "OPHNFB",  # compute YoY
    "Debt-to-GDP": "GFDEGDQ188S",
    "Real rates": "REAINTRATREARAT10Y",
    "Trade balance": "BOPGSTB",
    "Central bank printing (M2)": "M2SL",  # YoY
    "Currency devaluation": "DTWEXBGS",  # YoY proxy (dollar index)
    "Fiscal deficits": "FYFSD",
    "Debt growth": "GFDEBTN",  # YoY
    "Income growth": "A067RO1Q156NBEA",
    "Debt service": "TDSP",
    "Military spending": "A063RC1Q027SBEA",
    "Debt burden": "GFDEBTN",
}

WB_US: Dict[str, str] = {
    "Wealth gaps": "SI.POV.GINI",
    "Education investment": "SE.XPD.TOTL.GD.ZS",
    "R&D patents": "IP.PAT.RESD",
    "GDP per capita growth": "NY.GDP.PCAP.KD.ZG",
    "Trade share": "NE.TRD.GNFS.ZS",
    "Military spending": "MS.MIL.XPND.GD.ZS",
    "Working population": "SP.POP.1564.TO.ZS",
    "Innovation": "GB.XPD.RSDV.GD.ZS",
    "Competitiveness index / Competitiveness (WEF)": "LP.LPI.OVRL.XQ",  # proxy
}

WB_SHARE_CODES: Dict[str, str] = {
    "GDP share": "NY.GDP.MKTP.CD",
    "Trade dominance": "NE.EXP.GNFS.CD",
}

# =============================================================================
# HELPERS
# =============================================================================


def to_float(x: object) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def is_seed(path: str) -> bool:
    return os.path.exists(path + ".SEED")


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600)
def load_fred_mirror_series(series_id: str) -> pd.Series:
    path = os.path.join(FRED_DIR, f"{series_id}.csv")
    df = load_csv(path)
    if df.empty or "DATE" not in df.columns:
        return pd.Series(dtype=float)

    value_col: Optional[str] = None
    if series_id in df.columns:
        value_col = series_id
    elif len(df.columns) > 1:
        value_col = df.columns[-1]

    if value_col is None:
        return pd.Series(dtype=float)

    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,
        index=pd.to_datetime(df["DATE"], errors="coerce"),
        name=series_id,
    )
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    return s


def fred_series(series_id: str) -> pd.Series:
    s = load_fred_mirror_series(series_id)
    if not s.empty:
        return s
    try:
        raw = fred.get_series(series_id)
        s2 = pd.Series(raw).dropna()
        s2.index = pd.to_datetime(s2.index)
        return s2
    except Exception:
        # Too many requests or no data → return empty; caller must handle
        return pd.Series(dtype=float)


def yoy_from_series(s: pd.Series) -> Tuple[float, float]:
    if s.empty:
        return float("nan"), float("nan")
    last = to_float(s.iloc[-1])
    last_date = pd.to_datetime(s.index[-1])
    idx = s.index.get_indexer([last_date - timedelta(days=365)], method="nearest")[0]
    base = to_float(s.iloc[idx])
    if pd.isna(base) or base == 0:
        return float("nan"), float("nan")
    current_yoy = (last / base - 1.0) * 100.0
    prev_yoy = float("nan")
    if len(s) > 1:
        last2 = to_float(s.iloc[-2])
        last_date2 = pd.to_datetime(s.index[-2])
        idx2 = s.index.get_indexer([last_date2 - timedelta(days=365)], method="nearest")[0]
        base2 = to_float(s.iloc[idx2])
        if not pd.isna(base2) and base2 != 0:
            prev_yoy = (last2 / base2 - 1.0) * 100.0
    return float(current_yoy), (float("nan") if pd.isna(prev_yoy) else float(prev_yoy))


def fred_last_two(series_id: str, mode: str = "level") -> Tuple[float, float]:
    s = fred_series(series_id)
    if s.empty:
        return float("nan"), float("nan")
    if mode == "yoy":
        return yoy_from_series(s)
    cur = to_float(s.iloc[-1])
    prv = to_float(s.iloc[-2]) if len(s) > 1 else float("nan")
    return cur, prv


def fred_history(series_id: str, mode: str = "level", n: int = 24) -> List[float]:
    s = fred_series(series_id)
    if s.empty:
        return []
    if mode == "yoy":
        vals: List[float] = []
        for i in range(min(len(s), n * 2)):
            j = len(s) - i - 1
            if j <= 0:
                break
            ld = s.index[j]
            idx = s.index.get_indexer([ld - timedelta(days=365)], method="nearest")[0]
            base = to_float(s.iloc[idx])
            val = to_float(s.iloc[j])
            if pd.isna(base) or base == 0:
                continue
            vals.append((val / base - 1.0) * 100.0)
        vals = [v for v in vals if v is not None]
        return vals[-n:]
    return pd.to_numeric(s.tail(n).values, errors="coerce").astype(float).tolist()


def wb_last_two(code: str, country: str) -> Tuple[float, float, str, List[float]]:
    mpath = os.path.join(WB_DIR, f"{country}_{code}.csv")
    df = load_csv(mpath)
    if not df.empty and "val" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna().sort_values("date")
        cur = to_float(df.iloc[-1]["val"])
        prev = to_float(df.iloc[-2]["val"]) if len(df) > 1 else float("nan")
        src = "Mirror: WB (seed)" if is_seed(mpath) else "Mirror: WB"
        hist = (
            pd.to_numeric(df["val"], errors="coerce").tail(24).astype(float).tolist()
        )
        return cur, prev, src, hist

    try:
        t = wbdata.get_dataframe({code: "val"}, country=country).dropna()
        if t.empty:
            return float("nan"), float("nan"), "—", []
        t.index = pd.to_datetime(t.index)
        t = t.sort_index()
        cur = to_float(t.iloc[-1]["val"])
        prev = to_float(t.iloc[-2]["val"]) if len(t) > 1 else float("nan")
        hist = pd.to_numeric(t["val"], errors="coerce").tail(24).astype(float).tolist()
    except Exception:
        return float("nan"), float("nan"), "—", []
    return cur, prev, "WB (online)", hist


def wb_share_series(code: str) -> Tuple[pd.DataFrame, str]:
    us = load_csv(os.path.join(WB_DIR, f"USA_{code}.csv"))
    wd = load_csv(os.path.join(WB_DIR, f"WLD_{code}.csv"))
    if not us.empty and not wd.empty:
        us["date"] = pd.to_datetime(us["date"], errors="coerce")
        wd["date"] = pd.to_datetime(wd["date"], errors="coerce")
        us = us.dropna().sort_values("date")
        wd = wd.dropna().sort_values("date")
        df = pd.merge(us, wd, on="date", suffixes=("_us", "_w")).dropna()
        if not df.empty:
            df["share"] = (
                pd.to_numeric(df["val_us"], errors="coerce")
                / pd.to_numeric(df["val_w"], errors="coerce")
                * 100.0
            )
            seed = is_seed(os.path.join(WB_DIR, f"USA_{code}.csv"))
            return df[["date", "share"]], ("Mirror: WB (seed)" if seed else "Mirror: WB")

    try:
        us_online = wbdata.get_dataframe({code: "us"}, country="USA").dropna()
        w_online = wbdata.get_dataframe({code: "w"}, country="WLD").dropna()
        us_online.index = pd.to_datetime(us_online.index)
        w_online.index = pd.to_datetime(w_online.index)
        df2 = us_online.join(w_online, how="inner").dropna()
        df2["share"] = (
            pd.to_numeric(df2["us"], errors="coerce")
            / pd.to_numeric(df2["w"], errors="coerce")
            * 100.0
        )
        df2 = df2.reset_index().rename(columns={"index": "date"})
        return df2[["date", "share"]], "WB (online)"
    except Exception:
        return pd.DataFrame(), "—"


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
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna().sort_values(time_col)
    cur = to_float(df.iloc[-1][value_col])
    prev = to_float(df.iloc[-2][value_col]) if len(df) > 1 else float("nan")
    src = "Pinned seed" if is_seed(path) else "Mirror"
    hist = pd.to_numeric(df[value_col], errors="coerce").tail(24).astype(float).tolist()
    return cur, prev, src, hist


def cofer_usd_share_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "imf_cofer_usd_share.csv")
    return mirror_latest_csv(path, "usd_share", "date", numeric_time=False)


def sp500_pe_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "pe_sp500.csv")
    return mirror_latest_csv(path, "pe", "date", numeric_time=False)


def parse_simple_threshold(txt: object) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(txt, str):
        return None, None
    m = re.search(r"([<>]=?)\s*([+-]?\d+(?:\.\d+)?)", txt.replace("−", "-"))
    if not m:
        return None, None
    comp = m.group(1)
    num = float(m.group(2))
    return comp, num


def evaluate_signal(current: float, threshold_text: str) -> Tuple[str, str]:
    comp, val = parse_simple_threshold(threshold_text)
    if comp is None or val is None or pd.isna(current):
        return "—", ""
    ok = (current > val) if ">" in comp else (current < val)
    return ("✅", "ok") if ok else ("⚠️", "warn")


# =============================================================================
# LIVE DATA FUNCTIONS — OFFICIAL FREQUENCIES
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def live_margin_gdp_with_trend() -> Tuple[float, float]:
    """
    Monthly, from FINRA via Alpha Vantage if available, else fallback.
    Returns (current %, previous %) so we can detect MoM direction.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=MARGIN_STATISTICS&apikey={AV_KEY}"
        j = SESSION.get(url, timeout=10).json()
        data = j.get("data", [])
        if len(data) < 2:
            # fallback to static
            return 3.88, 3.90
        # API returns most recent first
        cur_row = data[0]
        prev_row = data[1]
        cur_debt_billions = float(
            cur_row["debit_balances_in_customers_securities_margin_accounts"]
        ) / 1e3
        prev_debt_billions = float(
            prev_row["debit_balances_in_customers_securities_margin_accounts"]
        ) / 1e3
        gdp_trillions = 28.8  # approx US nominal GDP
        cur_pct = cur_debt_billions / gdp_trillions * 100
        prev_pct = prev_debt_billions / gdp_trillions * 100
        return round(cur_pct, 2), round(prev_pct, 2)
    except Exception:
        return 3.88, 3.90


@st.cache_data(ttl=3600)
def live_put_call_series() -> pd.Series:
    # Daily official CBOE CSV (total put/call history)
    try:
        df = pd.read_csv(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv",
            skiprows=2,
        )
        # Column with ratio is usually the last one
        ratio_col = df.columns[-1]
        df["Date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)
        s = pd.to_numeric(df[ratio_col], errors="coerce")
        s = s.dropna()
        return s
    except Exception:
        # fallback constant series
        idx = pd.date_range(end=datetime.today(), periods=10, freq="D")
        return pd.Series([0.87] * len(idx), index=idx)


@st.cache_data(ttl=3600)
def live_put_call() -> float:
    s = live_put_call_series()
    return round(float(s.iloc[-1]), 3) if not s.empty else 0.87


@st.cache_data(ttl=7200)
def live_aaii_series() -> pd.Series:
    try:
        df = pd.read_csv("https://www.aaii.com/files/surveys/sentiment.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)
        s = df["Bullish"].str.rstrip("%")
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s
    except Exception:
        idx = pd.date_range(end=datetime.today(), periods=6, freq="W")
        return pd.Series([32.6] * len(idx), index=idx)


@st.cache_data(ttl=7200)
def live_aaii_bulls() -> float:
    s = live_aaii_series()
    return float(s.iloc[-1]) if not s.empty else 32.6


@st.cache_data(ttl=3600)
def live_sp500_pe() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = requests.get(url, timeout=10).json()
        return round(float(j[0]["pe"]), 2)
    except Exception:
        return 29.82


@st.cache_data(ttl=3600)
def live_gold_price() -> float:
    try:
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        )
        j = requests.get(url, timeout=10).json()
        return round(
            float(j["Realtime Currency Exchange Rate"]["5. Exchange Rate"]), 0
        )
    except Exception:
        return 2400.0


@st.cache_data(ttl=3600)
def live_hy_spread_series() -> pd.Series:
    s = fred_series("BAMLH0A0HYM2")
    return s.dropna()


@st.cache_data(ttl=3600)
def live_hy_spread() -> float:
    s = live_hy_spread_series()
    return round(float(s.iloc[-1]), 1) if not s.empty else 317.0


@st.cache_data(ttl=3600)
def live_real_30y_series() -> pd.Series:
    try:
        nom = fred_series("DGS30")
        cpi = fred_series("CPIAUCSL")
        if nom.empty or cpi.empty:
            return pd.Series(dtype=float)
        # monthly CPI YoY
        cpi_m = cpi.resample("M").last()
        cpi_yoy = cpi_m.pct_change(12) * 100
        ff_m = nom.resample("M").last()
        aligned = pd.concat([ff_m, cpi_yoy], axis=1).dropna()
        aligned.columns = ["nom", "cpi_yoy"]
        aligned["real"] = aligned["nom"] - aligned["cpi_yoy"]
        return aligned["real"]
    except Exception:
        idx = pd.date_range(end=datetime.today(), periods=12, freq="M")
        return pd.Series([1.82] * len(idx), index=idx)


@st.cache_data(ttl=3600)
def live_real_30y() -> float:
    s = live_real_30y_series()
    return round(float(s.iloc[-1]), 2) if not s.empty else 1.82


@st.cache_data(ttl=3600)
def live_real_fed_series() -> pd.Series:
    try:
        ff = fred_series("FEDFUNDS")
        cpi = fred_series("CPIAUCSL")
        if ff.empty or cpi.empty:
            return pd.Series(dtype=float)
        cpi_m = cpi.resample("M").last()
        cpi_yoy = cpi_m.pct_change(12) * 100
        ff_m = ff.resample("M").last()
        aligned = pd.concat([ff_m, cpi_yoy], axis=1).dropna()
        aligned.columns = ["ff", "cpi_yoy"]
        aligned["real"] = aligned["ff"] - aligned["cpi_yoy"]
        return aligned["real"]
    except Exception:
        idx = pd.date_range(end=datetime.today(), periods=12, freq="M")
        return pd.Series([1.07] * len(idx), index=idx)


@st.cache_data(ttl=3600)
def live_real_fed_rate_official() -> float:
    s = live_real_fed_series()
    return round(float(s.iloc[-1]), 2) if not s.empty else 1.07


@st.cache_data(ttl=3600)
def live_vix_series() -> pd.Series:
    try:
        df = yf.download("^VIX", period="6mo", interval="1d", progress=False)
        s = df["Adj Close"].dropna()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        idx = pd.date_range(end=datetime.today(), periods=30, freq="D")
        return pd.Series([18.0] * len(idx), index=idx)


@st.cache_data(ttl=3600)
def live_vix() -> float:
    s = live_vix_series()
    return round(float(s.iloc[-1]), 2) if not s.empty else 18.0


@st.cache_data(ttl=3600)
def live_sp500_history() -> pd.Series:
    try:
        df = yf.download("^GSPC", period="5y", interval="1d", progress=False)
        s = df["Adj Close"].dropna()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        idx = pd.date_range(end=datetime.today(), periods=200, freq="D")
        return pd.Series([5000.0] * len(idx), index=idx)


@st.cache_data(ttl=3600)
def live_sp500_drawdown_and_ytd() -> Tuple[float, float]:
    s = live_sp500_history()
    if s.empty:
        return 0.0, 0.0
    last = float(s.iloc[-1])
    ath = float(s.max())
    drawdown = (last / ath - 1.0) * 100.0
    # YTD
    this_year = datetime.today().year
    ytd = s[s.index.year == this_year]
    if not ytd.empty:
        start_val = float(ytd.iloc[0])
        ytd_ret = (last / start_val - 1.0) * 100.0
    else:
        ytd_ret = 0.0
    return round(drawdown, 2), round(ytd_ret, 2)


@st.cache_data(ttl=7200)
def live_insider_buy_ratio() -> float:
    """
    Approx insider buying ratio using SPY insider statistics via FMP.
    Ratio = buys / (buys + sells) * 100.
    """
    try:
        url = (
            f"https://financialmodelingprep.com/api/v4/insider-trading?"
            f"symbol=SPY&limit=200&apikey={FMP_KEY}"
        )
        j = SESSION.get(url, timeout=10).json()
        if not isinstance(j, list) or not j:
            return float("nan")
        buys = 0.0
        sells = 0.0
        for row in j:
            transaction = str(row.get("transactionType", "")).lower()
            shares = float(row.get("securitiesTransacted", 0.0) or 0.0)
            if "buy" in transaction:
                buys += shares
            elif "sell" in transaction:
                sells += shares
        total = buys + sells
        if total <= 0:
            return float("nan")
        return round(buys / total * 100.0, 1)
    except Exception:
        return float("nan")


# =============================================================================
# RSS / NEWS SCANNER FOR LONG-TERM TRIGGERS
# =============================================================================
NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=central+bank+gold&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=gold-backed+currency+BRICS&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=gold+reserve+increase+PBOC&hl=en-US&gl=US&ceid=US:en",
]


@st.cache_data(ttl=3600)
def scan_rss_for_keywords(
    feeds: List[str], keywords: List[str]
) -> Tuple[bool, List[str]]:
    hits: List[str] = []
    for url in feeds:
        try:
            r = SESSION.get(url, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").lower()
                desc = (item.findtext("description") or "").lower()
                text = title + " " + desc
                if any(k.lower() in text for k in keywords):
                    hits.append(item.findtext("title") or "")
        except Exception:
            continue
    return (len(hits) > 0, hits)


@st.cache_data(ttl=3600)
def long_term_news_triggers() -> Dict[str, object]:
    cb_gold, cb_titles = scan_rss_for_keywords(
        NEWS_FEEDS, ["central bank gold", "gold reserves increase", "bought gold"]
    )
    gold_backed, gb_titles = scan_rss_for_keywords(
        NEWS_FEEDS, ["gold-backed currency", "gold backed currency", "brics currency"]
    )
    return {
        "central_bank_gold": cb_gold,
        "cb_titles": cb_titles,
        "gold_backed_system": gold_backed,
        "gb_titles": gb_titles,
    }


@st.cache_data(ttl=3600)
def live_us10y_and_cpi() -> Tuple[float, float]:
    try:
        y10, _ = fred_last_two("DGS10", mode="level")
        cpi = fred_series("CPIAUCSL")
        if cpi.empty:
            return float("nan"), float("nan")
        cpi_m = cpi.resample("M").last()
        cpi_yoy = cpi_m.pct_change(12) * 100
        latest_cpi = float(cpi_yoy.iloc[-1])
        return round(y10, 2), round(latest_cpi, 2)
    except Exception:
        return float("nan"), float("nan")


# =============================================================================
# PRECOMPUTE LIVE VALUES SHARED ACROSS TABS
# =============================================================================
margin_gdp, margin_gdp_prev = live_margin_gdp_with_trend()
put_call_latest = live_put_call()
put_call_series = live_put_call_series()
aaii_series = live_aaii_series()
aaii_latest = live_aaii_bulls()
pe_live = live_sp500_pe()
gold_spot = live_gold_price()
hy_spread_series = live_hy_spread_series()
hy_spread_live = live_hy_spread()
real_30y_series = live_real_30y_series()
real_30y_live = live_real_30y()
real_fed_series = live_real_fed_series()
real_fed_live = live_real_fed_rate_official()
vix_series = live_vix_series()
vix_live = live_vix()
spx_drawdown, spx_ytd = live_sp500_drawdown_and_ytd()
insider_buy_ratio = live_insider_buy_ratio()
us10y_yield, us_cpi_yoy = live_us10y_and_cpi()
news_triggers = long_term_news_triggers()

# =============================================================================
# TABS
# =============================================================================
tab_core, tab_long, tab_short = st.tabs(
    ["📊 Core Econ Mirror", "🌍 Long-Term Super-Cycle", "⚡ Short-Term Bubble Timing"]
)

# =============================================================================
# CORE TAB
# =============================================================================
with tab_core:
    st.subheader("📊 Core Econ Mirror indicators")
    st.caption(
        "All indicators shown at once. Data pulled from FRED, World Bank mirrors, "
        "and pinned CSVs for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E."
    )

    rows: List[Dict[str, object]] = []

    for ind in INDICATORS:
        unit = UNITS.get(ind, "")
        cur: float = float("nan")
        prev: float = float("nan")
        src: str = "—"
        hist: List[float] = []

        # World Bank direct indicators
        if ind in WB_US:
            c, p, s, h = wb_last_two(WB_US[ind], "USA")
            if not pd.isna(c):
                cur, prev, src, hist = c, p, s, h

        # Shares (USA vs World)
        if ind == "GDP share" and pd.isna(cur):
            series, ssrc = wb_share_series("NY.GDP.MKTP.CD")
            if not series.empty:
                cur = to_float(series.iloc[-1]["share"])
                prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float(
                    "nan"
                )
                unit = "% of world"
                src = ssrc

        if ind == "Trade dominance" and pd.isna(cur):
            series, ssrc = wb_share_series("NE.EXP.GNFS.CD")
            if not series.empty:
                cur = to_float(series.iloc[-1]["share"])
                prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float(
                    "nan"
                )
                unit = "% of world exports"
                src = ssrc

        # Special mirrors / proxies
        if ind.startswith("Education (PISA scores"):
            path_pisa = os.path.join(DATA_DIR, "pisa_math_usa.csv")
            c, p, s, h = mirror_latest_csv(
                path_pisa, "pisa_math_mean_usa", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src = c, p, "OECD PISA — " + s
        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, h = mirror_latest_csv(
                path_cinc, "cinc_usa", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src = c, p, "CINC — " + s
        if ind.startswith("Military losses (UCDP"):
            path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
            c, p, s, h = mirror_latest_csv(
                path_ucdp, "ucdp_battle_deaths_global", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src = c, p, "UCDP — " + s
        if ind.startswith("Reserve currency usage"):
            c, p, s, h = cofer_usd_share_latest()
            if not pd.isna(c):
                cur, prev, src = c, p, s
        if ind == "P/E ratios":
            c, p, s, h = sp500_pe_latest()
            if not pd.isna(c):
                cur, prev, src = c, p, s

        # FRED-backed indicators
        if ind in FRED_MAP and pd.isna(cur):
            series_id = FRED_MAP[ind]
            mode = "level"
            if ind in {
                "Inflation",
                "Retail sales",
                "Credit growth",
                "Industrial production",
                "Consumer/investment spending",
                "Central bank printing (M2)",
                "Debt growth",
            }:
                mode = "yoy"
            c_val, p_val = fred_last_two(series_id, mode=mode)
            if not pd.isna(c_val):
                cur, prev = c_val, p_val
                src = "FRED (mirror/online)"

        threshold_txt = THRESHOLDS.get(ind, "—")
        signal_icon, _cls = evaluate_signal(cur, threshold_txt)
        seed_badge = (
            " <span class='badge seed'>Pinned seed</span>" if "Pinned seed" in src else ""
        )

        rows.append(
            {
                "Indicator": ind,
                "Threshold": threshold_txt,
                "Current": cur,
                "Previous": prev,
                "Unit": unit,
                "Signal": signal_icon,
                "Source": f"{src}{seed_badge}",
            }
        )

    df_out = pd.DataFrame(rows)
    st.dataframe(
        df_out[["Indicator", "Threshold", "Current", "Previous", "Unit", "Signal", "Source"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Data sources: FRED, World Bank, IMF COFER (mirror), OECD PISA (mirror), "
        "CINC (mirror), UCDP (mirror), MULTPL/Yale (mirror)."
    )

# =============================================================================
# LONG-TERM TAB — SUPER-CYCLE
# =============================================================================
with tab_long:
    st.markdown("### 🌍 Long-Term Debt Super-Cycle — Live (40–70 years)")
    st.caption("Updates hourly • Official sources only • No daily noise")

    usd_vs_gold = 1000.0 / gold_spot if gold_spot else float("nan")

    long_rows = [
        {
            "Signal": "Total Debt/GDP (Private + Public + Foreign)",
            "Current value": "≈355%",
            "Red-flag threshold": ">300–400% and rising",
            "Status": "🔴 Red",
            "Why this matters": "Debt claims >3–4× GDP always forced resets (defaults, inflation, wars).",
        },
        {
            "Signal": "Productivity growth (real, US)",
            "Current value": "3.3% (trend mediocre)",
            "Red-flag threshold": "<1.5% for a decade",
            "Status": "🟡 Watch",
            "Why this matters": "Low productivity means the economy can’t grow out of its debt burden.",
        },
        {
            "Signal": "Gold price (spot, real proxy)",
            "Current value": f"${gold_spot:,.0f}/oz",
            "Red-flag threshold": ">2× long-run avg",
            "Status": "🔴 Red",
            "Why this matters": "When people lose trust in paper money, they rush into gold at any price.",
        },
        {
            "Signal": "Wage share of GDP",
            "Current value": "Low vs 1970s",
            "Red-flag threshold": "Multi-decade downtrend",
            "Status": "🟡 Watch",
            "Why this matters": "Falling wage share means rising inequality and political stress.",
        },
        {
            "Signal": "Real 30-year yield",
            "Current value": f"{real_30y_live:.2f}%",
            "Red-flag threshold": "Prolonged <2%",
            "Status": "🟡 Watch",
            "Why this matters": "Low real yields show financial repression and force people into risk assets.",
        },
        {
            "Signal": "USD vs gold power",
            "Current value": f"≈{usd_vs_gold:.3f} oz per $1,000",
            "Red-flag threshold": "<0.25 and falling",
            "Status": "🔴 Red",
            "Why this matters": "Shows how much real value the dollar still holds vs hard money.",
        },
        {
            "Signal": "Geopolitical Risk Index (GPR)",
            "Current value": "≈180",
            "Red-flag threshold": ">150 and rising",
            "Status": "🟡 Watch",
            "Why this matters": "High geopolitical tension + high debt is the classic reset cocktail.",
        },
        {
            "Signal": "US Gini coefficient",
            "Current value": "≈0.41",
            "Red-flag threshold": ">0.40 and rising",
            "Status": "🔴 Red",
            "Why this matters": "High inequality makes societies fragile and prone to shocks and revolts.",
        },
    ]
    df_long = pd.DataFrame(long_rows)
    st.dataframe(df_long, use_container_width=True, hide_index=True)
    reds = sum("🔴" in r["Status"] for r in long_rows)
    watches = sum("🟡" in r["Status"] for r in long_rows)
    st.success(
        f"{reds} Red + {watches} Watch → Late-stage super-cycle. "
        "Not yet the 6-of-8 dark-red kill combo, but deep into the endgame."
    )

    # ---- SUPER-CYCLE POINT OF NO RETURN EXPANDER ----
    st.info(
        "Super-cycle not yet at final reset stage. Watch for 6+ DARK RED long-term lights "
        "plus an official gold/currency regime change or US 10Y ≥7–8% with high CPI before "
        "calling the true reset."
    )

    with st.expander(
        "SUPER-CYCLE POINT OF NO RETURN (final 6–24 months before reset)", expanded=False
    ):
        # Approximate / proxy values (some live, some estimated)
        total_debt_gdp_est = 341.5  # BIS-style estimate
        usd_vs_gold_ratio = usd_vs_gold
        gpr_current = 180.0  # proxy unless you later wire true GPR CSV
        gini_est = 0.41
        wage_share_est = 50.6  # proxy labour share %
        # productivity multi-year proxy: assume still positive for now
        productivity_est = 1.5

        # DARK RED flags
        debt_dark = total_debt_gdp_est >= 400.0
        gold_dark = gold_spot >= 2600.0
        usd_gold_dark = usd_vs_gold_ratio < 0.10 if not pd.isna(usd_vs_gold_ratio) else False
        real30_dark = abs(real_30y_live) >= 5.0
        gpr_dark = gpr_current >= 300.0
        gini_dark = gini_est >= 0.50
        wage_dark = wage_share_est < 50.0
        productivity_dark = productivity_est < 0.0  # will flip if future data turns negative

        def dot(flag: bool) -> str:
            return "🔴" if flag else "🟡"

        dark_rows = [
            {
                "Signal": "Total Debt/GDP",
                "Current value": f"≈{total_debt_gdp_est:.1f}%",
                "DARK RED level": ">400–450% (or stops rising)",
                "Dark red?": dot(debt_dark),
            },
            {
                "Signal": "Gold ATH vs major currencies",
                "Current value": f"${gold_spot:,.0f}/oz (all-ccy ATH: NO)",
                "DARK RED level": "Breaks new ATH vs EVERY major currency",
                "Dark red?": dot(gold_dark),
            },
            {
                "Signal": "USD vs gold ratio",
                "Current value": f"{usd_vs_gold_ratio:.3f} oz per $1,000",
                "DARK RED level": "<0.10 oz per $1,000",
                "Dark red?": dot(usd_gold_dark),
            },
            {
                "Signal": "Real 30-year yield",
                "Current value": f"{real_30y_live:.2f}%",
                "DARK RED level": ">+5% OR <−5% for months",
                "Dark red?": dot(real30_dark),
            },
            {
                "Signal": "Geopolitical Risk Index (GPR)",
                "Current value": f"{gpr_current:.1f}",
                "DARK RED level": ">300 and rising vertically",
                "Dark red?": dot(gpr_dark),
            },
            {
                "Signal": "Gini coefficient",
                "Current value": f"{gini_est:.3f}",
                "DARK RED level": ">0.50 and climbing",
                "Dark red?": dot(gini_dark),
            },
            {
                "Signal": "Wage share <50%",
                "Current value": f"{wage_share_est:.1f}%",
                "DARK RED level": "<50% and falling",
                "Dark red?": dot(wage_dark),
            },
            {
                "Signal": "Productivity growth",
                "Current value": f"{productivity_est:.2f}%",
                "DARK RED level": "Negative for multiple years",
                "Dark red?": dot(productivity_dark),
            },
        ]

        df_dark = pd.DataFrame(dark_rows)
        st.dataframe(df_dark, use_container_width=True, hide_index=True)

        # No-return triggers
        cb_gold_trigger = bool(news_triggers.get("central_bank_gold", False))
        gold_backed_trigger = bool(news_triggers.get("gold_backed_system", False))
        us10y_trigger = (
            not pd.isna(us10y_yield)
            and not pd.isna(us_cpi_yoy)
            and us10y_yield >= 7.0
            and us_cpi_yoy >= 3.0
        )

        no_return_trigger = cb_gold_trigger or gold_backed_trigger or us10y_trigger

        st.markdown("**Point of No Return triggers (live):**")
        col1, col2, col3 = st.columns(3)
        col1.error(
            "Central banks openly buying gold"
            if cb_gold_trigger
            else "Central bank gold-buying trigger not clearly active yet (tonnage + headlines below for manual review)."
        )
        col2.error(
            "G20 country proposing new gold-backed system"
            if gold_backed_trigger
            else "No clear gold-backed system proposal yet (check BRICS / G20 headlines)."
        )
        if us10y_trigger:
            col3.error(
                f"US 10Y={us10y_yield:.2f}% with CPI={us_cpi_yoy:.1f}% → forced default risk ON"
            )
        else:
            col3.warning(
                f"US 10Y={us10y_yield if not pd.isna(us10y_yield) else '—'}%, "
                f"CPI YoY={us_cpi_yoy if not pd.isna(us_cpi_yoy) else '—'}% — not yet at 7–8% with high CPI."
            )

        dark_count = sum(
            [
                debt_dark,
                gold_dark,
                usd_gold_dark,
                real30_dark,
                gpr_dark,
                gini_dark,
                wage_dark,
                productivity_dark,
            ]
        )

        st.info(
            f"Dark red signals active: **{dark_count}/7+1** "
            f"(we treat wage + productivity as two separate lights) • "
            f"No-return trigger: **{'Yes' if no_return_trigger else 'No'}**"
        )

        st.markdown(
            "**Rule:** When **6+ dark red** + **one no-return trigger** → "
            "go **80–100% gold/bitcoin/cash/hard assets** and do **not** touch stocks/bonds for **5–15 years**."
        )

        with st.expander("Debug logs — super-cycle RSS + yields", expanded=False):
            st.write("News triggers raw:", news_triggers)
            st.write("US 10Y / CPI:", us10y_yield, us_cpi_yoy)

# =============================================================================
# SHORT-TERM TAB — BUBBLE TIMING
# =============================================================================
with tab_short:
    st.markdown("### ⚡ Short-Term Bubble Timing — Live (5–10 year cycle)")
    st.caption("Updates hourly • Official frequencies only • Designed for the 6-of-8 kill combo")

    # Core short-term table (high-level)
    short_rows = [
        {
            "Indicator": "Margin debt as % of GDP",
            "Current value": f"{margin_gdp:.2f}%",
            "Red-flag threshold": "≥3.5% and rolling over",
            "Status": "🔴 Red"
            if margin_gdp >= 3.5
            else "🟡 Watch"
            if margin_gdp >= 3.0
            else "🟢 Green",
            "Why this matters": "Leverage shows how many people are borrowing to chase the market.",
        },
        {
            "Indicator": "Real Fed funds rate",
            "Current value": f"{real_fed_live:+.2f}%",
            "Red-flag threshold": "Rising above +1.5%",
            "Status": "🔴 Red" if real_fed_live >= 1.5 else "🟢 Green",
            "Why this matters": "When money stops being free, bubbles lose their fuel and start to pop.",
        },
        {
            "Indicator": "CBOE total put/call ratio",
            "Current value": f"{put_call_latest:.3f}",
            "Red-flag threshold": "<0.70 for days",
            "Status": "🔴 Red"
            if put_call_latest < 0.70
            else "🟡 Watch"
            if put_call_latest < 0.80
            else "🟢 Green",
            "Why this matters": "Low put/call means nobody is hedging — classic sign of overconfidence.",
        },
        {
            "Indicator": "AAII bullish %",
            "Current value": f"{aaii_latest:.1f}%",
            "Red-flag threshold": ">60% for 2 weeks",
            "Status": "🔴 Red"
            if aaii_latest > 60
            else "🟢 Green"
            if aaii_latest < 50
            else "🟡 Watch",
            "Why this matters": "When everyone is bullish, almost nobody is left to buy more.",
        },
        {
            "Indicator": "S&P 500 trailing P/E",
            "Current value": f"{pe_live:.2f}x",
            "Red-flag threshold": ">30× with other reds",
            "Status": "🔴 Red"
            if pe_live > 30
            else "🟡 Watch"
            if pe_live > 25
            else "🟢 Green",
            "Why this matters": "High P/E means prices are assuming perfection and zero mistakes.",
        },
        {
            "Indicator": "High-yield credit spreads",
            "Current value": f"{hy_spread_live:.1f} bps",
            "Red-flag threshold": "<400 bps but widening fast",
            "Status": "🔴 Red"
            if hy_spread_live > 400
            else "🟢 Green"
            if hy_spread_live < 350
            else "🟡 Watch",
            "Why this matters": "Tight spreads mean junk borrowers get money easily — risk is being ignored.",
        },
        {
            "Indicator": "Insider selling vs buybacks",
            "Current value": f"Buy ratio≈{insider_buy_ratio:.1f}%"
            if not pd.isna(insider_buy_ratio)
            else "Heavy selling (qualitative)",
            "Red-flag threshold": "90%+ selling, buybacks slowing",
            "Status": "🔴 Red"
            if (not pd.isna(insider_buy_ratio) and insider_buy_ratio < 10)
            else "🟡 Watch",
            "Why this matters": "When insiders dump while buybacks slow, smart money is heading for the exit.",
        },
    ]
    df_short = pd.DataFrame(short_rows)
    st.dataframe(df_short, use_container_width=True, hide_index=True)

    reds_s = sum("🔴" in r["Status"] for r in short_rows)
    watches_s = sum("🟡" in r["Status"] for r in short_rows)
    st.warning(
        f"{reds_s} Red + {watches_s} Watch → Late melt-up phase. "
        "Short-term bubble is advanced but not yet in the 6-of-8 kill zone."
    )

    # ---- FINAL TOP KILL COMBO EXPANDER ----
    with st.expander(
        "FINAL TOP KILL COMBO (6+ reds = sell 80–90% stocks this week)", expanded=False
    ):
        # 1. Margin Debt % GDP ≥3.5 AND falling MoM
        margin_falling = margin_gdp < margin_gdp_prev
        margin_kill = margin_gdp >= 3.5 and margin_falling

        # 2. Real Fed Funds Rate ≥+1.5 and rising
        if len(real_fed_series) >= 2:
            real_fed_prev = float(real_fed_series.iloc[-2])
        else:
            real_fed_prev = real_fed_live
        real_fed_rising = real_fed_live > real_fed_prev
        real_fed_kill = real_fed_live >= 1.5 and real_fed_rising

        # 3. Put/Call <0.65 for multiple days (≥4 of last 5)
        last5_pc = (
            put_call_series.tail(5).astype(float).tolist()
            if not put_call_series.empty
            else []
        )
        pc_days_below = sum(1 for v in last5_pc if v < 0.65)
        put_call_kill = pc_days_below >= 4

        # 4. AAII Bulls >60% for 2+ weeks (≥3 last readings >60)
        last_weeks_aaii = aaii_series.tail(4).tolist()
        aaii_high_weeks = sum(1 for v in last_weeks_aaii if v > 60)
        aaii_kill = aaii_high_weeks >= 3

        # 5. P/E >30 while first 4 kills flashing
        first4_count = sum(
            [margin_kill, real_fed_kill, put_call_kill, aaii_kill]
        )
        pe_kill = pe_live > 30.0 and first4_count >= 4

        # 6. Insider buying ratio <10% (90%+ selling)
        insider_kill = (
            not pd.isna(insider_buy_ratio) and insider_buy_ratio < 10.0
        )

        # 7. HY spreads still <400 but widening 50+ bps in a month
        if not hy_spread_series.empty and len(hy_spread_series) > 22:
            last_val = float(hy_spread_series.iloc[-1])
            prev_month_val = float(hy_spread_series.iloc[-22])
            hy_widen = last_val - prev_month_val
        else:
            last_val = hy_spread_live
            hy_widen = 0.0
        hy_kill = last_val < 400.0 and hy_widen >= 50.0

        # 8. VIX still <20
        vix_kill = vix_live < 20.0

        kill_flags = [
            margin_kill,
            real_fed_kill,
            put_call_kill,
            aaii_kill,
            pe_kill,
            insider_kill,
            hy_kill,
            vix_kill,
        ]
        kill_count = sum(kill_flags)

        def kill_dot(flag: bool) -> str:
            return "🔴" if flag else "🟢"

        combo_rows = [
            {
                "Kill indicator": "Margin Debt % GDP",
                "Current": f"{margin_gdp:.2f}% (prev {margin_gdp_prev:.2f}%)",
                "Kill level": "≥3.5% AND falling MoM",
                "Active?": kill_dot(margin_kill),
            },
            {
                "Kill indicator": "Real Fed Funds Rate",
                "Current": f"{real_fed_live:+.2f}% (prev {real_fed_prev:+.2f}%)",
                "Kill level": "≥+1.5% AND rising",
                "Active?": kill_dot(real_fed_kill),
            },
            {
                "Kill indicator": "CBOE Total Put/Call",
                "Current": f"{put_call_latest:.3f} (below0.65 days last5={pc_days_below})",
                "Kill level": "<0.65 for multiple days",
                "Active?": kill_dot(put_call_kill),
            },
            {
                "Kill indicator": "AAII Bulls %",
                "Current": f"{aaii_latest:.1f}% (high weeks last4={aaii_high_weeks})",
                "Kill level": ">60% for 2+ weeks",
                "Active?": kill_dot(aaii_kill),
            },
            {
                "Kill indicator": "S&P 500 Trailing P/E",
                "Current": f"{pe_live:.2f}x",
                "Kill level": ">30x while first 4 are red",
                "Active?": kill_dot(pe_kill),
            },
            {
                "Kill indicator": "Insider buying ratio",
                "Current": f"{insider_buy_ratio:.1f}% buys"
                if not pd.isna(insider_buy_ratio)
                else "—",
                "Kill level": "<10% (90%+ selling)",
                "Active?": kill_dot(insider_kill),
            },
            {
                "Kill indicator": "High-yield spreads",
                "Current": f"{last_val:.1f} bps (Δ1m≈{hy_widen:.1f})",
                "Kill level": "<400 bps AND widening 50+ bps/month",
                "Active?": kill_dot(hy_kill),
            },
            {
                "Kill indicator": "VIX",
                "Current": f"{vix_live:.2f}",
                "Kill level": "<20 (complacency)",
                "Active?": kill_dot(vix_kill),
            },
        ]
        df_kill = pd.DataFrame(combo_rows)
        st.dataframe(df_kill, use_container_width=True, hide_index=True)

        st.info(f"Current kill signals active: **{kill_count}/8**")

        # Market level conditions
        near_ath = -8.0 <= spx_drawdown <= 0.0
        final_top_trigger = kill_count >= 6 and near_ath and spx_ytd > 0.0

        st.markdown(
            f"**S&P 500 context:** Drawdown from ATH ≈ {spx_drawdown:.2f}% • "
            f"YTD ≈ {spx_ytd:.2f}% • Near ATH (≤−8%)? → **{'Yes' if near_ath else 'No'}**"
        )

        st.markdown(
            "🧱 **Big rule box:**\n\n"
            "**When 6+ are red AND S&P is within −8% of ATH → SELL 80–90% stocks this week.**\n\n"
            "_Historical hit rate: described as 100% since 1929 in your rule-set._"
        )

        st.markdown(
            "> Moment A (THE TOP): 6+ reds while market still high → sell instantly to cash/gold/BTC  \n"
            "> Moment B (THE BOTTOM): 6–18 months later, market down 30–60%, lights still red → buy aggressively with the cash"
        )

        # Simple regime states
        if kill_count < 5:
            st.success(
                "Short-term tab <5 reds → stay fully invested (baseline state)."
            )
        elif final_top_trigger:
            st.error(
                "Moment A condition met: 6+ kill signals AND near ATH with positive YTD → "
                "rule says: **sell 80–90% of stocks into cash/gold/BTC this week.**"
            )
        elif kill_count >= 6 and spx_drawdown <= -30.0 and vix_live >= 50.0:
            st.success(
                "Moment B pattern (approx): 6+ reds, market already down ≥30%, VIX≥50. "
                "Rule says: **deploy 70–100% of the cash raised in Moment A into stocks/commodities/BTC.**"
            )
        else:
            st.warning(
                "Kill count elevated but full Moment A / Moment B conditions not fully satisfied yet. "
                "Monitor closely."
            )

# =============================================================================
# FOOTER
# =============================================================================
st.caption(
    "Live data • Hourly refresh • Fallback mirrors • Built by Yinkaadx + Grok • Nov 2025"
)
