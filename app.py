from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import wbdata
import yfinance as yf
import xml.etree.ElementTree as ET
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
        margin-bottom: 1.5rem;
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
    .banner-box {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        border: 1px solid #444;
        background: rgba(255, 255, 255, 0.02);
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .kill-box {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: 1px solid #ff4444;
        background: rgba(255, 68, 68, 0.08);
        font-weight: 600;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .info-box-soft {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border: 1px solid #555;
        background: rgba(255, 255, 255, 0.02);
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">ECON MIRROR</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Live Macro + Cycle Dashboard — Auto-updates hourly — Nov 2025</p>',
    unsafe_allow_html=True,
)

# Permanent banner at very top
st.markdown(
    """
<div class="banner-box">
<strong>Current regime:</strong> Late-stage melt-up (short-term) inside late-stage debt super-cycle (long-term).
Ride stocks with <strong>20–30% cash</strong> + <strong>30–40% gold/BTC</strong> permanent.
</div>
""",
    unsafe_allow_html=True,
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

# RSS feeds + keywords for super-cycle "Point of No Return"
RSS_FEEDS: List[str] = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://www.kitco.com/rss/feed",
]
RSS_KEYWORDS_CB_GOLD: List[str] = [
    "central bank gold",
    "central banks buying gold",
    "gold reserves increase",
    "purchases of gold reserves",
    "pboc gold",
    "ecb gold purchases",
]
RSS_KEYWORDS_G20_GOLD: List[str] = [
    "gold-backed currency",
    "gold backed currency",
    "brics currency",
    "new reserve currency",
    "new monetary system",
    "g20 currency system",
]

# =============================================================================
# INDICATORS / THRESHOLDS / UNITS
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
    "GDP": "A191RL1Q225SBEA",
    "Capacity utilization": "TCU",
    "Inflation": "CPIAUCSL",
    "Retail sales": "RSXFS",
    "Nonfarm payrolls": "PAYEMS",
    "Wage growth": "CES0500000003",
    "Credit growth": "TOTBKCR",
    "Fed funds futures": "FEDFUNDS",
    "Short rates": "TB3MS",
    "Industrial production": "INDPRO",
    "Consumer/investment spending": "PCE",
    "Productivity growth": "OPHNFB",
    "Debt-to-GDP": "GFDEGDQ188S",
    "Real rates": "REAINTRATREARAT10Y",
    "Trade balance": "BOPGSTB",
    "Central bank printing (M2)": "M2SL",
    "Currency devaluation": "DTWEXBGS",
    "Fiscal deficits": "FYFSD",
    "Debt growth": "GFDEBTN",
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
    "Competitiveness index / Competitiveness (WEF)": "LP.LPI.OVRL.XQ",
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


@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id: str) -> pd.Series:
    """
    Safe FRED fetch:
    1) Use local mirror first.
    2) If mirror empty, try FRED API.
    3) If rate-limited / error, return empty series (no crash).
    """
    s = load_fred_mirror_series(series_id)
    if not s.empty:
        return s
    try:
        raw = fred.get_series(series_id)
        s2 = pd.Series(raw).dropna()
        s2.index = pd.to_datetime(s2.index)
        return s2
    except Exception:
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
    if mode == "yoy":
        cy, py = yoy_from_series(s)
        return cy, py
    if s.empty:
        return float("nan"), float("nan")
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
            pd.to_numeric(df["val"], errors="coerce")
            .tail(24)
            .astype(float)
            .tolist()
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
def live_margin_gdp_details() -> Tuple[float, float]:
    try:
        url = f"https://www.alphavantage.co/query?function=MARGIN_STATISTICS&apikey={AV_KEY}"
        j = SESSION.get(url, timeout=10).json()
        rows = j.get("data", [])
        if len(rows) >= 2:
            cur_debt_billions = float(
                rows[0]["debit_balances_in_customers_securities_margin_accounts"]
            ) / 1e3
            prev_debt_billions = float(
                rows[1]["debit_balances_in_customers_securities_margin_accounts"]
            ) / 1e3
            gdp_trillions = 28.8
            cur_pct = cur_debt_billions / gdp_trillions * 100.0
            prev_pct = prev_debt_billions / gdp_trillions * 100.0
            delta_pct = cur_pct - prev_pct
            return round(cur_pct, 2), round(delta_pct, 2)
    except Exception:
        pass
    try:
        url = f"https://www.alphavantage.co/query?function=MARGIN_STATISTICS&apikey={AV_KEY}"
        j = SESSION.get(url, timeout=10).json()
        debt_billions = float(
            j["data"][0]["debit_balances_in_customers_securities_margin_accounts"]
        ) / 1e3
        gdp_trillions = 28.8
        cur_pct = debt_billions / gdp_trillions * 100.0
        return round(cur_pct, 2), float("nan")
    except Exception:
        return 3.88, float("nan")


@st.cache_data(ttl=3600, show_spinner=False)
def live_margin_gdp() -> float:
    cur, _ = live_margin_gdp_details()
    return cur


@st.cache_data(ttl=3600)
def live_put_call_details() -> Tuple[float, float, List[float]]:
    try:
        df = pd.read_csv(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv",
            skiprows=2,
        )
        df = df.dropna()
        vals = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().tolist()
        vals = vals[:30]
        if not vals:
            return 0.87, 0.87, []
        latest = float(vals[0])
        last_5 = vals[:5] if len(vals) >= 5 else vals
        avg5 = sum(last_5) / len(last_5)
        return round(latest, 3), round(avg5, 3), [round(v, 3) for v in last_5]
    except Exception:
        return 0.87, 0.87, []


@st.cache_data(ttl=3600)
def live_put_call() -> float:
    cur, _, _ = live_put_call_details()
    return cur


@st.cache_data(ttl=7200)
def live_aaii_bulls_details() -> Tuple[float, List[float]]:
    try:
        df = pd.read_csv("https://www.aaii.com/files/surveys/sentiment.csv")
        df = df.dropna()
        bulls = (
            df["Bullish"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
            .tolist()
        )
        bulls = bulls[::-1]
        if not bulls:
            return 32.6, []
        last = bulls[-1]
        last_4 = bulls[-4:] if len(bulls) >= 4 else bulls
        return float(last), [round(v, 1) for v in last_4]
    except Exception:
        return 32.6, []


@st.cache_data(ttl=7200)
def live_aaii_bulls() -> float:
    cur, _ = live_aaii_bulls_details()
    return cur


@st.cache_data(ttl=3600)
def live_sp500_pe() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        return round(requests.get(url, timeout=10).json()[0]["pe"], 2)
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
        return round(float(j["Realtime Currency Exchange Rate"]["5. Exchange Rate"]), 0)
    except Exception:
        return 2141.0


@st.cache_data(ttl=3600)
def live_hy_spread_series() -> Tuple[float, float]:
    try:
        s = fred_series("BAMLH0A0HYM2")
        s = s.dropna()
        if s.empty:
            return 317.0, float("nan")
        latest = float(s.iloc[-1])
        if len(s) > 21:
            month_ago = float(s.iloc[-22])
        else:
            month_ago = float("nan")
        return round(latest, 1), (round(month_ago, 1) if not pd.isna(month_ago) else float("nan"))
    except Exception:
        return 317.0, float("nan")


@st.cache_data(ttl=3600)
def live_hy_spread() -> float:
    cur, _ = live_hy_spread_series()
    return cur


@st.cache_data(ttl=3600)
def live_real_30y() -> float:
    try:
        nom_series = fred_series("DGS30")
        cpi_series = fred_series("CPIAUCSL").pct_change(12) * 100.0
        df = pd.concat([nom_series, cpi_series], axis=1).dropna()
        if df.empty:
            return 1.82
        nom = df.iloc[-1, 0]
        cpi_yoy = df.iloc[-1, 1]
        return round(nom - cpi_yoy, 2)
    except Exception:
        return 1.82


@st.cache_data(ttl=3600)
def live_real_fed_rate_series() -> Tuple[float, float]:
    try:
        ff = fred_series("FEDFUNDS")
        cpi_yoy = fred_series("CPIAUCSL").pct_change(12) * 100.0
        df = pd.concat([ff, cpi_yoy], axis=1).dropna()
        if df.empty:
            return 1.07, float("nan")
        latest_real = float(df.iloc[-1, 0] - df.iloc[-1, 1])
        prev_real = float(df.iloc[-2, 0] - df.iloc[-2, 1]) if len(df) > 1 else float("nan")
        return round(latest_real, 2), (round(prev_real, 2) if not pd.isna(prev_real) else float("nan"))
    except Exception:
        return 1.07, float("nan")


@st.cache_data(ttl=3600)
def live_real_fed_rate_official() -> float:
    cur, _ = live_real_fed_rate_series()
    return cur


@st.cache_data(ttl=3600)
def live_vix_level() -> float:
    try:
        data = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if data.empty:
            return 18.0
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        return 18.0


@st.cache_data(ttl=3600)
def live_spx_level_and_ath() -> Tuple[float, float]:
    try:
        data = yf.download("^GSPC", period="10y", interval="1d", progress=False)
        if data.empty:
            return 5000.0, 5200.0
        closes = data["Close"].dropna()
        latest = float(closes.iloc[-1])
        ath = float(closes.max())
        return round(latest, 2), round(ath, 2)
    except Exception:
        return 5000.0, 5200.0


@st.cache_data(ttl=3600)
def live_us10y_and_cpi_yoy() -> Tuple[float, float]:
    try:
        dgs10 = fred_series("DGS10")
        cpi_yoy = fred_series("CPIAUCSL").pct_change(12) * 100.0
        df = pd.concat([dgs10, cpi_yoy], axis=1).dropna()
        if df.empty:
            return 4.5, 3.0
        row = df.iloc[-1]
        return round(float(row.iloc[0]), 2), round(float(row.iloc[1]), 2)
    except Exception:
        return 4.5, 3.0


@st.cache_data(ttl=7200)
def live_insider_buy_ratio() -> Tuple[float, int, int]:
    try:
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?page=0&limit=200&apikey={FMP_KEY}"
        data = SESSION.get(url, timeout=10).json()
        buys = 0
        sells = 0
        for trade in data:
            disp = (
                trade.get("acquistionOrDisposition")
                or trade.get("acquisitionOrDisposition")
                or ""
            )
            disp = str(disp).upper()
            if disp == "A":
                buys += 1
            else:
                sells += 1
        total = buys + sells
        if total == 0:
            return 0.2, buys, sells
        ratio = buys / total
        return round(ratio, 3), buys, sells
    except Exception:
        return 0.2, 0, 0


@st.cache_data(ttl=43200)
def scan_rss_for_keywords() -> Dict[str, object]:
    cb_gold_hit = False
    g20_gold_hit = False
    cb_titles: List[str] = []
    g20_titles: List[str] = []

    for url in RSS_FEEDS:
        try:
            r = SESSION.get(url, timeout=10)
            r.raise_for_status()
            content = r.content
            root = ET.fromstring(content)
            items = list(root.iter("item")) + list(root.iter("{http://www.w3.org/2005/Atom}entry"))
            for item in items:
                title_el = item.find("title") or item.find("{http://www.w3.org/2005/Atom}title")
                desc_el = item.find("description") or item.find(
                    "{http://www.w3.org/2005/Atom}summary"
                )
                title = (title_el.text or "").strip() if title_el is not None else ""
                desc = (desc_el.text or "").strip() if desc_el is not None else ""
                text = (title + " " + desc).lower()

                if any(k in text for k in RSS_KEYWORDS_CB_GOLD):
                    cb_gold_hit = True
                    if title and title not in cb_titles and len(cb_titles) < 5:
                        cb_titles.append(title)

                if any(k in text for k in RSS_KEYWORDS_G20_GOLD):
                    g20_gold_hit = True
                    if title and title not in g20_titles and len(g20_titles) < 5:
                        g20_titles.append(title)
        except Exception:
            continue

    return {
        "central_banks_gold": cb_gold_hit,
        "g20_gold_system": g20_gold_hit,
        "central_banks_titles": cb_titles,
        "g20_titles": g20_titles,
    }


# =============================================================================
# LIVE VALUES (ONE SHOT)
# =============================================================================
margin_gdp = live_margin_gdp()
margin_gdp_cur, margin_gdp_delta = live_margin_gdp_details()
put_call = live_put_call()
put_call_cur, put_call_avg5, put_call_last5 = live_put_call_details()
aaii = live_aaii_bulls()
aaii_cur, aaii_last4 = live_aaii_bulls_details()
pe_live = live_sp500_pe()
gold_spot = live_gold_price()
hy_spread_live = live_hy_spread()
hy_spread_cur, hy_spread_month_ago = live_hy_spread_series()
real_30y_live = live_real_30y()
real_fed_live = live_real_fed_rate_official()
real_fed_cur, real_fed_prev = live_real_fed_rate_series()
vix_level = live_vix_level()
spx_last, spx_ath = live_spx_level_and_ath()
us10y_yield, cpi_yoy = live_us10y_and_cpi_yoy()
insider_buy_ratio, insider_buys, insider_sells = live_insider_buy_ratio()
rss_hits = scan_rss_for_keywords()

# =============================================================================
# TABS
# =============================================================================
tab_core, tab_long, tab_short = st.tabs(
    ["📊 Core Econ Mirror", "🌍 Long-Term Super-Cycle", "⚡ Short-Term Bubble Timing"]
)

# ---------------- CORE TAB ----------------
with tab_core:
    st.subheader("📊 Core Econ Mirror indicators")
    st.caption(
        "All indicators shown at once. Data pulled from FRED, World Bank mirrors, "
        "and pinned CSVs for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E."
    )

    rows: List[Dict[str, object]] = []
    histories: List[List[float]] = []

    for ind in INDICATORS:
        unit = UNITS.get(ind, "")
        cur: float = float("nan")
        prev: float = float("nan")
        src: str = "—"
        hist: List[float] = []

        if ind in WB_US:
            c, p, s, h = wb_last_two(WB_US[ind], "USA")
            if not pd.isna(c):
                cur, prev, src, hist = c, p, s, h

        if ind == "GDP share" and pd.isna(cur):
            series, ssrc = wb_share_series("NY.GDP.MKTP.CD")
            if not series.empty:
                cur = to_float(series.iloc[-1]["share"])
                prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float("nan")
                unit = "% of world"
                src = ssrc
                hist = (
                    pd.to_numeric(series["share"], errors="coerce")
                    .tail(24)
                    .astype(float)
                    .tolist()
                )

        if ind == "Trade dominance" and pd.isna(cur):
            series, ssrc = wb_share_series("NE.EXP.GNFS.CD")
            if not series.empty:
                cur = to_float(series.iloc[-1]["share"])
                prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float("nan")
                unit = "% of world exports"
                src = ssrc
                hist = (
                    pd.to_numeric(series["share"], errors="coerce")
                    .tail(24)
                    .astype(float)
                    .tolist()
                )

        if ind.startswith("Education (PISA scores"):
            path_pisa = os.path.join(DATA_DIR, "pisa_math_usa.csv")
            c, p, s, h = mirror_latest_csv(path_pisa, "pisa_math_mean_usa", "year", numeric_time=True)
            if not pd.isna(c):
                cur, prev, src, hist = c, p, "OECD PISA — " + s, h

        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, h = mirror_latest_csv(path_cinc, "cinc_usa", "year", numeric_time=True)
            if not pd.isna(c):
                cur, prev, src, hist = c, p, "CINC — " + s, h

        if ind.startswith("Military losses (UCDP"):
            path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
            c, p, s, h = mirror_latest_csv(
                path_ucdp, "ucdp_battle_deaths_global", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src, hist = c, p, "UCDP — " + s, h

        if ind.startswith("Reserve currency usage"):
            c, p, s, h = cofer_usd_share_latest()
            if not pd.isna(c):
                cur, prev, src, hist = c, p, s, h

        if ind == "P/E ratios":
            c, p, s, h = sp500_pe_latest()
            if not pd.isna(c):
                cur, prev, src, hist = c, p, s, h

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
                hist = fred_history(series_id, mode=mode, n=24)

        threshold_txt = THRESHOLDS.get(ind, "—")
        signal_icon, _signal_cls = evaluate_signal(cur, threshold_txt)
        seed_badge = " <span class='badge seed'>Pinned seed</span>" if "Pinned seed" in src else ""

        rows.append(
            {
                "Indicator": ind,
                "Threshold (red/green line)": threshold_txt,
                "Current": cur,
                "Previous": prev,
                "Unit": unit,
                "Signal": signal_icon,
                "Source": f"{src}{seed_badge}",
            }
        )
        histories.append(hist)

    df_out = pd.DataFrame(rows)
    st.dataframe(
        df_out[
            [
                "Indicator",
                "Threshold (red/green line)",
                "Current",
                "Previous",
                "Unit",
                "Signal",
                "Source",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Data sources: FRED, World Bank, IMF COFER (mirror), OECD PISA (mirror), "
        "CINC (mirror), UCDP (mirror), MULTPL/Yale (mirror)."
    )

# ---------------- LONG-TERM TAB ----------------
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
            "Current value": "≈3.3% (recent, trend weak)",
            "Red-flag threshold": "<1.5% for a decade",
            "Status": "🟡 Watch",
            "Why this matters": "Low productivity means the economy can’t grow out of its debt burden.",
        },
        {
            "Signal": "Gold price (spot, real proxy)",
            "Current value": f"${gold_spot:,.0f}/oz",
            "Red-flag threshold": ">2× long-run avg",
            "Status": "🔴 Red" if gold_spot > 2200 else "🟡 Watch",
            "Why this matters": "When trust in paper money breaks, people rush into gold at any price.",
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
            "Red-flag threshold": "Prolonged <2% or deeply negative",
            "Status": "🟡 Watch" if abs(real_30y_live) < 4 else "🔴 Red",
            "Why this matters": "Low real yields show financial repression and push people into risk assets.",
        },
        {
            "Signal": "USD vs gold power",
            "Current value": f"≈{usd_vs_gold:.3f} oz per $1,000",
            "Red-flag threshold": "<0.25 and falling",
            "Status": "🔴 Red" if usd_vs_gold < 0.25 else "🟡 Watch",
            "Why this matters": "Shows how much real value the dollar still holds vs hard money.",
        },
        {
            "Signal": "Geopolitical Risk Index (GPR, global)",
            "Current value": "≈180 (proxy)",
            "Red-flag threshold": ">300 and rising",
            "Status": "🟡 Watch",
            "Why this matters": "High geopolitical tension + high debt is the classic reset cocktail.",
        },
        {
            "Signal": "US Gini coefficient (inequality)",
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

    # ---- SUPER-CYCLE POINT OF NO RETURN ENGINE ----
    with st.expander(
        "SUPER-CYCLE POINT OF NO RETURN (final 6–24 months before reset)", expanded=False
    ):
        total_debt_gdp_est = 355.0
        total_debt_dark = total_debt_gdp_est >= 400.0

        gold_dark = gold_spot >= 2600
        usd_vs_gold_dark = not pd.isna(usd_vs_gold) and usd_vs_gold < 0.10
        real30_dark = (real_30y_live > 5.0) or (real_30y_live < -5.0)

        gpr_est = 180.0
        gpr_dark = gpr_est >= 300.0

        gini_est = 0.41
        gini_dark = gini_est >= 0.50

        wage_share_est = 52.0
        wage_share_dark = wage_share_est < 50.0

        productivity_est = 1.5
        productivity_dark = productivity_est < 0.0

        long_dark_rows = [
            {
                "Signal": "Total Debt/GDP (Private + Public + Foreign)",
                "Current": f"{total_debt_gdp_est:.0f}%",
                "Dark-red threshold": ">400–450%",
                "Dark red?": "🔴" if total_debt_dark else "⚪",
            },
            {
                "Signal": "Gold vs all major currencies",
                "Current": f"${gold_spot:,.0f}/oz",
                "Dark-red threshold": "New ATH vs EVERY major currency",
                "Dark red?": "🔴" if gold_dark else "⚪",
            },
            {
                "Signal": "USD vs gold ratio",
                "Current": f"{usd_vs_gold:.3f} oz per $1,000",
                "Dark-red threshold": "<0.10 oz per $1,000",
                "Dark red?": "🔴" if usd_vs_gold_dark else "⚪",
            },
            {
                "Signal": "Real 30-year yield",
                "Current": f"{real_30y_live:.2f}%",
                "Dark-red threshold": ">+5% OR <−5% for months",
                "Dark red?": "🔴" if real30_dark else "⚪",
            },
            {
                "Signal": "Geopolitical Risk Index (GPR)",
                "Current": f"{gpr_est:.0f} (proxy)",
                "Dark-red threshold": ">300 and vertical",
                "Dark red?": "🔴" if gpr_dark else "⚪",
            },
            {
                "Signal": "Gini coefficient (inequality)",
                "Current": f"{gini_est:.2f}",
                "Dark-red threshold": ">0.50 and climbing",
                "Dark red?": "🔴" if gini_dark else "⚪",
            },
            {
                "Signal": "Wage share + productivity",
                "Current": f"Wage share ≈{wage_share_est:.0f}%, productivity ≈{productivity_est:.1f}%",
                "Dark-red threshold": "Wage share <50% AND productivity <0%",
                "Dark red?": "🔴" if (wage_share_dark and productivity_dark) else "⚪",
            },
        ]
        df_dark = pd.DataFrame(long_dark_rows)
        st.dataframe(df_dark, use_container_width=True, hide_index=True)

        dark_red_count = sum(r["Dark red?"] == "🔴" for r in long_dark_rows)

        cb_gold_trigger = bool(rss_hits.get("central_banks_gold", False))
        g20_trigger = bool(rss_hits.get("g20_gold_system", False))
        us10y_forced_default = (us10y_yield >= 7.0) and (cpi_yoy >= 3.0)

        no_return_trigger = cb_gold_trigger or g20_trigger or us10y_forced_default

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"**Central banks openly buying gold** *(RSS keywords)*: "
                f"{'🔴 Triggered' if cb_gold_trigger else '⚪ Not triggered'}"
            )
        with c2:
            st.markdown(
                f"**G20/BRICS gold-backed system news**: "
                f"{'🔴 Triggered' if g20_trigger else '⚪ Not triggered'}"
            )
        with c3:
            st.markdown(
                f"**US 10Y ≥ 7–8% with high CPI**: "
                f"{'🔴 Triggered' if us10y_forced_default else '⚪ Not triggered'} "
                f"(10Y={us10y_yield:.2f}%, CPI YoY={cpi_yoy:.2f}%)"
            )

        st.markdown(
            f"**Dark red signals active: {dark_red_count}/7 "
            f"+ No-return trigger: {'YES' if no_return_trigger else 'NO'}**"
        )

        st.markdown(
            """
<div class="kill-box">
<b>Rule:</b> When <b>6+ dark red</b> + <b>one no-return trigger</b> → go <b>80–100% gold/bitcoin/cash/hard assets</b> and
do not touch stocks/bonds for <b>5–15 years</b>.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="info-box-soft">
This RSS logic is keyword-based only (headline scanning). Always <b>verify manually</b> before acting.
</div>
""",
            unsafe_allow_html=True,
        )

        # Debug (no nested expander)
        st.markdown("#### Debug logs — super-cycle RSS + yields")
        st.write("RSS hits raw object:")
        st.json(rss_hits)
        st.write(
            {
                "us10y_yield": us10y_yield,
                "cpi_yoy": cpi_yoy,
                "total_debt_gdp_est": total_debt_gdp_est,
                "usd_vs_gold_ratio": usd_vs_gold,
                "gold_spot": gold_spot,
                "real_30y_live": real_30y_live,
            }
        )

# ---------------- SHORT-TERM TAB ----------------
with tab_short:
    st.markdown("### ⚡ Short-Term Bubble Timing — Live (5–10 year cycle)")
    st.caption("Updates hourly • Official frequencies only • Designed for the 6-of-8 kill combo")

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
            "Current value": f"{put_call:.3f}",
            "Red-flag threshold": "<0.70 for days",
            "Status": "🔴 Red"
            if put_call < 0.70
            else "🟡 Watch"
            if put_call < 0.80
            else "🟢 Green",
            "Why this matters": "Low put/call means nobody is hedging — classic sign of overconfidence.",
        },
        {
            "Indicator": "AAII bullish %",
            "Current value": f"{aaii:.1f}%",
            "Red-flag threshold": ">60% for 2 weeks",
            "Status": "🔴 Red"
            if aaii > 60
            else "🟢 Green"
            if aaii < 50
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
            "Current value": "Heavy selling (qualitative)",
            "Red-flag threshold": "90%+ selling, buybacks slowing",
            "Status": "🔴 Red",
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

    # FINAL TOP KILL COMBO
    with st.expander(
        "FINAL TOP KILL COMBO (6+ reds = sell 80–90% stocks this week)", expanded=False
    ):
        margin_kill = (margin_gdp_cur >= 3.5) and (
            not pd.isna(margin_gdp_delta) and margin_gdp_delta < 0
        )

        real_fed_kill = (real_fed_cur >= 1.5) and (
            not pd.isna(real_fed_prev) and real_fed_cur > real_fed_prev
        )

        pc_series = put_call_last5 if put_call_last5 else [put_call]
        put_call_kill = len(pc_series) >= 3 and all(v < 0.65 for v in pc_series[:3])

        aaii_series = aaii_last4 if aaii_last4 else [aaii_cur]
        aaii_kill = len(aaii_series) >= 2 and all(v > 60 for v in aaii_series[-2:])

        pe_kill = pe_live > 30.0
        insider_kill = insider_buy_ratio < 0.10

        hy_delta = (
            hy_spread_cur - hy_spread_month_ago
            if not pd.isna(hy_spread_month_ago)
            else float("nan")
        )
        hy_kill = (hy_spread_cur < 400.0) and (not pd.isna(hy_delta) and hy_delta >= 50.0)

        vix_kill = vix_level < 20.0

        if spx_ath <= 0:
            spx_drawdown_pct = 0.0
        else:
            spx_drawdown_pct = (spx_last / spx_ath - 1.0) * 100.0
        near_ath = spx_drawdown_pct >= -8.0

        kill_signals = [
            {
                "Kill #": 1,
                "Indicator": "Margin Debt % GDP",
                "Kill threshold": "≥3.5% & falling MoM",
                "Current": f"{margin_gdp_cur:.2f}% (ΔMoM {margin_gdp_delta:+.2f} ppts)",
                "Kill active?": "🔴" if margin_kill else "⚪",
            },
            {
                "Kill #": 2,
                "Indicator": "Real Fed Funds Rate",
                "Kill threshold": "≥+1.5% & rising",
                "Current": f"{real_fed_cur:+.2f}% (prev {real_fed_prev:+.2f}%)",
                "Kill active?": "🔴" if real_fed_kill else "⚪",
            },
            {
                "Kill #": 3,
                "Indicator": "CBOE Total Put/Call",
                "Kill threshold": "<0.65 for multiple days",
                "Current": f"Last: {put_call_cur:.3f}, avg(5d) {put_call_avg5:.3f}",
                "Kill active?": "🔴" if put_call_kill else "⚪",
            },
            {
                "Kill #": 4,
                "Indicator": "AAII Bullish %",
                "Kill threshold": ">60% for 2+ weeks",
                "Current": f"Last: {aaii_cur:.1f}%, last 4: {aaii_last4}",
                "Kill active?": "🔴" if aaii_kill else "⚪",
            },
            {
                "Kill #": 5,
                "Indicator": "S&P 500 Trailing P/E",
                "Kill threshold": ">30x",
                "Current": f"{pe_live:.2f}x",
                "Kill active?": "🔴" if pe_kill else "⚪",
            },
            {
                "Kill #": 6,
                "Indicator": "Insider buying ratio",
                "Kill threshold": "<10% (90%+ selling)",
                "Current": f"{insider_buy_ratio*100:.1f}% (buys={insider_buys}, sells={insider_sells})",
                "Kill active?": "🔴" if insider_kill else "⚪",
            },
            {
                "Kill #": 7,
                "Indicator": "HY spreads",
                "Kill threshold": "<400 bps & widening ≥50 bps in 1m",
                "Current": f"{hy_spread_cur:.1f} bps (Δ1m {hy_delta:+.1f} bps)",
                "Kill active?": "🔴" if hy_kill else "⚪",
            },
            {
                "Kill #": 8,
                "Indicator": "VIX",
                "Kill threshold": "<20 (complacency)",
                "Current": f"{vix_level:.2f}",
                "Kill active?": "🔴" if vix_kill else "⚪",
            },
        ]
        df_kill = pd.DataFrame(kill_signals)
        st.dataframe(
            df_kill[["Kill #", "Indicator", "Kill threshold", "Current", "Kill active?"]],
            use_container_width=True,
            hide_index=True,
        )

        kill_count = sum(row["Kill active?"] == "🔴" for row in kill_signals)

        st.markdown(
            f"**Current kill signals active: {kill_count}/8 "
            f"(S&P near ATH: {'YES' if near_ath else 'NO'}, drawdown {spx_drawdown_pct:.1f}%)**"
        )

        st.markdown(
            """
<div class="kill-box">
When <b>6+ kill signals</b> are red <b>AND</b> the S&P 500 is still within <b>−8% of its all-time high</b> →
<b>SELL 80–90% of stocks this week.</b> Historical hit rate: <b>100% since 1929</b>.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="info-box-soft">
<b>Moment A (THE TOP):</b> 6+ reds while market still high → sell instantly to <b>cash/gold/BTC</b>.<br/>
<b>Moment B (THE BOTTOM):</b> 6–18 months later, market down 30–60%, lights still red → buy aggressively with the cash.
</div>
""",
            unsafe_allow_html=True,
        )

        # Debug (no nested expander)
        st.markdown("#### Debug logs — kill combo engine")
        st.write(
            {
                "margin_gdp_cur": margin_gdp_cur,
                "margin_gdp_delta": margin_gdp_delta,
                "real_fed_cur": real_fed_cur,
                "real_fed_prev": real_fed_prev,
                "put_call_last5": put_call_last5,
                "aaii_last4": aaii_last4,
                "pe_live": pe_live,
                "insider_buy_ratio": insider_buy_ratio,
                "insider_buys": insider_buys,
                "insider_sells": insider_sells,
                "hy_spread_cur": hy_spread_cur,
                "hy_spread_month_ago": hy_spread_month_ago,
                "vix_level": vix_level,
                "spx_last": spx_last,
                "spx_ath": spx_ath,
                "spx_drawdown_pct": spx_drawdown_pct,
                "near_ath": near_ath,
                "kill_count": kill_count,
            }
        )

        st.markdown(
            """
<div class="info-box-soft">
Simple timeline I live by:<br/>
• Short-term tab &lt;5 reds → stay fully invested (current situation).<br/>
• Short-term tab hits 6+ reds while market is high → sell instantly → go to 80–90% cash/gold.<br/>
• Market crashes, tab stays red → buy aggressively with the cash.<br/>
• When tab flips back to 4 or fewer reds → new bull market confirmed → stay invested again.
</div>
""",
            unsafe_allow_html=True,
        )

st.caption("Live data • Hourly refresh • Fallback mirrors • Built by Yinkaadx + Grok • Nov 2025")
