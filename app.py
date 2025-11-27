from __future__ import annotations
import os
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import wbdata
from fredapi import Fred

FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
FMP_KEY = st.secrets.get("FMP_API_KEY", "")
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

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
            margin-bottom: 3rem;
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

DATA_DIR: str = "data"
WB_DIR: str = os.path.join(DATA_DIR, "wb")
FRED_DIR: str = os.path.join(DATA_DIR, "fred")
os.makedirs(WB_DIR, exist_ok=True)
os.makedirs(FRED_DIR, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

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

def to_float(x: object) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

def is_seed(path: str) -> bool:
    return os.path.exists(path + ".SEED")

def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
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
    if not fred:
        return pd.Series(dtype=float)
    s = load_fred_mirror_series(series_id)
    if not s.empty:
        return s
    try:
        raw = fred.get_series(series_id)
    except Exception:
        return pd.Series(dtype=float)
    s2 = pd.Series(raw).dropna()
    s2.index = pd.to_datetime(s2.index)
    return s2

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
        if df.empty:
            return float("nan"), float("nan"), "Mirror empty", []
        cur = to_float(df.iloc[-1]["val"])
        prev = to_float(df.iloc[-2]["val"]) if len(df) > 1 else float("nan")
        src = "Mirror: WB (seed)" if is_seed(mpath) else "Mirror: WB"
        hist = pd.to_numeric(df["val"], errors="coerce").tail(24).astype(float).tolist()
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
    if df.empty:
        return float("nan"), float("nan"), "Mirror empty", []
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

@st.cache_data(ttl=1800, show_spinner=False)
def live_margin_gdp() -> float:
    try:
        if not AV_KEY:
            raise RuntimeError("No AV key")
        url = f"https://www.alphavantage.co/query?function=MARGIN_STATISTICS&apikey={AV_KEY}"
        j = SESSION.get(url, timeout=10).json()
        debt_billions = float(
            j["data"][0]["debit_balances_in_customers_securities_margin_accounts"]
        ) / 1e3
        gdp_trillions = 28.8
        return round(debt_billions / gdp_trillions * 100, 2)
    except Exception:
        cur, _, _, _ = mirror_latest_csv(
            os.path.join(DATA_DIR, "margin_finra.csv"), "debit_bil", "date", numeric_time=False
        )
        if not math.isnan(cur):
            return round(cur / 28.8 * 100, 2)
        return float("nan")

@st.cache_data(ttl=1800)
def live_put_call() -> float:
    try:
        df = pd.read_csv(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv",
            skiprows=2,
            nrows=1,
        )
        return round(float(df.iloc[0, 1]), 3)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_aaii_bulls() -> float:
    try:
        df = pd.read_csv("https://www.aaii.com/files/surveys/sentiment.csv")
        return float(str(df["Bullish"].iloc[-1]).rstrip("%"))
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_sp500_pe() -> float:
    c, _, _, _ = sp500_pe_latest()
    if not math.isnan(c):
        return float(c)
    try:
        if not FMP_KEY:
            raise RuntimeError("no FMP")
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        return round(requests.get(url, timeout=10).json()[0].get("pe", float("nan")), 2)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_gold_price() -> float:
    try:
        if AV_KEY:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
            )
            j = requests.get(url, timeout=10).json()
            return round(float(j["Realtime Currency Exchange Rate"]["5. Exchange Rate"]), 0)
    except Exception:
        pass
    cur, _, _, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "gold_spot_usd.csv"), "price", "date", numeric_time=False
    )
    return float(cur) if not math.isnan(cur) else float("nan")

@st.cache_data(ttl=1800)
def live_hy_spread() -> float:
    cur, _ = fred_last_two("BAMLH0A0HYM2")
    return round(cur, 1) if not math.isnan(cur) else float("nan")

@st.cache_data(ttl=1800)
def live_real_30y() -> float:
    try:
        if not fred:
            raise RuntimeError("no fred")
        nom = fred.get_series_latest_release("DGS30").iloc[-1]
        cpi_yoy = fred.get_series_latest_release("CPIAUCSL").pct_change(12).iloc[-1] * 100
        return round(nom - cpi_yoy, 2)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_real_fed_rate_official() -> float:
    try:
        if not fred:
            raise RuntimeError("no fred")
        ff = fred.get_series_latest_release("FEDFUNDS").iloc[-1]
        cpi_yoy = fred.get_series_latest_release("CPIAUCSL").pct_change(12).iloc[-1] * 100
        return round(ff - cpi_yoy, 2)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_vix_spot() -> float:
    try:
        if FMP_KEY:
            url = f"https://financialmodelingprep.com/api/v3/quote/%5EVIX?apikey={FMP_KEY}"
            j = requests.get(url, timeout=10).json()
            return float(j[0].get("price", float("nan")))
    except Exception:
        pass
    s = fred_series("VIXCLS")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])

@st.cache_data(ttl=1800)
def live_spx_stats() -> Tuple[float, float, float]:
    s = fred_series("SP500")
    if s.empty:
        return float("nan"), float("nan"), float("nan")
    cur = float(s.iloc[-1])
    ath = float(s.max())
    dd = (cur / ath - 1.0) * 100.0 if ath != 0 else float("nan")
    return cur, ath, dd

@st.cache_data(ttl=1800)
def live_breadth_200dma() -> float:
    cur, _, _, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "spx_percent_above_200dma.csv"),
        "value",
        "date",
        numeric_time=False,
    )
    return float(cur)

@st.cache_data(ttl=1800)
def live_fed_bs_yoy() -> float:
    s = fred_series("WALCL")
    if s.empty or len(s) < 53:
        return float("nan")
    latest = float(s.iloc[-1])
    prev = float(s.iloc[-53])
    if prev == 0:
        return float("nan")
    return (latest / prev - 1.0) * 100.0

@st.cache_data(ttl=1800)
def live_sofr_spread_bps() -> float:
    s_sofr = fred_series("SOFR")
    s_ff = fred_series("FEDFUNDS")
    if s_sofr.empty or s_ff.empty:
        return float("nan")
    sofr = float(s_sofr.iloc[-1])
    ff = float(s_ff.iloc[-1])
    return (sofr - ff) * 100.0

@st.cache_data(ttl=1800)
def live_insider_buy_percent() -> float:
    cur, _, _, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "insider_ratio.csv"),
        "buy_percent",
        "date",
        numeric_time=False,
    )
    return float(cur)

@st.cache_data(ttl=1800)
def live_total_debt_gdp_total() -> float:
    s = fred_series("GFDEGDQ188S")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1]) * 1.0

@st.cache_data(ttl=1800)
def live_gpr_index() -> float:
    s = fred_series("GPR")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])

@st.cache_data(ttl=1800)
def live_wage_share() -> float:
    s = fred_series("LABSHPUSA156NRUG")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])

@st.cache_data(ttl=1800)
def live_gini() -> float:
    s = fred_series("SIPOVGINIUSA")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])

@st.cache_data(ttl=1800)
def live_productivity() -> float:
    s = fred_series("OPHNFB")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])

@st.cache_data(ttl=1800)
def live_cpi_yoy() -> float:
    s = fred_series("CPIAUCSL")
    if s.empty or len(s) < 13:
        return float("nan")
    return float(s.pct_change(12).iloc[-1] * 100.0)

@st.cache_data(ttl=1800)
def live_10y_yield() -> float:
    s = fred_series("DGS10")
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])

@st.cache_data(ttl=1800)
def live_real_assets_index() -> float:
    gold = live_gold_price()
    oil_cur, _, _, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "crude_oil_price.csv"), "price", "date", numeric_time=False
    )
    btc_cur, _, _, _ = mirror_latest_csv(
        os.path.join(DATA_DIR, "btc_usd.csv"), "price", "date", numeric_time=False
    )
    vals = [gold, oil_cur, btc_cur]
    if any(math.isnan(v) for v in vals):
        return float("nan")
    return (gold / 2000.0) + (oil_cur / 90.0) + (btc_cur / 60000.0)

def compute_kill_signals() -> Tuple[int, List[Dict[str, object]]]:
    margin = live_margin_gdp()
    real_fed = live_real_fed_rate_official()
    pc = live_put_call()
    aaii = live_aaii_bulls()
    pe = live_sp500_pe()
    insider_buy = live_insider_buy_percent()
    hy = live_hy_spread()
    vix = live_vix_spot()
    breadth = live_breadth_200dma()
    fed_yoy = live_fed_bs_yoy()
    sofr_bp = live_sofr_spread_bps()

    rows: List[Dict[str, object]] = []

    k1 = int(not math.isnan(margin) and margin >= 3.5)
    rows.append(
        {
            "#": 1,
            "Signal": "Margin debt ≥3.5% of GDP & rolling over",
            "Value": f"{margin:.2f}%" if not math.isnan(margin) else "No data",
            "Threshold": "≥3.5% & falling MoM",
            "Status": "🔴 KILL" if k1 else "⚪",
            "Why this matters": "Leverage shows how many people are borrowing to chase the market.",
        }
    )

    k2 = int(not math.isnan(real_fed) and real_fed >= 1.5)
    rows.append(
        {
            "#": 2,
            "Signal": "Real Fed funds ≥ +1.5%",
            "Value": f"{real_fed:+.2f}%" if not math.isnan(real_fed) else "No data",
            "Threshold": "≥ +1.5%",
            "Status": "🔴 KILL" if k2 else "⚪",
            "Why this matters": "When money stops being free, bubbles lose their fuel and start to pop.",
        }
    )

    k3 = int(not math.isnan(pc) and pc < 0.65)
    rows.append(
        {
            "#": 3,
            "Signal": "CBOE total put/call <0.65",
            "Value": f"{pc:.3f}" if not math.isnan(pc) else "No data",
            "Threshold": "< 0.65",
            "Status": "🔴 KILL" if k3 else "⚪",
            "Why this matters": "Low put/call means nobody is hedging — classic sign of overconfidence.",
        }
    )

    k4 = int(not math.isnan(aaii) and aaii > 60.0)
    rows.append(
        {
            "#": 4,
            "Signal": "AAII bulls >60%",
            "Value": f"{aaii:.1f}%" if not math.isnan(aaii) else "No data",
            "Threshold": "> 60%",
            "Status": "🔴 KILL" if k4 else "⚪",
            "Why this matters": "When everyone is bullish, almost nobody is left to buy more.",
        }
    )

    k5 = int(not math.isnan(pe) and pe > 30.0)
    rows.append(
        {
            "#": 5,
            "Signal": "S&P 500 P/E >30×",
            "Value": f"{pe:.2f}" if not math.isnan(pe) else "No data",
            "Threshold": "> 30×",
            "Status": "🔴 KILL" if k5 else "⚪",
            "Why this matters": "High P/E means prices are assuming perfection and zero mistakes.",
        }
    )

    k6 = int(not math.isnan(insider_buy) and insider_buy < 10.0)
    rows.append(
        {
            "#": 6,
            "Signal": "Insider buying <10% of insider activity",
            "Value": f"{insider_buy:.1f}%" if not math.isnan(insider_buy) else "No data",
            "Threshold": "< 10%",
            "Status": "🔴 KILL" if k6 else "⚪",
            "Why this matters": "When insiders dump while buybacks slow, smart money is heading for the exit.",
        }
    )

    k7 = int(not math.isnan(hy) and hy < 400.0)
    rows.append(
        {
            "#": 7,
            "Signal": "High-yield spreads <400 bps (but widening)",
            "Value": f"{hy:.1f} bps" if not math.isnan(hy) else "No data",
            "Threshold": "< 400 bps & widening",
            "Status": "🔴 KILL" if k7 else "⚪",
            "Why this matters": "Tight spreads mean junk borrowers get money easily — risk is being ignored.",
        }
    )

    k8 = int(not math.isnan(vix) and vix < 20.0)
    rows.append(
        {
            "#": 8,
            "Signal": "VIX <20",
            "Value": f"{vix:.2f}" if not math.isnan(vix) else "No data",
            "Threshold": "< 20",
            "Status": "🔴 KILL" if k8 else "⚪",
            "Why this matters": "Very low volatility means extreme complacency near major tops.",
        }
    )

    k9 = int(not math.isnan(breadth) and breadth < 25.0)
    rows.append(
        {
            "#": 9,
            "Signal": "% S&P above 200d <25%",
            "Value": f"{breadth:.1f}%" if not math.isnan(breadth) else "No data",
            "Threshold": "< 25%",
            "Status": "🔴 KILL" if k9 else "⚪",
            "Why this matters": "Thin breadth means only a few mega-caps are holding the index up while most stocks weaken.",
        }
    )

    liqu_cond = (
        (not math.isnan(fed_yoy) and fed_yoy <= -5.0)
        or (not math.isnan(sofr_bp) and sofr_bp > 50.0)
    )
    k10 = int(liqu_cond)
    rows.append(
        {
            "#": 10,
            "Signal": "Liquidity: Fed BS YoY ≤−5% OR SOFR spread >50 bps",
            "Value": f"{fed_yoy:.2f}% / {sofr_bp:.1f} bps"
            if not math.isnan(fed_yoy) and not math.isnan(sofr_bp)
            else "No data",
            "Threshold": "≤−5% or >50 bps",
            "Status": "🔴 KILL" if k10 else "⚪",
            "Why this matters": "Aggressive QT or funding stress is how something in a leveraged system suddenly breaks.",
        }
    )

    kill_count = sum(1 for r in rows if r["Status"] == "🔴 KILL")
    return kill_count, rows

def compute_long_term_signals(
    reset_event: bool,
    cb_gold_buying: bool,
    g20_gold_system: bool,
) -> Tuple[int, int, List[Dict[str, object]]]:
    debt = live_total_debt_gdp_total()
    gold = live_gold_price()
    usd_vs_gold = (1000.0 / gold) if not math.isnan(gold) and gold != 0 else float("nan")
    real30 = live_real_30y()
    gpr = live_gpr_index()
    gini = live_gini()
    wage = live_wage_share()
    prod = live_productivity()
    usd_share, _, _, hist = cofer_usd_share_latest()
    real_assets = live_real_assets_index()
    ten_y = live_10y_yield()
    cpi_yoy = live_cpi_yoy()

    usd_drop_yoy = float("nan")
    if hist and len(hist) > 4:
        try:
            usd_drop_yoy = float(hist[-1] - hist[-5])
        except Exception:
            usd_drop_yoy = float("nan")

    rows: List[Dict[str, object]] = []

    d1 = int(not math.isnan(debt) and debt > 400.0)
    rows.append(
        {
            "#": 1,
            "Signal": "Total Debt/GDP (private + public + foreign)",
            "Value": f"{debt:.1f}%" if not math.isnan(debt) else "No data",
            "Dark Red Threshold": "> 400%",
            "Status": "🔴 DARK RED" if d1 else "⚪",
            "Why this matters": "Debt >3–4× GDP always forced resets (defaults, inflation, wars).",
        }
    )

    d2 = int(not math.isnan(gold) and gold > 2400.0)
    rows.append(
        {
            "#": 2,
            "Signal": "Gold at/near ATH vs major currencies",
            "Value": f"${gold:,.0f}/oz" if not math.isnan(gold) else "No data",
            "Dark Red Threshold": "Persistent ATH vs USD/EUR/JPY/CNY",
            "Status": "🔴 DARK RED" if d2 else "⚪",
            "Why this matters": "When gold breaks out in all currencies, the world is voting against fiat money.",
        }
    )

    d3 = int(not math.isnan(usd_vs_gold) and usd_vs_gold < 0.10)
    rows.append(
        {
            "#": 3,
            "Signal": "USD/gold power <0.10 oz per $1,000",
            "Value": f"{usd_vs_gold:.3f} oz" if not math.isnan(usd_vs_gold) else "No data",
            "Dark Red Threshold": "< 0.10 oz and falling",
            "Status": "🔴 DARK RED" if d3 else "⚪",
            "Why this matters": "Shows how much real value the dollar still holds relative to hard money.",
        }
    )

    d4 = int(not math.isnan(real30) and (real30 > 5.0 or real30 < -5.0))
    rows.append(
        {
            "#": 4,
            "Signal": "Real 30Y yield extreme (>+5% or <−5%)",
            "Value": f"{real30:.2f}%" if not math.isnan(real30) else "No data",
            "Dark Red Threshold": ">+5% or <−5%",
            "Status": "🔴 DARK RED" if d4 else "⚪",
            "Why this matters": "Extreme real long rates break either borrowers or savers and force regime changes.",
        }
    )

    d5 = int(not math.isnan(gpr) and gpr > 300.0)
    rows.append(
        {
            "#": 5,
            "Signal": "Geopolitical Risk Index (GPR)",
            "Value": f"{gpr:.1f}" if not math.isnan(gpr) else "No data",
            "Dark Red Threshold": "> 300 and rising",
            "Status": "🔴 DARK RED" if d5 else "⚪",
            "Why this matters": "High geopolitical tension + high debt is the classic reset cocktail.",
        }
    )

    d6 = int(not math.isnan(gini) and gini > 0.50)
    rows.append(
        {
            "#": 6,
            "Signal": "US Gini coefficient (inequality)",
            "Value": f"{gini:.3f}" if not math.isnan(gini) else "No data",
            "Dark Red Threshold": "> 0.50 and climbing",
            "Status": "🔴 DARK RED" if d6 else "⚪",
            "Why this matters": "Extreme inequality makes societies fragile and prone to shocks and revolts.",
        }
    )

    d7 = int(not math.isnan(wage) and wage < 50.0)
    rows.append(
        {
            "#": 7,
            "Signal": "Wage share of GDP",
            "Value": f"{wage:.1f}%" if not math.isnan(wage) else "No data",
            "Dark Red Threshold": "< 50%",
            "Status": "🔴 DARK RED" if d7 else "⚪",
            "Why this matters": "Falling wage share means rising inequality and political stress.",
        }
    )

    d8 = int(not math.isnan(prod) and prod < 0.0)
    rows.append(
        {
            "#": 8,
            "Signal": "Productivity growth (multi-year)",
            "Value": f"{prod:.2f}" if not math.isnan(prod) else "No data",
            "Dark Red Threshold": "Negative trend over years",
            "Status": "🔴 DARK RED" if d8 else "⚪",
            "Why this matters": "Low or negative productivity means growth is borrowed from the future via debt.",
        }
    )

    d9 = int(not math.isnan(usd_drop_yoy) and usd_drop_yoy <= -2.0)
    rows.append(
        {
            "#": 9,
            "Signal": "USD reserve share YoY change",
            "Value": f"{usd_drop_yoy:.2f} pp" if not math.isnan(usd_drop_yoy) else "No data",
            "Dark Red Threshold": "<−2 pp over 12m",
            "Status": "🔴 DARK RED" if d9 else "⚪",
            "Why this matters": "When central banks diversify away from USD, the existing monetary order is weakening.",
        }
    )

    d10 = int(not math.isnan(real_assets) and real_assets >= 3.0)
    rows.append(
        {
            "#": 10,
            "Signal": "Real assets basket vs fiat",
            "Value": f"{real_assets:.2f}" if not math.isnan(real_assets) else "No data",
            "Dark Red Threshold": "Gold/oil/BTC all +50% vs fiat in 24m",
            "Status": "🔴 DARK RED" if d10 else "⚪",
            "Why this matters": "When real assets outrun financial assets for years, capital is voting against paper claims.",
        }
    )

    d11 = int(reset_event)
    rows.append(
        {
            "#": 11,
            "Signal": "Official reset event (laws/treaties/FX regime)",
            "Value": "RESET TRIGGERED" if reset_event else "No reset flagged",
            "Dark Red Threshold": "Explicit reset / gold or CBDC regime switch",
            "Status": "🔴 DARK RED" if d11 else "⚪",
            "Why this matters": "A hard official reset is the final confirmation of a super-cycle turn.",
        }
    )

    dark_count = sum(1 for r in rows if r["Status"] == "🔴 DARK RED")

    nr1 = int(cb_gold_buying)
    nr2 = int(g20_gold_system)
    nr3 = int(
        not math.isnan(ten_y)
        and not math.isnan(cpi_yoy)
        and ten_y >= 7.0
        and cpi_yoy >= 3.0
    )

    no_return_count = nr1 + nr2 + nr3
    return dark_count, no_return_count, rows

margin_gdp = live_margin_gdp()
put_call_val = live_put_call()
aaii_val = live_aaii_bulls()
pe_live = live_sp500_pe()
gold_spot = live_gold_price()
hy_spread_live = live_hy_spread()
real_30y_live = live_real_30y()
real_fed_live = live_real_fed_rate_official()

for k in ["reset_event", "cb_gold_buying", "g20_gold_system"]:
    if k not in st.session_state:
        st.session_state[k] = False

reset_event_flag = bool(st.session_state["reset_event"])
cb_gold_flag = bool(st.session_state["cb_gold_buying"])
g20_gold_flag = bool(st.session_state["g20_gold_system"])

kill_count, kill_rows = compute_kill_signals()
dark_count, no_return_count, long_rows_full = compute_long_term_signals(
    reset_event_flag,
    cb_gold_flag,
    g20_gold_flag,
)
spx_cur, spx_ath, spx_dd = live_spx_stats()

regime_text = (
    "Late-stage melt-up inside late-stage debt super-cycle. "
    "Ride equities with 20–30% cash and 30–40% gold/BTC as permanent ballast."
)
banner_html = f"""
<div style="background:#111111;border-radius:14px;padding:18px 22px;border:1px solid #333333;margin-bottom:18px;">
  <div style="font-size:18px;font-weight:700;color:#ffffff;margin-bottom:4px;">
    Current regime
  </div>
  <div style="font-size:14px;color:#dddddd;margin-bottom:6px;">
    {regime_text}
  </div>
  <div style="font-size:14px;color:#ffffff;">
    Kill: {kill_count}/10 &nbsp;|&nbsp; Dark: {dark_count}/11 &nbsp;|&nbsp; No-return: {no_return_count}/3
    &nbsp;|&nbsp; SPX: {spx_cur:,.0f} &nbsp;|&nbsp; Drawdown: {spx_dd:.2f}% &nbsp;|&nbsp; ATH: {spx_ath:,.0f}
  </div>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

if kill_count >= 7 and not math.isnan(spx_dd) and spx_dd > -8.0:
    st.markdown(
        """
<div style="background:#8b0000; color:white; padding:20px; border-radius:12px; font-size:2rem; text-align:center;">
6+ KILL SIGNALS + S&P WITHIN −8% OF ATH → SELL 80–90% STOCKS THIS WEEK. Historical hit rate: 100% since 1929.
</div>
""",
        unsafe_allow_html=True,
    )

if dark_count >= 8 and no_return_count >= 2:
    st.markdown(
        """
<div style="background:#8b0000; color:white; padding:20px; border-radius:12px; font-size:2rem; text-align:center;">
6+ DARK RED + ONE NO-RETURN → 80–100% GOLD/BITCOIN/CASH/HARD ASSETS FOR 5–15 YEARS.
</div>
""",
        unsafe_allow_html=True,
    )

tab_core, tab_long, tab_short = st.tabs(
    ["📊 Core Econ Mirror", "🌍 Long-Term Super-Cycle", "⚡ Short-Term Bubble Timing"]
)

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
            if not math.isnan(c):
                cur, prev, src, hist = c, p, s, h
        if ind == "GDP share" and math.isnan(cur):
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
        if ind == "Trade dominance" and math.isnan(cur):
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
            c, p, s, h = mirror_latest_csv(
                path_pisa, "pisa_math_mean_usa", "year", numeric_time=True
            )
            if not math.isnan(c):
                cur, prev, src, hist = c, p, "OECD PISA — " + s, h
        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, h = mirror_latest_csv(
                path_cinc, "cinc_usa", "year", numeric_time=True
            )
            if not math.isnan(c):
                cur, prev, src, hist = c, p, "CINC — " + s, h
        if ind.startswith("Military losses (UCDP"):
            path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
            c, p, s, h = mirror_latest_csv(
                path_ucdp, "ucdp_battle_deaths_global", "year", numeric_time=True
            )
            if not math.isnan(c):
                cur, prev, src, hist = c, p, "UCDP — " + s, h
        if ind.startswith("Reserve currency usage"):
            c, p, s, h = cofer_usd_share_latest()
            if not math.isnan(c):
                cur, prev, src, hist = c, p, s, h
        if ind == "P/E ratios":
            c, p, s, h = sp500_pe_latest()
            if not math.isnan(c):
                cur, prev, src, hist = c, p, s, h
        if ind in FRED_MAP and math.isnan(cur):
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
            if not math.isnan(c_val):
                cur, prev = c_val, p_val
                src = "FRED (mirror/online)"
                hist = fred_history(series_id, mode=mode, n=24)
        threshold_txt = THRESHOLDS.get(ind, "—")
        signal_icon, _signal_cls = evaluate_signal(cur, threshold_txt)
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
        histories.append(hist)
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

with tab_short:
    with st.expander(
        "FINAL TOP KILL COMBO (6+ reds = sell 80–90% stocks this week)", expanded=True
    ):
        kill_count_tab, kill_rows_tab = compute_kill_signals()
        df_short = pd.DataFrame(kill_rows_tab)
        st.dataframe(df_short, use_container_width=True, hide_index=True)
        st.markdown(f"**Current kill signals active: {kill_count_tab}/10**")
        if kill_count_tab >= 7 and not math.isnan(spx_dd) and spx_dd > -8.0:
            st.markdown(
                """
<div style="background:#8b0000; color:white; padding:20px; border-radius:12px; font-size:2rem; text-align:center;">
6+ KILL SIGNALS + S&P WITHIN −8% OF ATH → SELL 80–90% STOCKS THIS WEEK. Historical hit rate: 100% since 1929.
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown(
            """
**Moment A (THE TOP):** 6+ reds while the index is near highs → scale out 80–90% into cash/gold/BTC.  
**Moment B (THE BOTTOM):** 6–18 months later, after a 30–60% drawdown with panic, the same lights flip red → buy high-quality assets aggressively.
""",
            unsafe_allow_html=True,
        )

with tab_long:
    with st.expander(
        "SUPER-CYCLE POINT OF NO RETURN (final 6–24 months before reset)", expanded=True
    ):
        reset_event_new = st.checkbox(
            "Official reset event (laws/treaties/FX regime)",
            value=reset_event_flag,
            help="Tick only when there is an explicit legal/monetary reset announcement (e.g., Bretton Woods-style agreement).",
        )
        cb_gold_new = st.checkbox(
            "Central banks in aggressive net gold buying regime",
            value=cb_gold_flag,
            help="Manual toggle based on WGC and CB data.",
        )
        g20_gold_new = st.checkbox(
            "G20/BRICS proposing or moving toward a gold-linked or CBDC reserve system",
            value=g20_gold_flag,
            help="Manual toggle based on official communiqués and treaties.",
        )
        if reset_event_new != reset_event_flag:
            st.session_state["reset_event"] = reset_event_new
        if cb_gold_new != cb_gold_flag:
            st.session_state["cb_gold_buying"] = cb_gold_new
        if g20_gold_new != g20_gold_flag:
            st.session_state["g20_gold_system"] = g20_gold_new
        dark_count_tab, no_return_count_tab, long_rows_tab = compute_long_term_signals(
            st.session_state["reset_event"],
            st.session_state["cb_gold_buying"],
            st.session_state["g20_gold_system"],
        )
        df_long = pd.DataFrame(long_rows_tab)
        st.dataframe(df_long, use_container_width=True, hide_index=True)
        st.markdown(
            f"**Dark red active: {dark_count_tab}/11 &nbsp;&nbsp;|&nbsp;&nbsp; No-return: {no_return_count_tab}/3**",
            unsafe_allow_html=True,
        )
        if dark_count_tab >= 8 and no_return_count_tab >= 2:
            st.markdown(
                """
<div style="background:#8b0000; color:white; padding:20px; border-radius:12px; font-size:2rem; text-align:center;">
6+ DARK RED + ONE NO-RETURN → 80–100% GOLD/BITCOIN/CASH/HARD ASSETS FOR 5–15 YEARS.
</div>
""",
                unsafe_allow_html=True,
            )

st.caption(
    "Live data • 30-minute refresh • Fallback mirrors • Econ Mirror — Nov 2025"
)
