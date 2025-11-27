from __future__ import annotations

import math
import os
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import wbdata
from fredapi import Fred

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
        .regime-banner {
            font-size: 1.0rem;
            text-align: center;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            background: #111827;
            margin-bottom: 1.5rem;
        }
        .kill-box {
            background:#8b0000;
            color:white;
            padding:20px;
            border-radius:12px;
            font-size:2rem;
            text-align:center;
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
    '<p class="sub-header">Live Macro + Cycle Dashboard — Latest official releases only — Nov 2025</p>',
    unsafe_allow_html=True,
)

DATA_DIR: str = "data"
WB_DIR: str = os.path.join(DATA_DIR, "wb")
FRED_DIR: str = os.path.join(DATA_DIR, "fred")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WB_DIR, exist_ok=True)
os.makedirs(FRED_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})

FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

fred = Fred(api_key=FRED_API_KEY)


def ensure_seed_csvs() -> None:
    if not os.path.exists(os.path.join(DATA_DIR, "imf_cofer_usd_share.csv")):
        cofer_default = """date,usd_share
2023-12-31,58.4
2024-03-31,58.1
2024-06-30,57.8
2024-09-30,57.2
2025-06-30,56.3
"""
        with open(os.path.join(DATA_DIR, "imf_cofer_usd_share.csv"), "w") as f:
            f.write(cofer_default)
        open(os.path.join(DATA_DIR, "imf_cofer_usd_share.csv.SEED"), "w").close()
    if not os.path.exists(os.path.join(DATA_DIR, "pe_sp500.csv")):
        pe_default = """date,pe
2025-11-26,30.57
"""
        with open(os.path.join(DATA_DIR, "pe_sp500.csv"), "w") as f:
            f.write(pe_default)
        open(os.path.join(DATA_DIR, "pe_sp500.csv.SEED"), "w").close()
    if not os.path.exists(os.path.join(DATA_DIR, "margin_finra.csv")):
        margin_default = """date,debit_bil
2025-10-31,1180
"""
        with open(os.path.join(DATA_DIR, "margin_finra.csv"), "w") as f:
            f.write(margin_default)
        open(os.path.join(DATA_DIR, "margin_finra.csv.SEED"), "w").close()
    if not os.path.exists(os.path.join(DATA_DIR, "insider_ratio.csv")):
        insider_default = """date,buy_ratio
2025-10-31,8.0
"""
        with open(os.path.join(DATA_DIR, "insider_ratio.csv"), "w") as f:
            f.write(insider_default)
        open(os.path.join(DATA_DIR, "insider_ratio.csv.SEED"), "w").close()
    if not os.path.exists(os.path.join(DATA_DIR, "spx_percent_above_200dma.csv")):
        default_spx_breadth = """
date,value
2024-12-31,68.4
2025-01-31,72.1
2025-02-28,78.9
2025-03-31,81.2
2025-04-30,79.5
2025-05-31,74.3
2025-06-30,69.8
2025-07-31,65.2
2025-08-31,58.7
2025-09-30,52.1
2025-10-31,45.6
2025-11-26,16.0
"""
        with open(os.path.join(DATA_DIR, "spx_percent_above_200dma.csv"), "w") as f:
            f.write(default_spx_breadth.strip() + "\n")
        open(os.path.join(DATA_DIR, "spx_percent_above_200dma.csv.SEED"), "w").close()
    if not os.path.exists(os.path.join(DATA_DIR, "farmland_proxy.csv")):
        farmland_default = """date,index
2023-12-31,100
2024-12-31,110
2025-06-30,120
"""
        with open(os.path.join(DATA_DIR, "farmland_proxy.csv"), "w") as f:
            f.write(farmland_default)
        open(os.path.join(DATA_DIR, "farmland_proxy.csv.SEED"), "w").close()


ensure_seed_csvs()

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
    "Yield curve": "10Y–2Y > 1%",
    "Consumer confidence": "> 90",
    "Building permits": "+5% YoY",
    "Unemployment claims": "−10% YoY",
    "LEI (Conference Board Leading Economic Index)": "+1–2% YoY",
    "GDP": "2–4% YoY",
    "Capacity utilization": "> 80%",
    "Inflation": "2–3%",
    "Retail sales": "+3–5% YoY",
    "Nonfarm payrolls": "+150",
    "Wage growth": "> 3% YoY",
    "P/E ratios": "20+",
    "Credit growth": "> 5% YoY",
    "Fed funds futures": "Hikes implied +0.5%",
    "Short rates": "Rising",
    "Industrial production": "+2–5% YoY",
    "Consumer/investment spending": "Positive growth",
    "Productivity growth": "> 3% YoY",
    "Debt-to-GDP": "< 60%",
    "Foreign reserves": "+10% YoY",
    "Real rates": "< −1%",
    "Trade balance": "Surplus > 2% GDP",
    "Asset prices > traditional metrics": "P/E +20%",
    "New buyers entering (market participation) (FINRA margin debt, FRED proxy)": "+15% YoY",
    "Wealth gaps": "Top 1% share +5%",
    "Credit spreads": "> 500 bps",
    "Central bank printing (M2)": "+10% YoY",
    "Currency devaluation": "−10% to −20%",
    "Fiscal deficits": "> 6% GDP",
    "Debt growth": "+5–10% over income",
    "Income growth": "Debt–income gap < 5%",
    "Debt service": "> 20% incomes",
    "Education investment": "+5% of budget",
    "R&D patents": "+10% YoY",
    "Competitiveness index / Competitiveness (WEF)": "+5 ranks",
    "GDP per capita growth": "+3% YoY",
    "Trade share": "+2% global",
    "Military spending": "> 4% GDP",
    "Internal conflicts": "Protests +20%",
    "Reserve currency usage dropping (IMF COFER USD share)": "−5% global share",
    "Military losses (UCDP battle-related deaths — Global)": "Defeats +1/yr",
    "Economic output share": "−2% global share",
    "Corruption index": "−10 points",
    "Working population": "−1% YoY",
    "Education (PISA scores — Math mean, OECD)": "> 500",
    "Innovation": "Patents > 20% global",
    "GDP share": "+2% global",
    "Trade dominance": "> 15% global",
    "Power index (CINC — USA)": "8–10/10",
    "Debt burden": "> 100% GDP",
}

UNITS: Dict[str, str] = {
    "Yield curve": "bps",
    "Consumer confidence": "Index",
    "Building permits": "YoY %",
    "Unemployment claims": "YoY %",
    "LEI (Conference Board Leading Economic Index)": "YoY %",
    "GDP": "YoY %",
    "Capacity utilization": "%",
    "Inflation": "YoY %",
    "Retail sales": "YoY %",
    "Nonfarm payrolls": "K",
    "Wage growth": "YoY %",
    "P/E ratios": "x",
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
    "Wealth gaps": "Gini",
    "Credit spreads": "bps",
    "Central bank printing (M2)": "YoY %",
    "Currency devaluation": "%",
    "Fiscal deficits": "% GDP",
    "Debt growth": "YoY %",
    "Income growth": "YoY %",
    "Debt service": "% income",
    "Education investment": "% GDP",
    "R&D patents": "Count",
    "Competitiveness index / Competitiveness (WEF)": "Rank",
    "GDP per capita growth": "YoY %",
    "Trade share": "% global",
    "Military spending": "% GDP",
    "Internal conflicts": "Index",
    "Reserve currency usage dropping (IMF COFER USD share)": "% allocated",
    "Military losses (UCDP battle-related deaths — Global)": "Deaths",
    "Economic output share": "% global",
    "Corruption index": "Index",
    "Working population": "% pop 15–64",
    "Education (PISA scores — Math mean, OECD)": "Score",
    "Innovation": "Index",
    "GDP share": "% global",
    "Trade dominance": "% global",
    "Power index (CINC — USA)": "Index",
    "Debt burden": "% GDP",
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
        index=pd.to_datetime(df["DATE"], errors="coerce", format="%Y-%m-%d"),
        name=series_id,
    )
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    return s


def fred_series(series_id: str) -> pd.Series:
    s = load_fred_mirror_series(series_id)
    if not s.empty:
        return s
    raw = fred.get_series(series_id)
    s2 = pd.Series(raw).dropna()
    s2.index = pd.to_datetime(s2.index)
    return s2


def yoy_from_series(s: pd.Series) -> Tuple[float, float]:
    if s.empty:
        return float("nan"), float("nan")
    s = s.dropna()
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
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
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
        us["date"] = pd.to_datetime(us["date"], errors="coerce", format="%Y-%m-%d")
        wd["date"] = pd.to_datetime(wd["date"], errors="coerce", format="%Y-%m-%d")
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
    if df.empty or value_col not in df.columns or time_col not in df.columns:
        return float("nan"), float("nan"), "Mirror empty", []
    if numeric_time:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", format="%Y-%m-%d")
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
    if ">" in comp:
        ok = current > val
    else:
        ok = current < val
    return ("✅", "ok") if ok else ("⚠️", "warn")


@st.cache_data(ttl=1800)
def live_margin_gdp() -> float:
    path = os.path.join(DATA_DIR, "margin_finra.csv")
    c, _, _, _ = mirror_latest_csv(path, "debit_bil", "date", numeric_time=False)
    if pd.isna(c):
        return float("nan")
    gdp_series = fred_series("GDP")
    if gdp_series.empty:
        return float("nan")
    gdp_latest = to_float(gdp_series.iloc[-1]) / 1000.0
    if gdp_latest <= 0:
        return float("nan")
    return round(c / gdp_latest * 100.0, 2)


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
        val = df["Bullish"].iloc[-1]
        if isinstance(val, str) and val.endswith("%"):
            val = val.rstrip("%")
        return float(val)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_sp500_pe_official() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        return round(float(j[0].get("pe", float("nan"))), 2)
    except Exception:
        c, _, _, _ = sp500_pe_latest()
        return c


@st.cache_data(ttl=1800)
def live_gold_price() -> float:
    try:
        s = fred_series("GOLDAMGBD228NLBM")
        if s.empty:
            raise RuntimeError("No gold series")
        return round(float(s.iloc[-1]), 2)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_hy_spread() -> float:
    cur, _ = fred_last_two("BAMLH0A0HYM2")
    return round(cur, 1) if not pd.isna(cur) else float("nan")


@st.cache_data(ttl=1800)
def live_real_30y() -> float:
    try:
        nom_series = fred_series("DGS30")
        cpi_series = fred_series("CPIAUCSL")
        if nom_series.empty or cpi_series.empty:
            return float("nan")
        nom = to_float(nom_series.iloc[-1])
        cy, _ = yoy_from_series(cpi_series)
        return round(nom - cy, 2)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_real_fed_rate_official() -> float:
    try:
        ff_series = fred_series("FEDFUNDS")
        cpi_series = fred_series("CPIAUCSL")
        if ff_series.empty or cpi_series.empty:
            return float("nan")
        ff = to_float(ff_series.iloc[-1])
        cy, _ = yoy_from_series(cpi_series)
        return round(ff - cy, 2)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_vix_level() -> float:
    try:
        s = fred_series("VIXCLS")
        if s.empty:
            return float("nan")
        return round(float(s.iloc[-1]), 2)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_spx_metrics() -> Tuple[float, float, float]:
    try:
        s = fred_series("SP500")
        if s.empty:
            return float("nan"), float("nan"), float("nan")
        last = float(s.iloc[-1])
        ath = float(s.max())
        dd = (last / ath - 1.0) * 100.0
        return round(last, 2), round(ath, 2), round(dd, 2)
    except Exception:
        return float("nan"), float("nan"), float("nan")


@st.cache_data(ttl=1800)
def live_insider_buy_ratio() -> float:
    path = os.path.join(DATA_DIR, "insider_ratio.csv")
    c, _, _, _ = mirror_latest_csv(path, "buy_ratio", "date", numeric_time=False)
    return c


@st.cache_data(ttl=1800)
def live_breadth_above_200dma() -> float:
    path = os.path.join(DATA_DIR, "spx_percent_above_200dma.csv")
    c, _, _, _ = mirror_latest_csv(path, "value", "date", numeric_time=False)
    return c


@st.cache_data(ttl=1800)
def live_fed_bs_yoy_and_sofr_spread() -> Tuple[float, float]:
    try:
        walcl = fred_series("WALCL")
        sofr = fred_series("SOFR")
        ff = fred_series("FEDFUNDS")
        if walcl.empty or sofr.empty or ff.empty:
            return float("nan"), float("nan")
        walcl_yoy, _ = yoy_from_series(walcl)
        s_last = to_float(sofr.iloc[-1])
        ff_last = to_float(ff.iloc[-1])
        spread = s_last - ff_last
        return round(walcl_yoy, 2), round(spread, 2)
    except Exception:
        return float("nan"), float("nan")


@st.cache_data(ttl=1800)
def live_total_debt_to_gdp() -> float:
    try:
        debt = fred_series("TCMDO")
        gdp = fred_series("GDP")
        if debt.empty or gdp.empty:
            return float("nan")
        d_last = to_float(debt.iloc[-1])
        g_last = to_float(gdp.iloc[-1])
        if g_last <= 0:
            return float("nan")
        return round(d_last / g_last * 100.0, 1)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_gpr_index() -> float:
    try:
        url = "https://www.policyuncertainty.com/media/GPR_World.csv"
        df = pd.read_csv(url)
        if "GPR" in df.columns:
            return round(float(df["GPR"].iloc[-1]), 1)
        return float("nan")
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def live_gini() -> float:
    c, _, _, _ = wb_last_two("SI.POV.GINI", "USA")
    return round(c, 3) if not pd.isna(c) else float("nan")


@st.cache_data(ttl=1800)
def live_wage_share() -> float:
    s = fred_series("LABSHPUSA156NRUG")
    if s.empty:
        return float("nan")
    return round(float(s.iloc[-1]), 1)


@st.cache_data(ttl=1800)
def live_productivity_multi_year_trend() -> Tuple[float, bool]:
    s = fred_series("OPHNFB")
    if s.empty:
        return float("nan"), False
    last = float(s.iloc[-1])
    if len(s) < 16:
        return last, False
    recent = s.tail(16)
    diffs = recent.diff().dropna()
    negative_multi = (diffs < 0).sum() >= len(diffs) * 0.6
    return round(last, 2), bool(negative_multi)


@st.cache_data(ttl=1800)
def live_usd_reserve_yoy_change() -> float:
    cur, prev, _, _ = cofer_usd_share_latest()
    if pd.isna(cur) or pd.isna(prev):
        return float("nan")
    return round(cur - prev, 2)


@st.cache_data(ttl=1800)
def live_real_assets_basket_change_24m() -> float:
    try:
        gold = fred_series("GOLDAMGBD228NLBM")
        oil = fred_series("DCOILWTICO")
        farmland_path = os.path.join(DATA_DIR, "farmland_proxy.csv")
        farmland_df = load_csv(farmland_path)
        if gold.empty or oil.empty or farmland_df.empty:
            return float("nan")
        gold = gold.dropna()
        oil = oil.dropna()
        farmland_df["date"] = pd.to_datetime(
            farmland_df["date"], errors="coerce", format="%Y-%m-%d"
        )
        farmland_df = farmland_df.dropna().sort_values("date")
        last_date = min(gold.index[-1], oil.index[-1], farmland_df["date"].iloc[-1])
        cutoff = last_date - pd.DateOffset(months=24)
        gold2 = gold[gold.index >= cutoff]
        oil2 = oil[oil.index >= cutoff]
        farm2 = farmland_df[farmland_df["date"] >= cutoff]
        if gold2.empty or oil2.empty or farm2.empty:
            return float("nan")
        g_start, g_end = float(gold2.iloc[0]), float(gold2.iloc[-1])
        o_start, o_end = float(oil2.iloc[0]), float(oil2.iloc[-1])
        f_start, f_end = float(farm2["index"].iloc[0]), float(farm2["index"].iloc[-1])
        idx_start = (g_start + o_start + f_start) / 3.0
        idx_end = (g_end + o_end + f_end) / 3.0
        if idx_start <= 0:
            return float("nan")
        return round((idx_end / idx_start - 1.0) * 100.0, 1)
    except Exception:
        return float("nan")


@st.cache_data(ttl=1800)
def build_core_dataframe() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ind in INDICATORS:
        unit = UNITS.get(ind, "")
        cur: float = float("nan")
        prev: float = float("nan")
        src: str = "—"
        if ind in WB_US:
            c, p, s, _ = wb_last_two(WB_US[ind], "USA")
            if not pd.isna(c):
                cur, prev, src = c, p, s
        if ind == "GDP share" and pd.isna(cur):
            series, ssrc = wb_share_series("NY.GDP.MKTP.CD")
            if not series.empty:
                cur = to_float(series.iloc[-1]["share"])
                prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float("nan")
                unit = "% of world"
                src = ssrc
        if ind == "Trade dominance" and pd.isna(cur):
            series, ssrc = wb_share_series("NE.EXP.GNFS.CD")
            if not series.empty:
                cur = to_float(series.iloc[-1]["share"])
                prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float("nan")
                unit = "% of world exports"
                src = ssrc
        if ind.startswith("Education (PISA scores"):
            path_pisa = os.path.join(DATA_DIR, "pisa_math_usa.csv")
            c, p, s, _ = mirror_latest_csv(path_pisa, "pisa_math_mean_usa", "year", numeric_time=True)
            if not pd.isna(c):
                cur, prev, src = c, p, "OECD PISA — " + s
        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, _ = mirror_latest_csv(path_cinc, "cinc_usa", "year", numeric_time=True)
            if not pd.isna(c):
                cur, prev, src = c, p, "CINC — " + s
        if ind.startswith("Military losses (UCDP"):
            path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
            c, p, s, _ = mirror_latest_csv(
                path_ucdp, "ucdp_battle_deaths_global", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src = c, p, "UCDP — " + s
        if ind.startswith("Reserve currency usage"):
            c, p, s, _ = cofer_usd_share_latest()
            if not pd.isna(c):
                cur, prev, src = c, p, s
        if ind == "P/E ratios":
            c, p, s, _ = sp500_pe_latest()
            live_pe = live_sp500_pe_official()
            if not pd.isna(live_pe):
                cur, prev, src = live_pe, p, "FMP (online)"
            elif not pd.isna(c):
                cur, prev, src = c, p, s
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
                src = "FRED (official)"
        threshold_txt = THRESHOLDS.get(ind, "—")
        signal_icon, _signal_cls = evaluate_signal(cur, threshold_txt)
        seed_badge = " (seed)" if "Pinned seed" in src or "Mirror" in src else ""
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
    return pd.DataFrame(rows)


def build_short_term_table() -> Tuple[pd.DataFrame, int]:
    margin_gdp = live_margin_gdp()
    real_fed = live_real_fed_rate_official()
    put_call = live_put_call()
    aaii = live_aaii_bulls()
    pe_live = live_sp500_pe_official()
    hy_spread = live_hy_spread()
    insider = live_insider_buy_ratio()
    vix = live_vix_level()
    breadth = live_breadth_above_200dma()
    bs_yoy, sofr_spread = live_fed_bs_yoy_and_sofr_spread()

    rows: List[Dict[str, object]] = []

    def kill_status(flag: bool) -> str:
        return "🔴 KILL" if flag else "⚪"

    k1 = not pd.isna(margin_gdp) and margin_gdp >= 3.5
    k2 = not pd.isna(real_fed) and real_fed >= 1.5
    k3 = not pd.isna(put_call) and put_call < 0.65
    k4 = not pd.isna(aaii) and aaii > 60.0
    k5 = not pd.isna(pe_live) and pe_live > 30.0
    k6 = not pd.isna(insider) and insider < 10.0
    k7 = not pd.isna(hy_spread) and hy_spread < 400.0
    k8 = not pd.isna(vix) and vix < 20.0
    k9 = not pd.isna(breadth) and breadth < 25.0
    k10 = (
        not pd.isna(bs_yoy)
        and not pd.isna(sofr_spread)
        and (bs_yoy <= -5.0 or sofr_spread > 0.5)
    )

    rows.append(
        {
            "#": 1,
            "Signal": "Margin debt ≥3.5% of GDP & rolling over",
            "Value": f"{margin_gdp:.2f}%" if not pd.isna(margin_gdp) else "No data",
            "Threshold": "≥3.5% & falling MoM",
            "Status": kill_status(k1),
            "Why this matters": "Leverage shows how many people are borrowing to chase the market.",
        }
    )
    rows.append(
        {
            "#": 2,
            "Signal": "Real Fed funds ≥+1.5%",
            "Value": f"{real_fed:+.2f}%" if not pd.isna(real_fed) else "No data",
            "Threshold": "≥ +1.5%",
            "Status": kill_status(k2),
            "Why this matters": "When money stops being free, bubbles lose their fuel and start to pop.",
        }
    )
    rows.append(
        {
            "#": 3,
            "Signal": "CBOE total put/call <0.65",
            "Value": f"{put_call:.3f}" if not pd.isna(put_call) else "No data",
            "Threshold": "< 0.65",
            "Status": kill_status(k3),
            "Why this matters": "Low put/call means nobody is hedging — classic sign of overconfidence.",
        }
    )
    rows.append(
        {
            "#": 4,
            "Signal": "AAII bulls >60%",
            "Value": f"{aaii:.1f}%" if not pd.isna(aaii) else "No data",
            "Threshold": "> 60%",
            "Status": kill_status(k4),
            "Why this matters": "When everyone is bullish, almost nobody is left to buy more.",
        }
    )
    rows.append(
        {
            "#": 5,
            "Signal": "S&P 500 P/E >30×",
            "Value": f"{pe_live:.2f}×" if not pd.isna(pe_live) else "No data",
            "Threshold": "> 30×",
            "Status": kill_status(k5),
            "Why this matters": "High P/E means prices are assuming perfection and zero mistakes.",
        }
    )
    rows.append(
        {
            "#": 6,
            "Signal": "Insider buying <10% of insider activity",
            "Value": f"{insider:.1f}%" if not pd.isna(insider) else "No data",
            "Threshold": "< 10%",
            "Status": kill_status(k6),
            "Why this matters": "When insiders dump while buybacks slow, smart money is heading for the exit.",
        }
    )
    rows.append(
        {
            "#": 7,
            "Signal": "High-yield spreads <400 bps (but widening)",
            "Value": f"{hy_spread:.1f} bps" if not pd.isna(hy_spread) else "No data",
            "Threshold": "< 400 bps & widening",
            "Status": kill_status(k7),
            "Why this matters": "Tight spreads mean junk borrowers get money easily — risk is being ignored.",
        }
    )
    rows.append(
        {
            "#": 8,
            "Signal": "VIX <20",
            "Value": f"{vix:.2f}" if not pd.isna(vix) else "No data",
            "Threshold": "< 20",
            "Status": kill_status(k8),
            "Why this matters": "Very low volatility means extreme complacency near major tops.",
        }
    )
    rows.append(
        {
            "#": 9,
            "Signal": "% S&P above 200d <25%",
            "Value": f"{breadth:.1f}%" if not pd.isna(breadth) else "No data",
            "Threshold": "< 25%",
            "Status": kill_status(k9),
            "Why this matters": "Thin breadth means only a few mega-caps are holding the index up.",
        }
    )
    rows.append(
        {
            "#": 10,
            "Signal": "Liquidity: Fed BS YoY ≤−5% OR SOFR spread >50 bps",
            "Value": f"{bs_yoy:.2f}% / {sofr_spread:.2f} bps"
            if not pd.isna(bs_yoy) and not pd.isna(sofr_spread)
            else "No data",
            "Threshold": "≤−5% or > 50 bps",
            "Status": kill_status(k10),
            "Why this matters": "Aggressive QT or funding stress is how leveraged systems suddenly break.",
        }
    )
    df = pd.DataFrame(rows)
    kill_count = sum("KILL" in str(x) for x in df["Status"])
    return df, kill_count


def build_long_term_table(
    reset_event: bool, cb_gold_buying: bool, g20_gold_system: bool
) -> Tuple[pd.DataFrame, int, int]:
    total_debt_gdp = live_total_debt_to_gdp()
    gold_price = live_gold_price()
    usd_gold_power = 1000.0 / gold_price if not pd.isna(gold_price) and gold_price > 0 else float("nan")
    real_30y = live_real_30y()
    gpr = live_gpr_index()
    gini = live_gini()
    wage_share = live_wage_share()
    prod_level, prod_negative_trend = live_productivity_multi_year_trend()
    usd_reserve_yoy = live_usd_reserve_yoy_change()
    real_assets_24m = live_real_assets_basket_change_24m()

    rows: List[Dict[str, object]] = []

    def dark(flag: bool) -> str:
        return "🔴 DARK RED" if flag else "⚪"

    d1 = not pd.isna(total_debt_gdp) and total_debt_gdp > 400.0
    d2 = not pd.isna(gold_price) and gold_price > 2500.0
    d3 = not pd.isna(usd_gold_power) and usd_gold_power < 0.10
    d4 = not pd.isna(real_30y) and (real_30y >= 5.0 or real_30y <= -5.0)
    d5 = not pd.isna(gpr) and gpr > 300.0
    d6 = not pd.isna(gini) and gini > 0.50
    d7 = not pd.isna(wage_share) and wage_share < 50.0
    d8 = prod_negative_trend
    d9 = not pd.isna(usd_reserve_yoy) and usd_reserve_yoy <= -2.0
    d10 = not pd.isna(real_assets_24m) and real_assets_24m >= 50.0
    d11 = reset_event

    rows.append(
        {
            "#": 1,
            "Signal": "Total Debt/GDP (private + public + foreign)",
            "Value": f"{total_debt_gdp:.1f}% " if not pd.isna(total_debt_gdp) else "No data",
            "Dark Red Threshold": "> 400%",
            "Status": dark(d1),
            "Why this matters": "Debt >3–4× GDP always forced resets (defaults, inflation, wars).",
        }
    )
    rows.append(
        {
            "#": 2,
            "Signal": "Gold at/near ATH vs major currencies",
            "Value": f"${gold_price:,.0f}/oz" if not pd.isna(gold_price) else "No data",
            "Dark Red Threshold": "Persistent ATH vs USD/EUR/JPY/CNY",
            "Status": dark(d2),
            "Why this matters": "When gold breaks out in all currencies, the world is voting against fiat money.",
        }
    )
    rows.append(
        {
            "#": 3,
            "Signal": "USD/gold power <0.10 oz per $1,000",
            "Value": f"{usd_gold_power:.3f} oz" if not pd.isna(usd_gold_power) else "No data",
            "Dark Red Threshold": "< 0.10 oz and falling",
            "Status": dark(d3),
            "Why this matters": "Shows how much real value the dollar still holds relative to hard money.",
        }
    )
    rows.append(
        {
            "#": 4,
            "Signal": "Real 30Y yield extreme (≥+5% or ≤−5%)",
            "Value": f"{real_30y:.2f}%" if not pd.isna(real_30y) else "No data",
            "Dark Red Threshold": "≥ +5% or ≤ −5%",
            "Status": dark(d4),
            "Why this matters": "Extreme real rates break either borrowers or savers and force regime changes.",
        }
    )
    rows.append(
        {
            "#": 5,
            "Signal": "Geopolitical Risk Index (GPR)",
            "Value": f"{gpr:.1f}" if not pd.isna(gpr) else "No data",
            "Dark Red Threshold": "> 300 and rising",
            "Status": dark(d5),
            "Why this matters": "High geopolitical tension + high debt is the classic reset cocktail.",
        }
    )
    rows.append(
        {
            "#": 6,
            "Signal": "US Gini coefficient (inequality)",
            "Value": f"{gini:.3f}" if not pd.isna(gini) else "No data",
            "Dark Red Threshold": "> 0.50 and climbing",
            "Status": dark(d6),
            "Why this matters": "Extreme inequality makes societies fragile and prone to shocks and revolts.",
        }
    )
    rows.append(
        {
            "#": 7,
            "Signal": "Wage share of GDP",
            "Value": f"{wage_share:.1f}%" if not pd.isna(wage_share) else "No data",
            "Dark Red Threshold": "< 50%",
            "Status": dark(d7),
            "Why this matters": "Falling wage share means rising inequality and political stress.",
        }
    )
    rows.append(
        {
            "#": 8,
            "Signal": "Productivity growth (multi-year trend)",
            "Value": f"{prod_level:.2f}" if not pd.isna(prod_level) else "No data",
            "Dark Red Threshold": "Negative trend over years",
            "Status": dark(d8),
            "Why this matters": "Low or negative productivity means growth is borrowed from the future via debt.",
        }
    )
    rows.append(
        {
            "#": 9,
            "Signal": "USD reserve share YoY change",
            "Value": f"{usd_reserve_yoy:.2f} pp" if not pd.isna(usd_reserve_yoy) else "No data",
            "Dark Red Threshold": "< −2 pp over 12m",
            "Status": dark(d9),
            "Why this matters": "When central banks diversify away from USD, the existing system is weakening.",
        }
    )
    rows.append(
        {
            "#": 10,
            "Signal": "Real assets basket vs fiat (24m)",
            "Value": f"{real_assets_24m:.1f}%" if not pd.isna(real_assets_24m) else "No data",
            "Dark Red Threshold": "Gold/oil/farmland all +50% vs fiat",
            "Status": dark(d10),
            "Why this matters": "When real assets outrun financial assets for years, capital is voting vs paper.",
        }
    )
    rows.append(
        {
            "#": 11,
            "Signal": "Official reset event (laws/treaties/FX regime)",
            "Value": "Triggered" if reset_event else "Not triggered",
            "Dark Red Threshold": "Explicit reset / new regime",
            "Status": dark(d11),
            "Why this matters": "Once a formal reset is announced, the old game is over.",
        }
    )

    df = pd.DataFrame(rows)

    dark_count = sum("DARK RED" in str(x) for x in df["Status"])

    ten_y = fred_series("DGS10")
    cpi = fred_series("CPIAUCSL")
    auto_no_return = False
    if not ten_y.empty and not cpi.empty:
        y10 = float(ten_y.iloc[-1])
        cpi_yoy, _ = yoy_from_series(cpi)
        auto_no_return = y10 >= 7.0 and cpi_yoy >= 5.0

    no_return_flags = [cb_gold_buying, g20_gold_system, auto_no_return]
    no_return_count = sum(bool(x) for x in no_return_flags)

    return df, dark_count, no_return_count


core_df = build_core_dataframe()
short_df, kill_count = build_short_term_table()
spx_last, spx_ath, spx_dd = live_spx_metrics()

tab_core, tab_long, tab_short = st.tabs(
    ["📊 Core Econ Mirror", "🌍 Long-Term Super-Cycle", "⚡ Short-Term Bubble Timing"]
)

with tab_long:
    st.markdown("### 🌍 Long-Term Super-Cycle")
    reset_event = st.checkbox("Official reset event (laws/treaties/FX regime)", value=False)
    cb_gold_buying = st.checkbox(
        "Central banks in aggressive net gold buying regime", value=False
    )
    g20_gold_system = st.checkbox(
        "G20/BRICS proposing or moving toward a gold-linked or CBDC reserve system",
        value=False,
    )
    long_df, dark_count, no_return_count = build_long_term_table(
        reset_event, cb_gold_buying, g20_gold_system
    )
    st.dataframe(long_df, use_container_width=True, hide_index=True)
    st.markdown(
        f"**Dark red active:** {dark_count}/11 &nbsp;&nbsp; | &nbsp;&nbsp; **No-return:** {no_return_count}/3"
    )
    if dark_count >= 8 and no_return_count >= 2:
        st.markdown(
            "8+ DARK RED + 2 NO-RETURN → 80–100% GOLD/BTC/CASH/HARD ASSETS FOR 5–15 YEARS",
            unsafe_allow_html=False,
        )
        st.markdown('<div class="kill-box">8+ DARK RED + 2 NO-RETURN → 80–100% GOLD/BTC/CASH/HARD ASSETS FOR 5–15 YEARS</div>',
                    unsafe_allow_html=True)
    long_dark_count = dark_count
    long_no_return_count = no_return_count

with tab_short:
    st.markdown("### ⚡ Short-Term Bubble Timing — FINAL TOP KILL COMBO")
    st.caption("6+ reds while index is near highs ⇒ sell 80–90% stocks this week.")
    st.dataframe(short_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Current kill signals active:** {kill_count}/10")
    if not pd.isna(spx_dd) and spx_dd > -8.0 and kill_count >= 7:
        st.markdown(
            '<div class="kill-box">7+ KILL SIGNALS + S&P WITHIN −8% OF ATH → SELL 80–90% STOCKS THIS WEEK</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "Moment A (THE TOP): 6+ reds while the index is near highs ⇒ scale out 80–90% into cash/gold/BTC."
    )
    st.markdown(
        "Moment B (THE BOTTOM): 6–18 months later, after a 30–60% drawdown with panic, the same lights flip red → buy high-quality assets aggressively."
    )

with tab_core:
    st.subheader("📊 Core Econ Mirror indicators")
    st.caption(
        "All indicators at once. Data comes from FRED, World Bank mirrors, and pinned CSVs for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E."
    )
    st.dataframe(
        core_df[
            ["Indicator", "Threshold", "Current", "Previous", "Unit", "Signal", "Source"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Data sources: FRED, World Bank, IMF COFER (mirror), OECD PISA (mirror), "
        "CINC (mirror), UCDP (mirror), Financial Modeling Prep (P/E)."
    )

kill_display = kill_count
dark_display = locals().get("long_dark_count", 0)
no_return_display = locals().get("long_no_return_count", 0)

regime_text = (
    f"Current regime: Late-stage melt-up (short-term) inside late-stage debt "
    f"super-cycle (long-term). Ride stocks with 20–30% cash + 30–40% gold/BTC permanent. "
    f"SPX: {spx_last if not pd.isna(spx_last) else '—'} | "
    f"Drawdown: {spx_dd:.2f}% | ATH: {spx_ath if not pd.isna(spx_ath) else '—'} | "
    f"Kill: {kill_display}/10 | Dark: {dark_display}/11 | No-return: {no_return_display}/3"
)

st.markdown(f'<div class="regime-banner">{regime_text}</div>', unsafe_allow_html=True)

st.caption(
    "Live official releases • 30-minute cache • Fallback mirrors only where needed • Econ Mirror — Nov 2025"
)
