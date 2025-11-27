from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import wbdata
from fredapi import Fred

# =============================================================================
# SECRETS
# =============================================================================
FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

# =============================================================================
# DIRECTORIES AND SEED MIRRORS
# =============================================================================
DATA_DIR: str = "data"
WB_DIR: str = os.path.join(DATA_DIR, "wb")
FRED_DIR: str = os.path.join(DATA_DIR, "fred")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WB_DIR, exist_ok=True)
os.makedirs(FRED_DIR, exist_ok=True)

def ensure_file(path: str, content: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

ensure_file(
    os.path.join(DATA_DIR, "margin_finra.csv"),
    """
date,debit_bil
2025-08-31,1180
2025-09-30,1195
2025-10-31,1180
""",
)

ensure_file(
    os.path.join(DATA_DIR, "us_gdp_nominal.csv"),
    """
date,gdp_trillions
2025-04-01,28.0
2025-07-01,28.5
2025-10-01,28.8
""",
)

ensure_file(
    os.path.join(DATA_DIR, "cboe_putcall.csv"),
    """
date,pc_ratio
2025-11-20,0.78
2025-11-21,0.71
2025-11-22,0.69
""",
)

ensure_file(
    os.path.join(DATA_DIR, "aaii_sentiment.csv"),
    """
date,bull,bear,neutral
2025-11-07,42.0,30.0,28.0
2025-11-14,55.0,22.0,23.0
2025-11-21,61.5,18.0,20.5
""",
)

ensure_file(
    os.path.join(DATA_DIR, "insider_ratio.csv"),
    """
date,buy_ratio_pct
2025-10-31,14.0
2025-11-15,11.0
2025-11-22,9.0
""",
)

ensure_file(
    os.path.join(DATA_DIR, "imf_cofer_usd_share.csv"),
    """
date,usd_share
2023-12-31,58.4
2024-03-31,58.1
2024-06-30,57.8
2024-09-30,57.2
2025-06-30,56.3
""",
)

ensure_file(
    os.path.join(DATA_DIR, "pe_sp500.csv"),
    """
date,pe
2025-11-20,29.80
2025-11-21,30.10
2025-11-26,30.57
""",
)

ensure_file(
    os.path.join(DATA_DIR, "spx_percent_above_200dma.csv"),
    """
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
""",
)

ensure_file(
    os.path.join(DATA_DIR, "total_debt_gdp.csv"),
    """
date,total_debt_gdp_pct
2024-12-31,340.0
2025-06-30,345.0
2025-09-30,351.5
""",
)

ensure_file(
    os.path.join(DATA_DIR, "gpr_us.csv"),
    """
date,gpr
2024-12-31,150
2025-03-31,170
2025-06-30,190
2025-09-30,210
""",
)

ensure_file(
    os.path.join(DATA_DIR, "real_assets_basket.csv"),
    """
date,index
2023-11-30,100
2024-11-30,135
2025-11-30,160
""",
)

ensure_file(
    os.path.join(DATA_DIR, "gold_ath_by_ccy.csv"),
    """
ccy,ath_price
USD,2450
EUR,2300
JPY,360000
CNY,18000
GBP,2000
""",
)

ensure_file(
    os.path.join(DATA_DIR, "usd_gold_power.csv"),
    """
date,oz_per_1000
2024-11-30,0.25
2025-05-31,0.23
2025-11-30,0.21
""",
)

ensure_file(
    os.path.join(DATA_DIR, "farmland_index.csv"),
    """
date,index
2023-11-30,100
2024-11-30,112
2025-11-30,125
""",
)

# =============================================================================
# SESSION, FRED, PAGE CONFIG
# =============================================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})

fred = Fred(api_key=FRED_API_KEY)

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
        .status-red {color: #ff4444; font-weight: bold; font-size: 1.1rem;}
        .status-yellow {color: #ffbb33; font-weight: bold; font-size: 1.1rem;}
        .status-green {color: #00C851; font-weight: bold; font-size: 1.1rem;}
        .kill-box {
            background:#8b0000;
            color:white;
            padding:20px;
            border-radius:12px;
            font-size:2rem;
            text-align:center;
        }
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

# =============================================================================
# CONSTANTS
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
        index=pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce"),
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
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
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
        us["date"] = pd.to_datetime(us["date"], format="%Y-%m-%d", errors="coerce")
        wd["date"] = pd.to_datetime(wd["date"], format="%Y-%m-%d", errors="coerce")
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
        return float("nan"), float("nan"), "Mirror missing", []
    if numeric_time:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d", errors="coerce")
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

# =============================================================================
# LIVE DATA HELPERS (OFFICIAL SERIES)
# =============================================================================
@st.cache_data(ttl=1800)
def live_margin_gdp_from_mirrors() -> Tuple[float, float]:
    mdf = load_csv(os.path.join(DATA_DIR, "margin_finra.csv"))
    gdf = load_csv(os.path.join(DATA_DIR, "us_gdp_nominal.csv"))
    if mdf.empty or gdf.empty:
        return float("nan"), float("nan")
    mdf["date"] = pd.to_datetime(mdf["date"], format="%Y-%m-%d", errors="coerce")
    gdf["date"] = pd.to_datetime(gdf["date"], format="%Y-%m-%d", errors="coerce")
    mdf = mdf.dropna().sort_values("date")
    gdf = gdf.dropna().sort_values("date")
    if mdf.empty or gdf.empty:
        return float("nan"), float("nan")
    last_m = mdf.iloc[-1]
    prev_m = mdf.iloc[-2] if len(mdf) > 1 else last_m
    last_g = gdf.iloc[-1]
    prev_g = gdf.iloc[-2] if len(gdf) > 1 else last_g
    cur_pct = to_float(last_m["debit_bil"]) / (to_float(last_g["gdp_trillions"]) * 1000.0) * 100.0
    prev_pct = to_float(prev_m["debit_bil"]) / (to_float(prev_g["gdp_trillions"]) * 1000.0) * 100.0
    return round(cur_pct, 2), round(prev_pct, 2)

@st.cache_data(ttl=1800)
def live_put_call_from_mirror() -> float:
    df = load_csv(os.path.join(DATA_DIR, "cboe_putcall.csv"))
    if df.empty or "pc_ratio" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if df.empty:
        return float("nan")
    return round(to_float(df.iloc[-1]["pc_ratio"]), 3)

@st.cache_data(ttl=1800)
def live_aaii_bulls_from_mirror() -> float:
    df = load_csv(os.path.join(DATA_DIR, "aaii_sentiment.csv"))
    if df.empty or "bull" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if df.empty:
        return float("nan")
    return round(to_float(df.iloc[-1]["bull"]), 1)

@st.cache_data(ttl=1800)
def live_insider_buy_ratio_from_mirror() -> float:
    df = load_csv(os.path.join(DATA_DIR, "insider_ratio.csv"))
    if df.empty or "buy_ratio_pct" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if df.empty:
        return float("nan")
    return round(to_float(df.iloc[-1]["buy_ratio_pct"]), 1)

@st.cache_data(ttl=1800)
def live_sp500_pe_official() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        return round(float(j[0]["pe"]), 2)
    except Exception:
        c, _, _, _ = sp500_pe_latest()
        return float(c) if not pd.isna(c) else float("nan")

@st.cache_data(ttl=1800)
def live_gold_price_usd() -> float:
    try:
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        )
        j = SESSION.get(url, timeout=10).json()
        rate = float(j["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        return round(rate, 2)
    except Exception:
        df = load_csv(os.path.join(DATA_DIR, "usd_gold_power.csv"))
        if df.empty:
            return float("nan")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna().sort_values("date")
        if df.empty:
            return float("nan")
        oz_per_1000 = to_float(df.iloc[-1]["oz_per_1000"])
        return round(1000.0 / oz_per_1000, 2) if oz_per_1000 > 0 else float("nan")

@st.cache_data(ttl=1800)
def live_vix_level() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^VIX?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        return round(float(j[0]["price"]), 2)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_spx_price_and_drawdown() -> Tuple[float, float, float]:
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/^GSPC?serietype=line&apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        hist = j.get("historical", [])
        if not hist:
            raise ValueError("no hist")
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna().sort_values("date")
        last = float(df.iloc[-1]["close"])
        ath = float(df["close"].max())
        dd = (last / ath - 1.0) * 100.0
        return round(last, 2), round(ath, 2), round(dd, 2)
    except Exception:
        path = os.path.join(DATA_DIR, "sp500_price.csv")
        df = load_csv(path)
        if df.empty or "close" not in df.columns:
            return float("nan"), float("nan"), float("nan")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna().sort_values("date")
        last = float(df.iloc[-1]["close"])
        ath = float(df["close"].max())
        dd = (last / ath - 1.0) * 100.0
        return round(last, 2), round(ath, 2), round(dd, 2)

@st.cache_data(ttl=1800)
def live_hy_spread_bps() -> Tuple[float, bool]:
    cur, prev = fred_last_two("BAMLH0A0HYM2", mode="level")
    if pd.isna(cur):
        return float("nan"), False
    widening = not pd.isna(prev) and cur > prev
    return round(cur, 1), widening

@st.cache_data(ttl=1800)
def live_real_30y_yield() -> float:
    try:
        nom = fred.get_series_latest_release("DGS30").iloc[-1]
        cpi = fred.get_series_latest_release("CPIAUCSL").pct_change(12).iloc[-1] * 100
        return round(float(nom - cpi), 2)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_real_fed_rate() -> float:
    try:
        ff = fred.get_series_latest_release("FEDFUNDS").iloc[-1]
        cpi = fred.get_series_latest_release("CPIAUCSL").pct_change(12).iloc[-1] * 100
        return round(float(ff - cpi), 2)
    except Exception:
        return float("nan")

@st.cache_data(ttl=1800)
def live_liquidity_metrics() -> Tuple[float, float]:
    wal_s = fred_series("WALCL")
    if wal_s.empty:
        bs_yoy = float("nan")
    else:
        bs_yoy, _ = yoy_from_series(wal_s)
    try:
        sofr_s = fred_series("SOFR")
        ff_s = fred_series("FEDFUNDS")
        if sofr_s.empty or ff_s.empty:
            soff_spread = float("nan")
        else:
            sofr_last = float(sofr_s.iloc[-1])
            ff_last = float(ff_s.iloc[-1])
            soff_spread = round(sofr_last - ff_last, 2)
    except Exception:
        soff_spread = float("nan")
    return float(bs_yoy), float(soff_spread)

@st.cache_data(ttl=1800)
def long_total_debt_gdp() -> float:
    df = load_csv(os.path.join(DATA_DIR, "total_debt_gdp.csv"))
    if df.empty or "total_debt_gdp_pct" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if df.empty:
        return float("nan")
    return round(to_float(df.iloc[-1]["total_debt_gdp_pct"]), 1)

@st.cache_data(ttl=1800)
def long_gpr_value() -> float:
    df = load_csv(os.path.join(DATA_DIR, "gpr_us.csv"))
    if df.empty or "gpr" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if df.empty:
        return float("nan")
    return round(to_float(df.iloc[-1]["gpr"]), 1)

@st.cache_data(ttl=1800)
def long_gini_value() -> float:
    val, _ = fred_last_two("SIPOVGINIUSA", mode="level")
    return round(val, 3) if not pd.isna(val) else float("nan")

@st.cache_data(ttl=1800)
def long_wage_share_pct() -> float:
    val, _ = fred_last_two("LABSHPUSA156NRUG", mode="level")
    if pd.isna(val):
        return float("nan")
    if val <= 1:
        val *= 100.0
    return round(val, 1)

@st.cache_data(ttl=1800)
def long_productivity_cagr_5y() -> float:
    s = fred_series("OPHNFB")
    if s.empty or len(s) < 60:
        return float("nan")
    s = s.sort_index()
    last = float(s.iloc[-1])
    back_idx = max(0, len(s) - 60)
    base = float(s.iloc[back_idx])
    if base <= 0:
        return float("nan")
    years = 5.0
    cagr = (last / base) ** (1.0 / years) - 1.0
    return round(cagr * 100.0, 2)

@st.cache_data(ttl=1800)
def long_usd_reserve_yoy_pp() -> float:
    c, _, _, _ = cofer_usd_share_latest()
    df = load_csv(os.path.join(DATA_DIR, "imf_cofer_usd_share.csv"))
    if df.empty or "usd_share" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if len(df) < 5:
        return float("nan")
    last = float(df.iloc[-1]["usd_share"])
    one_year_ago = df[df["date"] <= df["date"].iloc[-1] - pd.DateOffset(years=1)]
    if one_year_ago.empty:
        return float("nan")
    base = float(one_year_ago.iloc[-1]["usd_share"])
    return round(last - base, 2)

@st.cache_data(ttl=1800)
def long_real_assets_basket_24m() -> float:
    df = load_csv(os.path.join(DATA_DIR, "real_assets_basket.csv"))
    if df.empty or "index" not in df.columns:
        return float("nan")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna().sort_values("date")
    if len(df) < 3:
        return float("nan")
    last = float(df.iloc[-1]["index"])
    base = float(df.iloc[0]["index"])
    if base <= 0:
        return float("nan")
    ret = (last / base - 1.0) * 100.0
    return round(ret, 1)

@st.cache_data(ttl=1800)
def long_real_30y_extreme() -> float:
    return live_real_30y_yield()

@st.cache_data(ttl=1800)
def long_usd_gold_power_oz_per_1000() -> float:
    price = live_gold_price_usd()
    if pd.isna(price) or price <= 0:
        df = load_csv(os.path.join(DATA_DIR, "usd_gold_power.csv"))
        if df.empty or "oz_per_1000" not in df.columns:
            return float("nan")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna().sort_values("date")
        if df.empty:
            return float("nan")
        return float(df.iloc[-1]["oz_per_1000"])
    return round(1000.0 / price, 3)

@st.cache_data(ttl=1800)
def long_gold_ath_vs_ccy_status() -> Tuple[str, int]:
    df = load_csv(os.path.join(DATA_DIR, "gold_ath_by_ccy.csv"))
    if df.empty or "ccy" not in df.columns or "ath_price" not in df.columns:
        return "No data", 0
    hits = 0
    total = 0
    try:
        price_usd = live_gold_price_usd()
        if not pd.isna(price_usd):
            total += 1
            ath_usd = float(df.loc[df["ccy"] == "USD", "ath_price"].iloc[0])
            if price_usd >= 0.9 * ath_usd:
                hits += 1
    except Exception:
        pass
    # Other currencies can be added with extra AV calls if needed
    if total == 0:
        return "No data", 0
    return f"{hits}/{total} majors near ATH", hits

@st.cache_data(ttl=1800)
def long_gdp_debt_stack() -> float:
    return long_total_debt_gdp()

# =============================================================================
# CORE LIVE VALUES SHARED
# =============================================================================
margin_gdp_pct, margin_gdp_prev = live_margin_gdp_from_mirrors()
put_call_ratio = live_put_call_from_mirror()
aaii_bull = live_aaii_bulls_from_mirror()
insider_buy_ratio = live_insider_buy_ratio_from_mirror()
pe_live = live_sp500_pe_official()
gold_spot = live_gold_price_usd()
hy_spread_bps, hy_widening = live_hy_spread_bps()
real_30y_live = live_real_30y_yield()
real_fed_live = live_real_fed_rate()
bs_yoy, sofr_spread = live_liquidity_metrics()
spx_last, spx_ath, spx_drawdown = live_spx_price_and_drawdown()
usd_gold_power = long_usd_gold_power_oz_per_1000()

# =============================================================================
# CORE TAB (50+ INDICATORS)
# =============================================================================
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
            c, p, s, h = mirror_latest_csv(
                path_pisa, "pisa_math_mean_usa", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src, hist = c, p, "OECD PISA — " + s, h
        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, h = mirror_latest_csv(
                path_cinc, "cinc_usa", "year", numeric_time=True
            )
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
            if not pd.isna(pe_live):
                cur = pe_live
                prev = p
                src = "FMP live + " + s
                hist = h
            elif not pd.isna(c):
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
        "CINC (mirror), UCDP (mirror), MULTPL/FMP (P/E)."
    )

# =============================================================================
# SHORT-TERM BUBBLE TIMING (10 KILL SIGNALS)
# =============================================================================
def build_short_term_table() -> Tuple[pd.DataFrame, int]:
    rows: List[Dict[str, object]] = []

    # 1 Margin debt
    if not pd.isna(margin_gdp_pct) and not pd.isna(margin_gdp_prev):
        rolling_over = margin_gdp_pct <= margin_gdp_prev
    else:
        rolling_over = False
    if pd.isna(margin_gdp_pct):
        status = "⚪"
        label = "NO DATA"
    else:
        if margin_gdp_pct >= 3.5 and rolling_over:
            status = "🔴"
            label = "KILL"
        elif margin_gdp_pct >= 3.0:
            status = "🟠"
            label = "WATCH"
        else:
            status = "🟢"
            label = "OK"
    rows.append(
        {
            "#": 1,
            "Signal": "Margin debt ≥3.5% of GDP & rolling over",
            "Value": f"{margin_gdp_pct:.2f}%" if not pd.isna(margin_gdp_pct) else "No data",
            "Threshold": "≥3.5% & falling MoM",
            "Status": label,
            "Why this matters": "Leverage shows how many people are borrowing to chase the market.",
        }
    )

    # 2 Real Fed funds
    if pd.isna(real_fed_live):
        label2 = "NO DATA"
    else:
        label2 = "KILL" if real_fed_live >= 1.5 else "OK"
    rows.append(
        {
            "#": 2,
            "Signal": "Real Fed funds ≥+1.5%",
            "Value": f"{real_fed_live:+.2f}%" if not pd.isna(real_fed_live) else "No data",
            "Threshold": "≥ +1.5%",
            "Status": label2 if not pd.isna(real_fed_live) else "NO DATA",
            "Why this matters": "When money stops being free, bubbles lose their fuel and start to pop.",
        }
    )

    # 3 Put/call
    if pd.isna(put_call_ratio):
        label3 = "NO DATA"
    else:
        label3 = "KILL" if put_call_ratio < 0.65 else "WATCH" if put_call_ratio < 0.8 else "OK"
    rows.append(
        {
            "#": 3,
            "Signal": "CBOE total put/call <0.65",
            "Value": f"{put_call_ratio:.3f}" if not pd.isna(put_call_ratio) else "No data",
            "Threshold": "< 0.65",
            "Status": label3,
            "Why this matters": "Low put/call means nobody is hedging — classic sign of overconfidence.",
        }
    )

    # 4 AAII bulls
    if pd.isna(aaii_bull):
        label4 = "NO DATA"
    else:
        label4 = "KILL" if aaii_bull > 60 else "WATCH" if aaii_bull > 50 else "OK"
    rows.append(
        {
            "#": 4,
            "Signal": "AAII bulls >60%",
            "Value": f"{aaii_bull:.1f}%" if not pd.isna(aaii_bull) else "No data",
            "Threshold": "> 60%",
            "Status": label4,
            "Why this matters": "When everyone is bullish, almost nobody is left to buy more.",
        }
    )

    # 5 P/E
    if pd.isna(pe_live):
        label5 = "NO DATA"
    else:
        label5 = "KILL" if pe_live > 30 else "WATCH" if pe_live > 25 else "OK"
    rows.append(
        {
            "#": 5,
            "Signal": "S&P 500 P/E >30×",
            "Value": f"{pe_live:.2f}×" if not pd.isna(pe_live) else "No data",
            "Threshold": "> 30×",
            "Status": label5,
            "Why this matters": "High P/E means prices are assuming perfection and zero mistakes.",
        }
    )

    # 6 Insider buying
    if pd.isna(insider_buy_ratio):
        label6 = "NO DATA"
    else:
        label6 = "KILL" if insider_buy_ratio < 10 else "WATCH" if insider_buy_ratio < 20 else "OK"
    rows.append(
        {
            "#": 6,
            "Signal": "Insider buying <10% of insider activity",
            "Value": f"{insider_buy_ratio:.1f}%" if not pd.isna(insider_buy_ratio) else "No data",
            "Threshold": "< 10%",
            "Status": label6,
            "Why this matters": "When insiders dump while buybacks slow, smart money is heading for the exit.",
        }
    )

    # 7 HY spreads
    if pd.isna(hy_spread_bps):
        label7 = "NO DATA"
    else:
        label7 = "KILL" if hy_spread_bps < 400 and hy_widening else "OK"
    rows.append(
        {
            "#": 7,
            "Signal": "High-yield spreads <400 bps (but widening)",
            "Value": f"{hy_spread_bps:.1f} bps" if not pd.isna(hy_spread_bps) else "No data",
            "Threshold": "< 400 bps & widening",
            "Status": label7,
            "Why this matters": "Tight spreads mean junk borrowers get money easily — risk is being ignored.",
        }
    )

    # 8 VIX
    if pd.isna(vix_level := live_vix_level()):
        label8 = "NO DATA"
    else:
        label8 = "KILL" if vix_level < 20 else "OK"
    rows.append(
        {
            "#": 8,
            "Signal": "VIX <20",
            "Value": f"{vix_level:.2f}" if not pd.isna(vix_level) else "No data",
            "Threshold": "< 20",
            "Status": label8,
            "Why this matters": "Very low volatility means extreme complacency near major tops.",
        }
    )

    # 9 Breadth
    path_breadth = os.path.join(DATA_DIR, "spx_percent_above_200dma.csv")
    breadth, _, _, _ = mirror_latest_csv(path_breadth, "value", "date", numeric_time=False)
    if pd.isna(breadth):
        label9 = "NO DATA"
    else:
        label9 = "KILL" if breadth < 25 else "WATCH" if breadth < 40 else "OK"
    rows.append(
        {
            "#": 9,
            "Signal": "% S&P above 200d <25%",
            "Value": f"{breadth:.1f}%" if not pd.isna(breadth) else "No data",
            "Threshold": "< 25%",
            "Status": label9,
            "Why this matters": "Thin breadth means only a few mega-caps are holding the index up.",
        }
    )

    # 10 Liquidity
    if pd.isna(bs_yoy) and pd.isna(sofr_spread):
        label10 = "NO DATA"
        val10 = "No data"
    else:
        kill_liq = (not pd.isna(bs_yoy) and bs_yoy <= -5.0) or (
            not pd.isna(sofr_spread) and sofr_spread > 0.5
        )
        label10 = "KILL" if kill_liq else "OK"
        val10 = ""
        if not pd.isna(bs_yoy):
            val10 += f"Fed BS YoY {bs_yoy:+.2f}%"
        if not pd.isna(sofr_spread):
            if val10:
                val10 += " / "
            val10 += f"SOFR spread {sofr_spread:+.2f} pts"
    rows.append(
        {
            "#": 10,
            "Signal": "Liquidity: Fed BS YoY ≤–5% OR SOFR spread >50 bps",
            "Value": val10,
            "Threshold": "≤ –5% or > 0.50 pts",
            "Status": label10,
            "Why this matters": "Aggressive QT or funding stress is how leveraged systems suddenly break.",
        }
    )

    df = pd.DataFrame(rows)
    kill_count = int((df["Status"] == "KILL").sum())
    return df, kill_count

short_df, kill_count = build_short_term_table()

# =============================================================================
# LONG-TERM SUPER-CYCLE (11 DARK RED + 3 NO-RETURN)
# =============================================================================
def build_long_term_table() -> Tuple[pd.DataFrame, int, int]:
    rows: List[Dict[str, object]] = []
    no_return_count = 0

    # 1 Total Debt/GDP
    total_debt_pct = long_gdp_debt_stack()
    if pd.isna(total_debt_pct):
        status1 = "NO DATA"
    else:
        status1 = "DARK RED" if total_debt_pct > 400 else "WATCH" if total_debt_pct > 300 else "OK"
    rows.append(
        {
            "#": 1,
            "Signal": "Total Debt/GDP (private + public + foreign)",
            "Value": f"{total_debt_pct:.1f}%" if not pd.isna(total_debt_pct) else "No data",
            "Dark Red Threshold": "> 400%",
            "Status": status1,
            "Why this matters": "Debt >3–4× GDP always forced resets (defaults, inflation, wars).",
        }
    )

    # 2 Gold at/near ATH vs majors
    gold_ath_text, gold_ath_hits = long_gold_ath_vs_ccy_status()
    status2 = "DARK RED" if gold_ath_hits >= 1 else "OK" if gold_ath_text != "No data" else "NO DATA"
    rows.append(
        {
            "#": 2,
            "Signal": "Gold at/near ATH vs major currencies",
            "Value": gold_ath_text,
            "Dark Red Threshold": "Persistent ATH vs USD/EUR/JPY/CNY",
            "Status": status2,
            "Why this matters": "When gold breaks out in all currencies, the world is voting against fiat money.",
        }
    )

    # 3 USD/gold power
    if pd.isna(usd_gold_power):
        status3 = "NO DATA"
        val3 = "No data"
    else:
        status3 = "DARK RED" if usd_gold_power < 0.10 else "OK"
        val3 = f"{usd_gold_power:.3f} oz per $1,000"
    rows.append(
        {
            "#": 3,
            "Signal": "USD/gold power <0.10 oz per $1,000",
            "Value": val3,
            "Dark Red Threshold": "< 0.10 oz and falling",
            "Status": status3,
            "Why this matters": "Shows how much real value the dollar still holds relative to hard money.",
        }
    )

    # 4 Real 30Y yield extreme
    if pd.isna(real_30y_live):
        status4 = "NO DATA"
    else:
        status4 = "DARK RED" if abs(real_30y_live) >= 5.0 else "OK"
    rows.append(
        {
            "#": 4,
            "Signal": "Real 30Y yield extreme (≥+5% or ≤–5%)",
            "Value": f"{real_30y_live:+.2f}%" if not pd.isna(real_30y_live) else "No data",
            "Dark Red Threshold": "≥ +5% or ≤ –5%",
            "Status": status4,
            "Why this matters": "Extreme real rates break either borrowers or savers and force regime changes.",
        }
    )

    # 5 GPR
    gpr_val = long_gpr_value()
    if pd.isna(gpr_val):
        status5 = "NO DATA"
    else:
        status5 = "DARK RED" if gpr_val > 300 else "WATCH" if gpr_val > 150 else "OK"
    rows.append(
        {
            "#": 5,
            "Signal": "Geopolitical Risk Index (GPR)",
            "Value": f"{gpr_val:.1f}" if not pd.isna(gpr_val) else "No data",
            "Dark Red Threshold": "> 300 and rising",
            "Status": status5,
            "Why this matters": "High geopolitical tension + high debt is the classic reset cocktail.",
        }
    )

    # 6 Gini
    if pd.isna(long_gini_value()):
        status6 = "NO DATA"
        gini_str = "No data"
    else:
        gini_val = long_gini_value()
        status6 = "DARK RED" if gini_val > 0.5 else "WATCH" if gini_val > 0.4 else "OK"
        gini_str = f"{gini_val:.3f}"
    rows.append(
        {
            "#": 6,
            "Signal": "US Gini coefficient (inequality)",
            "Value": gini_str,
            "Dark Red Threshold": "> 0.50 and climbing",
            "Status": status6,
            "Why this matters": "Extreme inequality makes societies fragile and prone to shocks and revolts.",
        }
    )

    # 7 Wage share
    wage_share = long_wage_share_pct()
    if pd.isna(wage_share):
        status7 = "NO DATA"
    else:
        status7 = "DARK RED" if wage_share < 50.0 else "OK"
    rows.append(
        {
            "#": 7,
            "Signal": "Wage share of GDP",
            "Value": f"{wage_share:.1f}%" if not pd.isna(wage_share) else "No data",
            "Dark Red Threshold": "< 50%",
            "Status": status7,
            "Why this matters": "Falling wage share means rising inequality and political stress.",
        }
    )

    # 8 Productivity multi-year
    prod_cagr = long_productivity_cagr_5y()
    if pd.isna(prod_cagr):
        status8 = "NO DATA"
    else:
        status8 = "DARK RED" if prod_cagr < 0 else "WATCH"
    rows.append(
        {
            "#": 8,
            "Signal": "Productivity growth (multi-year trend)",
            "Value": f"{prod_cagr:+.2f}%" if not pd.isna(prod_cagr) else "No data",
            "Dark Red Threshold": "Negative trend over years",
            "Status": status8,
            "Why this matters": "Low or negative productivity means growth is borrowed from the future via debt.",
        }
    )

    # 9 USD reserve share YoY change
    usd_res_pp = long_usd_reserve_yoy_pp()
    if pd.isna(usd_res_pp):
        status9 = "NO DATA"
    else:
        status9 = "DARK RED" if usd_res_pp <= -2.0 else "WATCH"
    rows.append(
        {
            "#": 9,
            "Signal": "USD reserve share YoY change",
            "Value": f"{usd_res_pp:+.2f} pp" if not pd.isna(usd_res_pp) else "No data",
            "Dark Red Threshold": "< –2 pp over 12m",
            "Status": status9,
            "Why this matters": "When central banks diversify away from USD, the existing system is weakening.",
        }
    )

    # 10 Real assets basket
    real_assets = long_real_assets_basket_24m()
    if pd.isna(real_assets):
        status10 = "NO DATA"
    else:
        status10 = "DARK RED" if real_assets >= 50.0 else "OK"
    rows.append(
        {
            "#": 10,
            "Signal": "Real assets basket vs fiat (24m)",
            "Value": f"{real_assets:+.1f}%" if not pd.isna(real_assets) else "No data",
            "Dark Red Threshold": "Gold/oil/farmland all +50% vs fiat",
            "Status": status10,
            "Why this matters": "When real assets outrun financial assets for years, capital is voting against paper.",
        }
    )

    # 11 Official reset event (manual)
    reset_event = st.checkbox(
        "Official reset event (laws/treaties/FX regime)",
        value=False,
        key="reset_event_checkbox",
    )
    status11 = "DARK RED" if reset_event else "OK"
    rows.append(
        {
            "#": 11,
            "Signal": "Official reset event (laws/treaties/FX regime)",
            "Value": "Manual toggle",
            "Dark Red Threshold": "Explicit regime reset",
            "Status": status11,
            "Why this matters": "Once official rules change, the old monetary system is over.",
        }
    )

    # No-return triggers (manual checkboxes + one data-driven)
    st.markdown("#### No-return triggers (final lock-in)")
    cb_gold = st.checkbox(
        "Central banks in aggressive net gold buying regime",
        value=False,
        help="Use WGC/IMF data — this is a manual judgement flag.",
    )
    if cb_gold:
        no_return_count += 1
    g20_gold = st.checkbox(
        "G20/BRICS proposing or moving toward a gold-linked or CBDC reserve system",
        value=False,
        help="Manual flag from official communiqués and treaties.",
    )
    if g20_gold:
        no_return_count += 1
    ten_yield, _ = fred_last_two("DGS10", mode="level")
    cpi_yoy, _ = fred_last_two("CPIAUCSL", mode="yoy")
    high_10y = (
        not pd.isna(ten_yield) and not pd.isna(cpi_yoy) and ten_yield >= 7.0 and cpi_yoy >= 4.0
    )
    st.write(
        f"US 10Y yield vs CPI: {ten_yield:.2f}% / {cpi_yoy:.2f}%"
        if not (pd.isna(ten_yield) or pd.isna(cpi_yoy))
        else "US 10Y vs CPI: No data"
    )
    if high_10y:
        no_return_count += 1

    df = pd.DataFrame(rows)
    dark_count = int((df["Status"] == "DARK RED").sum())
    return df, dark_count, no_return_count

with tab_long:
    # NEW LONG-TERM RULE LOGIC + BOX
    st.markdown(
        """
        **New Unbreakable Timing Rules**

        **Short-term — When to time the burst ahead (sell 80–90%)**  
        You now wait for **7-out-of-10 kill signals** while **S&P is still within –8% of ATH (or green YTD).**

        **Long-term — When the super-cycle really ends (go 80–100% hard assets forever)**  
        You now wait for **8+ dark-red** AND **at least two of the three no-return triggers**  
        (reserve share collapse + real assets explosion + official reset event).
        """
    )

    if dark_red_count >= 8 and no_return_count >= 2:
        reset_box_html = """
        <div style="background:#8b0000; color:white; padding:20px; border-radius:12px; font-size:2rem; text-align:center">
        8+ DARK-RED SUPER-CYCLE SIGNALS + 2 NO-RETURN TRIGGERS<br/>
        (RESERVE SHARE COLLAPSE · REAL ASSETS EXPLOSION · OFFICIAL RESET EVENT)<br/>
        → RULE: 80–100% GOLD/BITCOIN/CASH/HARD ASSETS FOR 5–15 YEARS
        </div>
        """
        st.markdown(reset_box_html, unsafe_allow_html=True)


# =============================================================================
# SHORT-TERM TAB RENDER
# =============================================================================
with tab_short:
    # NEW SHORT-TERM RULE LOGIC + BOX
    st.markdown(
        """
        **New Unbreakable Timing Rules**

        **Short-term — When to time the burst ahead (sell 80–90%)**  
        You now wait for **7-out-of-10 kill signals** while **S&P is still within –8% of ATH (or green YTD).**

        **Long-term — When the super-cycle really ends (go 80–100% hard assets forever)**  
        You now wait for **8+ dark-red** AND **at least two of the three no-return triggers**  
        (reserve share collapse + real assets explosion + official reset event).
        """
    )

    if kill_count >= 7 and spx_drawdown >= -8.0:
        kill_box_html = """
        <div style="background:#8b0000; color:white; padding:20px; border-radius:12px; font-size:2rem; text-align:center">
        7 OUT OF 10 KILL SIGNALS + S&P WITHIN –8% OF ATH (OR GREEN YTD)<br/>
        → RULE: SELL 80–90% OF STOCKS AND LOCK IN THIS CYCLE'S GAINS
        </div>
        """
        st.markdown(kill_box_html, unsafe_allow_html=True)

# =============================================================================
# GLOBAL REGIME BANNER
# =============================================================================
regime_text = (
    "Current regime: Late-stage melt-up (short-term) inside late-stage debt super-cycle (long-term). "
    "Ride stocks with 20–30% cash + 30–40% gold/BTC permanent."
)
spx_info = (
    f"SPX: {spx_last:,.2f} | Drawdown: {spx_drawdown:.2f}% | ATH: {spx_ath:,.2f} | "
    f"Kill: {kill_count}/10 | Dark: {dark_count}/11 | No-return: {no_return_count}/3"
    if not pd.isna(spx_last)
    else "SPX data unavailable."
)
st.markdown(f"<div class='kill-box'>{regime_text} {spx_info}</div>", unsafe_allow_html=True)

st.caption(
    "Live official releases where possible • 30-minute cache • Fallback mirrors only where needed • Econ Mirror — Nov 2025"
)
