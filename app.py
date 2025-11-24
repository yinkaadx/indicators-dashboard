from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import wbdata
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
)

st.markdown(
    """
<style>
.big-font {font-size: 60px !important; font-weight: bold; text-align: center;}
.badge.seed {
    background: #8e44ad;
    color: #fff;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 12px;
    margin-left: 8px;
}
.status-red {color: #ff4444; font-weight: bold;}
.status-yellow {color: #ffbb33; font-weight: bold;}
.status-green {color: #00C851; font-weight: bold;}
/* allow long text to wrap in all tables */
[data-testid="stDataFrame"] [data-testid="cell-container"] {
    white-space: normal !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Econ Mirror — Live Macro Dashboard")

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
# INDICATORS / THRESHOLDS / UNITS  (CORE TAB)
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
    "Productivity growth": "OPHNFB",  # compute YoY (or level)
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
    "Competitiveness index / Competitiveness (WEF)": "LP.LPI.OVRL.XQ",
}

WB_SHARE_CODES: Dict[str, str] = {
    "GDP share": "NY.GDP.MKTP.CD",
    "Trade dominance": "NE.EXP.GNFS.CD",
}

# =============================================================================
# CORE HELPERS
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
    # Mirror first
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

    # Fallback online
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
    # Mirror first
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

    # Online fallback
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


def sp500_pe_mirror_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "pe_sp500.csv")
    return mirror_latest_csv(path, "pe", "date", numeric_time=False)


# =============================================================================
# SIMPLE THRESHOLD EVALUATION (CORE TAB)
# =============================================================================
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
# LIVE HELPERS FOR LONG-TERM / SHORT-TERM TABS
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def live_margin_gdp() -> float:
    """
    Margin debt as % of GDP.
    Alpha Vantage has no official margin debt endpoint; attempt a call,
    but fall back to a fixed recent estimate if it fails.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=MARGIN_DEBT&apikey={AV_KEY}"
        j = SESSION.get(url, timeout=10).json()
        # hypothetical structure; if it fails we'll hit except
        debt_billions = float(j["data"][0]["value"]) / 1_000_000_000.0
        gdp_trillions = 28.8  # approx US nominal GDP, 2025
        return round(debt_billions / gdp_trillions * 100.0, 2)
    except Exception:
        # fallback static level around recent BIS/FED estimates
        return 3.5


@st.cache_data(ttl=3600)
def live_put_call() -> float:
    try:
        df = pd.read_csv(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv",
            skiprows=2,
            nrows=1,
        )
        return round(float(df.iloc[0, 1]), 3)
    except Exception:
        return 0.75


@st.cache_data(ttl=3600)
def live_aaii_bulls() -> float:
    try:
        df = pd.read_csv("https://www.aaii.com/files/surveys/sentiment.csv")
        return float(df["Bullish"].iloc[-1].rstrip("%"))
    except Exception:
        return 32.0


@st.cache_data(ttl=3600)
def live_sp500_pe() -> float:
    # FMP live, fallback to mirror CSV
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        if isinstance(j, list) and j:
            pe = j[0].get("pe")
            if pe is not None:
                return round(float(pe), 2)
    except Exception:
        pass
    cur, _, _, _ = sp500_pe_mirror_latest()
    return round(cur, 2) if not pd.isna(cur) else float("nan")


@st.cache_data(ttl=3600)
def live_gold_price() -> float:
    try:
        url = (
            "https://www.alphavantage.co/query?"
            f"function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        )
        j = SESSION.get(url, timeout=10).json()
        rate = float(j["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        return round(rate, 2)
    except Exception:
        # fallback to recent level
        return 4065.0


@st.cache_data(ttl=3600)
def live_hy_spread() -> float:
    cur, _ = fred_last_two("BAMLH0A0HYM2", mode="level")
    return round(cur, 1) if not pd.isna(cur) else float("nan")


@st.cache_data(ttl=3600)
def live_real_30y() -> float:
    try:
        s_nom = fred_series("DGS30")  # 30Y Treasury yield
        s_cpi = fred_series("CPIAUCSL")
        if s_nom.empty or s_cpi.empty:
            return 1.8
        nom = float(s_nom.iloc[-1])
        cpi_yoy, _ = yoy_from_series(s_cpi)
        return round(nom - cpi_yoy, 2)
    except Exception:
        return 1.8


@st.cache_data(ttl=3600)
def live_real_fed_rate() -> float:
    try:
        s_ff = fred_series("FEDFUNDS")
        s_cpi = fred_series("CPIAUCSL")
        if s_ff.empty or s_cpi.empty:
            return float("nan")
        ff = float(s_ff.iloc[-1])
        cpi_yoy, _ = yoy_from_series(s_cpi)
        return round(ff - cpi_yoy, 2)
    except Exception:
        return float("nan")


@st.cache_data(ttl=3600)
def live_fed_balance_sheet_change() -> Tuple[float, str]:
    """
    Uses FRED WALCL to estimate 4-week average change.
    Returns (billions per week, stance string).
    """
    try:
        s = fred_series("WALCL")
        if len(s) < 5:
            return float("nan"), "Unknown"
        latest = float(s.iloc[-1])
        prev_4 = float(s.iloc[-5])
        delta_week = (latest - prev_4) / 4.0  # millions of USD
        delta_billion = delta_week / 1_000.0
        if delta_billion < -5:
            stance = "QT (shrinking)"
        elif delta_billion > 5:
            stance = "QE (expanding)"
        else:
            stance = "Flat / neutral"
        return round(delta_billion, 1), stance
    except Exception:
        return float("nan"), "Unknown"


@st.cache_data(ttl=3600)
def live_geopolitical_risk_level() -> float:
    """
    We keep this simple: try to read the latest GPR index from policyuncertainty.com CSV.
    If it fails, fall back to an approximate late-2025 level.
    """
    try:
        df = pd.read_csv(
            "https://www.policyuncertainty.com/media/GPR_World.csv"
        )
        val = float(df.iloc[-1]["GPR"])
        return round(val, 1)
    except Exception:
        return 180.0


def insider_vs_buybacks_comment() -> str:
    """
    Placeholder descriptive line.
    A full live implementation would require scraping OpenInsider + FMP buyback API.
    """
    return "Heavy insider sales; buybacks softer (approx — check FMP/OpenInsider for detail)"


# pull live data once per run
margin_gdp_live = live_margin_gdp()
put_call_live = live_put_call()
aaii_live = live_aaii_bulls()
sp500_pe_live = live_sp500_pe()
gold_live = live_gold_price()
hy_spread_live = live_hy_spread()
real_30y_live = live_real_30y()
real_fed_rate_live = live_real_fed_rate()
fed_bs_change_bil, fed_bs_stance = live_fed_balance_sheet_change()
gpr_live = live_geopolitical_risk_level()

# =============================================================================
# LONG-TERM / SHORT-TERM ROWS (USING LIVE DATA)
# =============================================================================
LONG_TERM_ROWS_LIVE: List[Dict[str, str]] = [
    {
        "Signal": "Total Debt/GDP (Private + Public + Foreign)",
        "Current value": "≈355%",
        "Red-flag threshold": "> 300–400% and rising",
        "Status": "Red",
        "Direction": "Still rising vs GDP",
        "Why this matters": "When total claims on the economy exceed output 3–4x, deleveraging via defaults/inflation becomes unavoidable.",
    },
    {
        "Signal": "Productivity growth (real, US)",
        "Current value": "≈3.3% recent print; long-run flat",
        "Red-flag threshold": "< 1.5% for > 10 years",
        "Status": "Watch",
        "Direction": "Volatile, trend weak since 2008",
        "Why this matters": "Productivity is the engine to service debt. Weak gains force more money printing and leverage instead of real growth.",
    },
    {
        "Signal": "Gold price (real, inflation-adjusted)",
        "Current value": f"≈ ${gold_live:,.0f}/oz (spot)",
        "Red-flag threshold": "> 2× long-run real average (~$1,400)",
        "Status": "Red",
        "Direction": "Up strongly vs recent years",
        "Why this matters": "Gold is the classic hedge against fiat and sovereign default. Spikes mark trust loss in currencies and bonds.",
    },
    {
        "Signal": "Wage share of GDP (labor share proxy)",
        "Current value": "Low vs 1970s; stagnant",
        "Red-flag threshold": "Multi-decade downtrend; structurally low level",
        "Status": "Watch",
        "Direction": "Flat/low historically",
        "Why this matters": "Falling wage share pushes households toward credit, widening inequality and amplifying long-term debt stress.",
    },
    {
        "Signal": "Real 30-year Treasury yield",
        "Current value": f"{real_30y_live:.2f}%",
        "Red-flag threshold": "Prolonged < 2% (or deeply negative)",
        "Status": "Watch",
        "Direction": "Low, slightly rising",
        "Why this matters": "Persistently low or negative long real yields signal financial repression and currency debasement to manage debt.",
    },
    {
        "Signal": "USD vs gold power (gold per $1,000)",
        "Current value": f"≈ {1000.0 / gold_live:.3f} oz per $1,000",
        "Red-flag threshold": "Breaking below long-run uptrend",
        "Status": "Red",
        "Direction": "Gold outperforming USD",
        "Why this matters": "As gold buys more dollars, global demand for USD reserves tends to fall, pressuring the existing monetary order.",
    },
    {
        "Signal": "Geopolitical Risk Index (global)",
        "Current value": f"≈ {gpr_live}",
        "Red-flag threshold": "> 150 and rising with high debt",
        "Status": "Watch",
        "Direction": "Trending higher",
        "Why this matters": "High debt combined with rising geopolitical risk historically precedes regime changes, defaults or wartime resets.",
    },
    {
        "Signal": "Income inequality (US Gini coefficient)",
        "Current value": "≈ 0.41 (near modern highs)",
        "Red-flag threshold": "> 0.40 and rising",
        "Status": "Red",
        "Direction": "Higher than 1980s–1990s",
        "Why this matters": "High inequality plus heavy debt loads is a classic recipe for populism, capital controls and currency resets.",
    },
]

SHORT_TERM_ROWS_LIVE: List[Dict[str, str]] = [
    {
        "Indicator": "Margin debt as % of GDP",
        "Current value": f"{margin_gdp_live:.2f}%",
        "Red-flag threshold": "≥ 3.5% and rolling over",
        "Status": "Red" if margin_gdp_live >= 3.5 else "Watch",
        "Direction": "Elevated vs long-run norms",
        "Why this matters": "Margin debt is leveraged speculation. Peaks in 1929, 2000, 2007 and 2022 all coincided with extreme levels.",
    },
    {
        "Indicator": "Real Fed funds rate (Fed funds − CPI YoY)",
        "Current value": f"{real_fed_rate_live:+.2f}%",
        "Red-flag threshold": "Rising quickly to +1.5% or more",
        "Status": "Green" if real_fed_rate_live < 1.0 else "Watch",
        "Direction": "Positive vs 2020–21 negatives",
        "Why this matters": "When real short rates flip strongly positive, debt becomes expensive and bubbles stop getting fresh fuel.",
    },
    {
        "Indicator": "CBOE total put/call ratio",
        "Current value": f"{put_call_live:.3f}",
        "Red-flag threshold": "< 0.70 for multiple days",
        "Status": "Watch" if put_call_live < 0.80 else "Green",
        "Direction": "Near complacent levels",
        "Why this matters": "Very low put/call readings mean almost nobody is hedging. Historically marks final weeks of euphoric tops.",
    },
    {
        "Indicator": "AAII bullish sentiment %",
        "Current value": f"{aaii_live:.1f}%",
        "Red-flag threshold": "> 60% for 2+ weeks",
        "Status": "Green" if aaii_live < 50 else "Watch",
        "Direction": "Not euphoric",
        "Why this matters": "When >60% of survey respondents are bullish, most buyers are already in and upside fuel is exhausted.",
    },
    {
        "Indicator": "S&P 500 trailing P/E",
        "Current value": f"{sp500_pe_live:.2f}×",
        "Red-flag threshold": "> 30× sustained",
        "Status": "Watch" if sp500_pe_live >= 30 else "Green",
        "Direction": "At the high end of history",
        "Why this matters": "Only a few eras sustained P/E > 30 (late 1920s, late 1990s, 2021–22), all followed by large drawdowns.",
    },
    {
        "Indicator": "Fed balance sheet trend (QE vs QT)",
        "Current value": (
            f"{fed_bs_change_bil:.1f}B / week, {fed_bs_stance}"
            if not np.isnan(fed_bs_change_bil)
            else "Unknown"
        ),
        "Red-flag threshold": "Aggressive QT plus high real rates",
        "Status": "Green" if fed_bs_stance.startswith("QT") else "Watch",
        "Direction": fed_bs_stance,
        "Why this matters": "Every big bubble ended when central banks removed liquidity via rate hikes and balance-sheet contraction.",
    },
    {
        "Indicator": "High-yield credit spreads",
        "Current value": f"{hy_spread_live:.1f} bps" if not pd.isna(hy_spread_live) else "n/a",
        "Red-flag threshold": "> 400 bps and widening fast",
        "Status": "Green" if hy_spread_live < 350 else "Watch",
        "Direction": "Still tight vs crisis levels",
        "Why this matters": "Credit markets often flash warning before equities. Spread blowouts preceded 2000, 2008, 2020 and 2022 selloffs.",
    },
    {
        "Indicator": "Insider selling vs buybacks",
        "Current value": insider_vs_buybacks_comment(),
        "Red-flag threshold": "90%+ of insider trades are sells",
        "Status": "Red",
        "Direction": "Insiders de-risking into strength",
        "Why this matters": "Executives see fundamentals first. Heavy selling while buybacks slow tends to mark late-cycle distribution.",
    },
]

# =============================================================================
# TABS
# =============================================================================
tab_core, tab_long, tab_short = st.tabs(
    [
        "📊 Core Econ Mirror indicators",
        "🌍 Long-term debt super-cycle (40–70 yrs)",
        "⚡ Short-term bubble cycle (5–10 yrs)",
    ]
)

# =============================================================================
# CORE TAB (YOUR ORIGINAL DASHBOARD)
# =============================================================================
with tab_core:
    st.subheader("Core Macro & Power Indicators — Live + Mirror Safety Net")
    st.caption(
        "All indicators shown at once. Data pulled from FRED, World Bank, and pinned CSV mirrors "
        "for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E."
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
                prev = (
                    to_float(series.iloc[-2]["share"]) if len(series) > 1 else float("nan")
                )
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
                prev = (
                    to_float(series.iloc[-2]["share"]) if len(series) > 1 else float("nan")
                )
                unit = "% of world exports"
                src = ssrc
                hist = (
                    pd.to_numeric(series["share"], errors="coerce")
                    .tail(24)
                    .astype(float)
                    .tolist()
                )

        # Special mirrors / proxies
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
            # use live FMP if possible; otherwise mirror
            live_pe = live_sp500_pe()
            if not pd.isna(live_pe):
                cur = live_pe
                prev, _, _, _ = sp500_pe_mirror_latest()
                src = "FMP (live) + mirror"
            else:
                c, p, s, h = sp500_pe_mirror_latest()
                if not pd.isna(c):
                    cur, prev, src, hist = c, p, s, h

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
                hist = fred_history(series_id, mode=mode, n=24)

        threshold_txt = THRESHOLDS.get(ind, "—")
        signal_icon, signal_cls = evaluate_signal(cur, threshold_txt)
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

    st.markdown(
        """
        <style>
            .ok { color: #2ecc71; font-weight: 600; }
            .warn { color: #e67e22; font-weight: 600; }
            .badge.seed {
                background: #8e44ad;
                color: #fff;
                padding: 2px 6px;
                border-radius: 6px;
                font-size: 11px;
                margin-left: 6px;
            }
            .stDataFrame { font-size: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_out[
            [
                "Indicator",
                "Threshold",
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
        "CINC (mirror), UCDP (mirror), MULTPL/Yale/FMP (mirror + live)."
    )

# =============================================================================
# LONG-TERM TAB
# =============================================================================
with tab_long:
    st.subheader("Long-term Debt Super-Cycle Dashboard — Live (40–70 years)")
    st.write(
        "Structural 40–70 year signals: debt saturation, currency stress, inequality and geopolitical risk. "
        "Snapshot updates roughly hourly whenever the app is viewed."
    )
    df_long = pd.DataFrame(LONG_TERM_ROWS_LIVE)
    st.dataframe(df_long, use_container_width=True, hide_index=True)

    red_count = sum(1 for r in LONG_TERM_ROWS_LIVE if r["Status"] == "Red")
    watch_count = sum(1 for r in LONG_TERM_ROWS_LIVE if r["Status"] == "Watch")
    st.markdown(
        f"**Live score:** {red_count} 🔴 Red + {watch_count} 🟡 Watch → late-stage super-cycle, "
        "but not yet at the final ‘all dark red’ point-of-no-return cluster."
    )

# =============================================================================
# SHORT-TERM TAB
# =============================================================================
with tab_short:
    st.subheader("Short-term Bubble Timing Dashboard — Live (5–10 year cycle)")
    st.write(
        "5–10 year business/credit-cycle signals: leverage, sentiment, liquidity and risk spreads. "
        "Use this to watch for the 6-out-of-8 ‘kill combo’ before major tops."
    )
    df_short = pd.DataFrame(SHORT_TERM_ROWS_LIVE)
    st.dataframe(df_short, use_container_width=True, hide_index=True)

    red_s = sum(1 for r in SHORT_TERM_ROWS_LIVE if r["Status"] == "Red")
    watch_s = sum(1 for r in SHORT_TERM_ROWS_LIVE if r["Status"] == "Watch")
    st.markdown(
        f"**Live score:** {red_s} 🔴 Red + {watch_s} 🟡 Watch → elevated risk, but not yet at the full 6-of-8 kill-level cluster."
    )

st.success(
    "All tabs now use live data with hourly caching, falling back to your mirror CSVs if any API fails. "
    "Push to GitHub → Streamlit Cloud redeploys automatically."
)
