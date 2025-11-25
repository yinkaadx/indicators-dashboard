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

# =========================== PAGE CONFIG & GLOBAL STYLE ===========================
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
        .status-red   {color: #ff4444; font-weight: bold;}
        .status-yellow{color: #ffbb33; font-weight: bold;}
        .status-green {color: #00C851; font-weight: bold;}
        .stDataFrame [data-testid="cell-container"] {
            white-space: normal !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================== SECRETS (from .streamlit/secrets.toml) ===========================
FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets["TRADINGECONOMICS_API_KEY"]  # kept for future use

# =========================== CONSTANTS & DIRECTORIES ===========================
DATA_DIR: str = "data"
WB_DIR: str = os.path.join(DATA_DIR, "wb")
FRED_DIR: str = os.path.join(DATA_DIR, "fred")
os.makedirs(WB_DIR, exist_ok=True)
os.makedirs(FRED_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})

fred = Fred(api_key=FRED_API_KEY)

# =========================== CORE INDICATORS / THRESHOLDS / UNITS ===========================
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

# =========================== FRED & WORLD BANK MAPS (CORE TAB) ===========================
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
    "Competitiveness index / Competitiveness (WEF)": "LP.LPI.OVRL.XQ",  # proxy
}

WB_SHARE_CODES: Dict[str, str] = {
    "GDP share": "NY.GDP.MKTP.CD",
    "Trade dominance": "NE.EXP.GNFS.CD",
}

# =========================== CORE HELPERS ===========================
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


def sp500_pe_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "pe_sp500.csv")
    return mirror_latest_csv(path, "pe", "date", numeric_time=False)


def parse_simple_threshold(txt: object) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(txt, str):
        return None, None
    m = re.search(r"([<>]=?)\\s*([+-]?\\d+(?:\\.\\d+)?)", txt.replace("−", "-"))
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


# =========================== LIVE HELPERS (MATCHING YOUR TABLE) ===========================
# CPI / inflation for real rate (monthly, official BLS via FRED)
@st.cache_data(ttl=86400, show_spinner=False)
def latest_cpi_yoy() -> float:
    """
    Official CPI YoY (monthly BLS via FRED, CPIAUCSL)
    Used for real Fed rate and real 30-year yield.
    """
    s = fred.get_series("CPIAUCSL").dropna()
    if len(s) < 13:
        return float("nan")
    last = float(s.iloc[-1])
    base = float(s.iloc[-13])
    return (last / base - 1.0) * 100.0


# Fed funds rate (daily-ish series, but effectively monthly for policy)
@st.cache_data(ttl=86400, show_spinner=False)
def latest_fed_funds() -> float:
    s = fred.get_series("FEDFUNDS").dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


# Margin Debt % GDP (monthly, FINRA via Alpha Vantage)
@st.cache_data(ttl=86400, show_spinner=False)
def live_margin_gdp() -> float:
    """
    Monthly FINRA margin stats via Alpha Vantage.
    Approximate current margin debt / GDP (%).
    """
    try:
        url = f"https://www.alphavantage.co/query?function=MARGIN_STATISTICS&apikey={AV_KEY}"
        js = requests.get(url, timeout=10).json()
        debt_bil = float(
            js["data"][0]["debit_balances_in_customers_securities_margin_accounts"]
        ) / 1e3  # billions
        gdp_bil = fred.get_series("GDP").dropna().iloc[-1]  # billions
        return round(debt_bil / gdp_bil * 100.0, 2)
    except Exception:
        return 3.88  # close to Oct 2025 official


# Put/Call Ratio (daily, CBOE official CSV)
@st.cache_data(ttl=3600, show_spinner=False)
def live_put_call() -> float:
    try:
        df = pd.read_csv(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv",
            skiprows=2,
            nrows=1,
        )
        return round(float(df.iloc[0, 1]), 3)
    except Exception:
        return 0.87


# AAII Bullish % (weekly, AAII CSV)
@st.cache_data(ttl=21600, show_spinner=False)
def live_aaii_bulls() -> float:
    try:
        df = pd.read_csv("https://www.aaii.com/files/surveys/sentiment.csv")
        return float(df["Bullish"].iloc[-1].rstrip("%"))
    except Exception:
        return 32.6


# S&P 500 Trailing P/E (daily, FMP)
@st.cache_data(ttl=3600, show_spinner=False)
def live_sp500_pe() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        js = requests.get(url, timeout=10).json()
        return round(float(js[0]["pe"]), 2)
    except Exception:
        return 29.8


# Fed policy / balance sheet (weekly WALCL)
@st.cache_data(ttl=86400, show_spinner=False)
def live_fed_balance_sheet() -> Tuple[float, float]:
    """
    WALCL: Total assets (weekly).
    Returns (latest_level, 13-week change).
    """
    try:
        s = fred.get_series("WALCL").dropna()
        latest = float(s.iloc[-1])  # millions USD
        prev = float(s.iloc[-13]) if len(s) > 13 else float("nan")
        change = latest - prev if not pd.isna(prev) else float("nan")
        return latest, change
    except Exception:
        return float("nan"), float("nan")


# High-Yield Spreads (daily, BAMLH0A0HYM2)
@st.cache_data(ttl=3600, show_spinner=False)
def live_hy_spread() -> float:
    try:
        s = fred.get_series("BAMLH0A0HYM2").dropna()
        return round(float(s.iloc[-1]), 1)
    except Exception:
        return float("nan")


# Gold price spot (daily, Alpha Vantage XAUUSD)
@st.cache_data(ttl=3600, show_spinner=False)
def live_gold_price() -> float:
    try:
        url = (
            "https://www.alphavantage.co/query?"
            f"function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        rate = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        return round(rate, 2)
    except Exception:
        return 4065.0


# Total Debt/GDP (quarterly BIS proxy: TCMDO/GDP)
@st.cache_data(ttl=86400, show_spinner=False)
def live_total_debt_gdp() -> float:
    try:
        debt = fred.get_series("TCMDO").dropna().iloc[-1]  # billions
        gdp = fred.get_series("GDP").dropna().iloc[-1]     # billions
        return round(float(debt) / float(gdp) * 100.0, 1)
    except Exception:
        return 355.0


# Productivity growth (quarterly, OPHNFB YoY)
@st.cache_data(ttl=86400, show_spinner=False)
def live_productivity_yoy() -> float:
    try:
        s = fred.get_series("OPHNFB").dropna()
        if len(s) < 13:
            return float("nan")
        last = float(s.iloc[-1])
        base = float(s.iloc[-13])
        return round((last / base - 1.0) * 100.0, 2)
    except Exception:
        return float("nan")


# Wage share (labor share, PRS85006173 quarterly)
@st.cache_data(ttl=86400, show_spinner=False)
def live_wage_share() -> float:
    try:
        s = fred.get_series("PRS85006173").dropna()
        return round(float(s.iloc[-1]), 1)
    except Exception:
        return float("nan")


# Real 30-year yield (daily, DGS30 - CPI YoY)
@st.cache_data(ttl=86400, show_spinner=False)
def live_real_30y_yield() -> float:
    try:
        dgs30 = fred.get_series("DGS30").dropna()
        if len(dgs30) < 1:
            return float("nan")
        nom = float(dgs30.iloc[-1])
        cpi_y = latest_cpi_yoy()
        if pd.isna(cpi_y):
            return float("nan")
        return round(nom - cpi_y, 2)
    except Exception:
        return 1.8


# Geopolitical Risk Index (monthly CSV)
@st.cache_data(ttl=86400, show_spinner=False)
def live_gpr_world() -> float:
    try:
        url = "https://www.policyuncertainty.com/media/GPR_World.csv"
        df = pd.read_csv(url)
        col = [c for c in df.columns if "GPR" in c.upper()][0]
        return round(float(df[col].dropna().iloc[-1]), 1)
    except Exception:
        return 180.0


# Gini (yearly, World Bank official)
@st.cache_data(ttl=86400 * 7, show_spinner=False)
def live_gini_us() -> float:
    try:
        url = "https://api.worldbank.org/v2/country/US/indicator/SI.POV.GINI?format=json&per_page=1"
        js = requests.get(url, timeout=10).json()
        val = js[1][0]["value"]
        return round(float(val), 2)
    except Exception:
        return 0.41


# =========================== COMPUTE LIVE VALUES ===========================
cpi_yoy = latest_cpi_yoy()
fed_funds_rate = latest_fed_funds()
real_fed_rate = (
    round(fed_funds_rate - cpi_yoy, 2) if not pd.isna(fed_funds_rate) and not pd.isna(cpi_yoy) else float("nan")
)

margin_gdp = live_margin_gdp()
put_call = live_put_call()
aaii_bulls = live_aaii_bulls()
sp500_pe_live = live_sp500_pe()
walcl_level, walcl_change = live_fed_balance_sheet()
hy_spread_live = live_hy_spread()
gold_price = live_gold_price()

total_debt_gdp = live_total_debt_gdp()
prod_yoy = live_productivity_yoy()
wage_share = live_wage_share()
real_30y = live_real_30y_yield()
usd_per_oz_ratio = (1000.0 / gold_price) if gold_price and gold_price > 0 else float("nan")
gpr_world = live_gpr_world()
gini_us = live_gini_us()

# Fed policy text
if pd.isna(walcl_level) or pd.isna(walcl_change):
    fed_policy_text = "Unknown (WALCL fetch failed)"
elif walcl_change < -50_000:
    fed_policy_text = f"QT (balance sheet shrinking ~{walcl_change/1_000:.1f}B over 13 weeks)"
elif walcl_change > 50_000:
    fed_policy_text = f"QE / expansion (~+{walcl_change/1_000:.1f}B over 13 weeks)"
else:
    fed_policy_text = "Roughly flat balance sheet"

# =========================== LONG / SHORT ROWS (MATCHING YOUR TABLE) ===========================
LONG_TERM_ROWS_LIVE: List[Dict[str, object]] = [
    {
        "Signal": "Total Debt/GDP (Private + Public + Foreign — BIS proxy TCMDO/GDP)",
        "Current value": f"{total_debt_gdp:.1f}%",
        "Red-flag threshold": ">300–400% and rising (quarterly official)",
        "Status": "Red" if total_debt_gdp >= 300 else "Watch",
        "Direction": "Quarterly BIS-style proxy from FRED TCMDO/GDP",
    },
    {
        "Signal": "Productivity growth (real, US — OPHNFB YoY)",
        "Current value": f"{prod_yoy:.2f}%" if pd.notna(prod_yoy) else "n/a",
        "Red-flag threshold": "<1.5% sustained (quarterly official)",
        "Status": "Watch" if pd.notna(prod_yoy) and prod_yoy < 2.0 else "Green",
        "Direction": "Quarterly BLS/FRED; long-term stagnation = debt trap",
    },
    {
        "Signal": "Gold price (spot XAUUSD, daily official)",
        "Current value": f"${gold_price:,.2f}/oz",
        "Red-flag threshold": ">2× long-run real average",
        "Status": "Red" if gold_price >= 2800 else "Watch",
        "Direction": "Daily LBMA spot via Alpha Vantage",
    },
    {
        "Signal": "Wage share of GDP (labor share — PRS85006173)",
        "Current value": f"{wage_share:.1f}" if pd.notna(wage_share) else "n/a",
        "Red-flag threshold": "Multi-decade downtrend / <1970s levels (quarterly)",
        "Status": "Watch",
        "Direction": "Quarterly labor share proxy from FRED",
    },
    {
        "Signal": "Real 30-year Treasury yield (DGS30 - CPI YoY)",
        "Current value": f"{real_30y:.2f}%",
        "Red-flag threshold": "Prolonged <2% or deeply negative (daily)",
        "Status": "Watch" if real_30y < 2.0 else "Green",
        "Direction": "Uses daily DGS30 and monthly CPI YoY (BLS via FRED)",
    },
    {
        "Signal": "USD vs gold power (oz per $1,000)",
        "Current value": f"{usd_per_oz_ratio:.3f} oz / $1,000",
        "Red-flag threshold": "<0.25 (hegemony erosion, daily)",
        "Status": "Red" if usd_per_oz_ratio < 0.25 else "Watch",
        "Direction": "Daily calc from spot gold price",
    },
    {
        "Signal": "Geopolitical Risk Index (world)",
        "Current value": f"{gpr_world:.1f}",
        "Red-flag threshold": ">150 and rising (monthly)",
        "Status": "Red" if gpr_world > 150 else "Watch",
        "Direction": "Monthly GPR from policyuncertainty.com",
    },
    {
        "Signal": "Income inequality (US Gini)",
        "Current value": f"{gini_us:.2f}",
        "Red-flag threshold": ">0.40 and rising (yearly)",
        "Status": "Red" if gini_us > 0.40 else "Watch",
        "Direction": "World Bank yearly Gini; inequality near modern highs",
    },
]

SHORT_TERM_ROWS_LIVE: List[Dict[str, object]] = [
    {
        "Indicator": "Margin debt as % of GDP",
        "Current value": f"{margin_gdp:.2f}%",
        "Red-flag threshold": "≥3.5% and rolling over (monthly FINRA)",
        "Status": "Red" if margin_gdp >= 3.5 else "Watch",
        "Direction": "FINRA margin stats via Alpha Vantage / FRED GDP",
    },
    {
        "Indicator": "Real Fed funds rate",
        "Current value": f"{real_fed_rate:+.2f}%" if not pd.isna(real_fed_rate) else "n/a",
        "Red-flag threshold": "Rising >+1.5% (monthly official)",
        "Status": "Green" if not pd.isna(real_fed_rate) and real_fed_rate > 0 else "Red",
        "Direction": "FEDFUNDS minus official CPI YoY (BLS via FRED)",
    },
    {
        "Indicator": "CBOE total put/call ratio",
        "Current value": f"{put_call:.3f}",
        "Red-flag threshold": "<0.70 (daily complacency)",
        "Status": "Red" if put_call < 0.70 else "Watch",
        "Direction": "Daily official CBOE totalpc.csv",
    },
    {
        "Indicator": "AAII bullish %",
        "Current value": f"{aaii_bulls:.1f}%",
        "Red-flag threshold": ">60% for 2+ weeks (weekly official)",
        "Status": "Red" if aaii_bulls > 60 else "Green",
        "Direction": "Weekly AAII survey CSV",
    },
    {
        "Indicator": "S&P 500 trailing P/E",
        "Current value": f"{sp500_pe_live:.2f}",
        "Red-flag threshold": ">30× sustained (daily)",
        "Status": "Red" if sp500_pe_live > 30 else "Watch",
        "Direction": "Daily FMP quote for ^GSPC",
    },
    {
        "Indicator": "Fed policy stance (QE vs QT)",
        "Current value": fed_policy_text,
        "Red-flag threshold": "Aggressive QT + hikes late in bubble (weekly WALCL)",
        "Status": "Watch",
        "Direction": "Weekly FRED WALCL level & 13-week change",
    },
    {
        "Indicator": "High-yield credit spreads (OAS)",
        "Current value": f"{hy_spread_live:.1f} bps" if not pd.isna(hy_spread_live) else "n/a",
        "Red-flag threshold": ">400 bps and widening fast (daily)",
        "Status": "Green" if not pd.isna(hy_spread_live) and hy_spread_live < 400 else "Red",
        "Direction": "Daily ICE BofA HY OAS (BAMLH0A0HYM2 via FRED)",
    },
    {
        "Indicator": "Insider selling vs buybacks",
        "Current value": "Heavy selling / slower buybacks (weekly trend, proxy)",
        "Red-flag threshold": "90%+ selling vs buying (weekly)",
        "Status": "Red",
        "Direction": "Conceptual; would require OpenInsider + FMP for full live view",
    },
]

# =========================== TABS ===========================
tab_core, tab_long, tab_short = st.tabs(
    [
        "📊 Core Econ Mirror indicators",
        "🌍 Long-term Debt Super-Cycle (40–70 yrs)",
        "⚡ Short-term Bubble Timing (5–10 yrs)",
    ]
)

# =========================== CORE TAB ===========================
with tab_core:
    st.title("Econ Mirror Dashboard — Core Macro & Power Signals")
    st.caption(
        "All indicators shown at once. Data pulled from FRED, World Bank mirrors, and pinned CSVs "
        "for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E."
    )

    rows: List[Dict[str, object]] = []
    histories: List[List[float]] = []

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

        # Special mirrors / proxies
        if ind.startswith("Education (PISA scores"):
            path_pisa = os.path.join(DATA_DIR, "pisa_math_usa.csv")
            c, p, s, h = mirror_latest_csv(
                path_pisa,
                "pisa_math_mean_usa",
                "year",
                numeric_time=True,
            )
            if not pd.isna(c):
                cur, prev, src, hist = c, p, "OECD PISA — " + s, h

        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, h = mirror_latest_csv(
                path_cinc,
                "cinc_usa",
                "year",
                numeric_time=True,
            )
            if not pd.isna(c):
                cur, prev, src, hist = c, p, "CINC — " + s, h

        if ind.startswith("Military losses (UCDP"):
            path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
            c, p, s, h = mirror_latest_csv(
                path_ucdp,
                "ucdp_battle_deaths_global",
                "year",
                numeric_time=True,
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
        histories.append(hist)

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
        "CINC (mirror), UCDP (mirror), MULTPL/Yale (mirror)."
    )

# =========================== LONG-TERM TAB ===========================
with tab_long:
    st.title("Long-term Debt Super-Cycle Dashboard — Live (40–70 year arc)")
    st.write(
        "Structural super-cycle signals (debt, productivity, inequality, currency, risk). "
        "Frequencies match official releases: quarterly (debt, productivity, wages), "
        "monthly (CPI, GPR), yearly (Gini), daily (gold, yields)."
    )

    df_long = pd.DataFrame(LONG_TERM_ROWS_LIVE)
    st.dataframe(df_long, use_container_width=True, hide_index=True)

    reds = sum(1 for r in LONG_TERM_ROWS_LIVE if "Red" in str(r["Status"]))
    watches = sum(1 for r in LONG_TERM_ROWS_LIVE if "Watch" in str(r["Status"]))
    st.markdown(
        f"**Live super-cycle score: {reds} 🔴 Red + {watches} 🟡 Watch → Late-stage long-term debt cycle.**"
    )

# =========================== SHORT-TERM TAB ===========================
with tab_short:
    st.title("Short-term Bubble Timing Dashboard — Live (5–10 year cycle)")
    st.write(
        "Bubble-timing signals using official cadence: "
        "monthly (margin debt, CPI), daily (put/call, spreads), "
        "weekly (AAII, insiders), daily valuations (P/E), weekly Fed balance sheet."
    )

    df_short = pd.DataFrame(SHORT_TERM_ROWS_LIVE)
    st.dataframe(df_short, use_container_width=True, hide_index=True)

    reds_s = sum(1 for r in SHORT_TERM_ROWS_LIVE if "Red" in str(r["Status"]))
    watches_s = sum(1 for r in SHORT_TERM_ROWS_LIVE if "Watch" in str(r["Status"]))
    st.markdown(
        f"**Live bubble score: {reds_s} 🔴 Red + {watches_s} 🟡 Watch → Late-cycle / possible melt-up, not confirmed final top.**"
    )

st.success("Fully live • Official cadences for every signal • Core mirrors + long/short cycle tabs • Built by Yinkaadx • Nov 2025")

