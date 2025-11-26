from __future__ import annotations

import os
import re
import datetime as dt
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import wbdata
import yfinance as yf
from bs4 import BeautifulSoup
from fredapi import Fred

# =============================================================================
# SECRETS (stored in .streamlit/secrets.toml on Streamlit Cloud)
# =============================================================================

FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

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
        margin-bottom: 1rem;
    }
    .regime-banner {
        padding: 0.75rem 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #444;
        background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .badge.seed {
        background: #8e44ad;
        color: #fff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        margin-left: 8px;
    }
    .status-red {color: #ff4444; font-weight: bold; font-size: 1.0rem;}
    .status-yellow {color: #ffbb33; font-weight: bold; font-size: 1.0rem;}
    .status-green {color: #00C851; font-weight: bold; font-size: 1.0rem;}
    .kill-box {
        border-radius: 0.75rem;
        border: 1px solid #ff4444;
        background: rgba(255, 68, 68, 0.06);
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0 0.75rem 0;
        font-size: 0.95rem;
    }
    .info-box-soft {
        border-radius: 0.75rem;
        border: 1px solid #555;
        background: rgba(255,255,255,0.02);
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0 0.75rem 0;
        font-size: 0.9rem;
    }
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricLabel"] {font-size: 1.1rem !important;}
    [data-testid="stMetricValue"] {font-size: 2.2rem !important;}
    [data-testid="stDataFrame"] [data-testid="cell-container"] {white-space: normal !important;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">ECON MIRROR</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Live Macro + Cycle Dashboard — Auto-updates hourly — Nov 2025</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="regime-banner">
<b>Current regime:</b> Late-stage melt-up (short-term) inside late-stage debt super-cycle (long-term). 
Ride stocks with <b>20–30% cash</b> + <b>30–40% gold/BTC</b> permanent.
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

# =============================================================================
# INDICATORS / THRESHOLDS / UNITS (from your original app)
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
# HELPERS — GENERAL
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


# =============================================================================
# HELPERS — FRED & MIRRORS (rate-limit safe)
# =============================================================================


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
        return pd.Series(dtype=float)


def yoy_from_series(s: pd.Series) -> Tuple[float, float]:
    if s.empty:
        return float("nan"), float("nan")
    last = to_float(s.iloc[-1])
    last_date = pd.to_datetime(s.index[-1])
    try:
        idx = s.index.get_indexer([last_date - timedelta(days=365)], method="nearest")[0]
        base = to_float(s.iloc[idx])
    except Exception:
        return float("nan"), float("nan")
    if pd.isna(base) or base == 0:
        return float("nan"), float("nan")
    current_yoy = (last / base - 1.0) * 100.0
    prev_yoy = float("nan")
    if len(s) > 2:
        last2 = to_float(s.iloc[-2])
        last_date2 = pd.to_datetime(s.index[-2])
        try:
            idx2 = s.index.get_indexer([last_date2 - timedelta(days=365)], method="nearest")[0]
            base2 = to_float(s.iloc[idx2])
            if not pd.isna(base2) and base2 != 0:
                prev_yoy = (last2 / base2 - 1.0) * 100.0
        except Exception:
            prev_yoy = float("nan")
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
            try:
                idx = s.index.get_indexer([ld - timedelta(days=365)], method="nearest")[0]
            except Exception:
                continue
            base = to_float(s.iloc[idx])
            val = to_float(s.iloc[j])
            if pd.isna(base) or base == 0:
                continue
            vals.append((val / base - 1.0) * 100.0)
        vals = [v for v in vals if v is not None]
        return vals[-n:]
    return pd.to_numeric(s.tail(n).values, errors="coerce").astype(float).tolist()


# =============================================================================
# HELPERS — WORLD BANK & MIRROR SPECIALS
# =============================================================================


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
        return cur, prev, "WB (online)", hist
    except Exception:
        return float("nan"), float("nan"), "—", []


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


def sp500_pe_mirror_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "pe_sp500.csv")
    return mirror_latest_csv(path, "pe", "date", numeric_time=False)


# =============================================================================
# THRESHOLD PARSING
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
# NEW LIVE HELPERS — FINRA, OPENINSIDER, WALCL, SPXA200R, SOFR
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_margin_debt_finra() -> Tuple[float, float]:
    """
    Returns latest and previous FINRA margin debt in USD billions.
    Uses FINRA live page; falls back to local mirror if needed.
    """
    url = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
    latest_bil = float("nan")
    prev_bil = float("nan")

    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            data_rows: List[float] = []
            for row in rows[1:]:
                cols = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cols) < 2:
                    continue
                raw_val = cols[1].replace("$", "").replace(",", "")
                try:
                    val = float(raw_val) / 1e9  # USD → billions
                    data_rows.append(val)
                except Exception:
                    continue
            if len(data_rows) >= 1:
                latest_bil = float(data_rows[0])
            if len(data_rows) >= 2:
                prev_bil = float(data_rows[1])
    except Exception:
        pass

    # Mirror fallback: data/margin_finra.csv with columns [date,debit_bil]
    if pd.isna(latest_bil) or pd.isna(prev_bil):
        path = os.path.join(DATA_DIR, "margin_finra.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "debit_bil" in df.columns and not df.empty:
                df = df.dropna(subset=["debit_bil"]).sort_values(df.columns[0], ascending=False)
                latest_bil = float(df.iloc[0]["debit_bil"])
                if len(df) > 1:
                    prev_bil = float(df.iloc[1]["debit_bil"])

    # Final fallback if nothing worked
    if pd.isna(latest_bil):
        latest_bil = 1180.0
    if pd.isna(prev_bil):
        prev_bil = latest_bil

    return latest_bil, prev_bil


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_insider_ratio_openinsider() -> float:
    """
    Returns insider buy ratio = buys / (buys + sells) * 100 (%)
    using OpenInsider latest page.
    """
    url = "http://openinsider.com/latest-insider-trading"
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        buys = 0
        sells = 0
        for td in soup.find_all("td"):
            txt = td.get_text(strip=True)
            if txt.startswith("P - Purchase"):
                buys += 1
            elif txt.startswith("S - Sale"):
                sells += 1

        total = buys + sells
        if total <= 0:
            return 8.0

        return round(buys / total * 100.0, 1)
    except Exception:
        return 8.0


@st.cache_data(ttl=43200, show_spinner=False)
def live_fed_balance_yoy() -> float:
    """
    YoY change of Fed balance sheet (WALCL) in %.
    Uses FRED mirror helper; if not available, returns 0.0.
    """
    s = fred_series("WALCL")
    if s.empty:
        return 0.0
    s = s.dropna()
    latest = float(s.iloc[-1])
    if len(s) > 52:
        base = float(s.iloc[-53])
    else:
        base = float(s.iloc[0])
    if base == 0:
        return 0.0
    yoy = (latest / base - 1.0) * 100.0
    return round(yoy, 2)


@st.cache_data(ttl=1800, show_spinner=False)
def live_spx_above_200ma() -> float:
    """
    Approximate % of S&P 500 stocks above 200-day MA using StockCharts SPXA200R.
    Falls back to local mirror if scraping fails.
    """
    url = "https://stockcharts.com/sc3/ui/?s=%24SPXA200R"
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        html = resp.text
        m = re.search(r'"Last"\s*:\s*"([\d\.]+)"', html)
        if m:
            return float(m.group(1))
        m2 = re.search(r"Value[^0-9]*([\d\.]+)%", html)
        if m2:
            return float(m2.group(1))
    except Exception:
        pass

    path = os.path.join(DATA_DIR, "spx_above_200.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "percent" in df.columns and not df.empty:
            df = df.dropna(subset=["percent"]).sort_values(df.columns[0])
            return float(df.iloc[-1]["percent"])

    return 50.0


@st.cache_data(ttl=43200, show_spinner=False)
def live_sofr_spread() -> float:
    """
    SOFR - FEDFUNDS (both from FRED via fred_series).
    Returns spread in percentage points.
    """
    try:
        sofr = fred_series("SOFR")
        ff = fred_series("FEDFUNDS")
        df = pd.concat([sofr, ff], axis=1).dropna()
        if df.empty:
            return 0.0
        df.columns = ["sofr", "ff"]
        last = df.iloc[-1]
        spread = float(last["sofr"]) - float(last["ff"])
        return round(spread, 3)
    except Exception:
        return 0.0


# =============================================================================
# LIVE DATA — SHORT-TERM (margin, AAII, put/call, P/E, HY, VIX, SPX)
# =============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def live_margin_gdp_details() -> Tuple[float, float]:
    """
    Returns:
      - current margin debt as % of GDP
      - month-over-month change in percentage points
    Uses FINRA live page + FRED GDP, with mirror fallbacks.
    """
    try:
        margin_latest_bil, margin_prev_bil = fetch_margin_debt_finra()

        gdp_series = fred_series("GDP")
        if gdp_series.empty:
            gdp_latest_tril = 28.8
            gdp_prev_tril = 28.0
        else:
            gdp_series = gdp_series.dropna()
            gdp_latest_tril = float(gdp_series.iloc[-1]) / 1_000.0
            gdp_prev_tril = float(
                gdp_series.iloc[-2] if len(gdp_series) > 1 else gdp_series.iloc[-1]
            ) / 1_000.0

        if gdp_latest_tril <= 0 or gdp_prev_tril <= 0:
            gdp_latest_tril = 28.8
            gdp_prev_tril = 28.0

        cur_pct = margin_latest_bil / gdp_latest_tril * 100.0
        prev_pct = margin_prev_bil / gdp_prev_tril * 100.0
        delta_pp = cur_pct - prev_pct

        return round(cur_pct, 2), round(delta_pp, 2)
    except Exception:
        return 3.88, 0.0


@st.cache_data(ttl=3600)
def live_put_call_details() -> Tuple[float, float, List[float]]:
    try:
        df = pd.read_csv(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv",
            skiprows=2,
        )
        df = df.dropna()
        df["ratio"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df = df.dropna(subset=["ratio"])
        df = df.sort_values(df.columns[0])
        ratios = df["ratio"].tolist()
        if not ratios:
            return 0.87, 0.87, []
        last5 = ratios[-5:]
        last5_rev = last5[::-1]
        latest = last5_rev[0]
        avg5 = sum(last5) / len(last5)
        return round(latest, 3), round(avg5, 3), [round(v, 3) for v in last5_rev]
    except Exception:
        return 0.87, 0.87, []


@st.cache_data(ttl=7200)
def live_aaii_bulls_details() -> Tuple[float, List[float], List[float]]:
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
            return 32.6, [], []
        last = bulls[0]
        last4 = bulls[:4]
        return float(last), [round(v, 1) for v in last4], [round(v, 1) for v in bulls]
    except Exception:
        return 32.6, [], []


@st.cache_data(ttl=3600)
def live_sp500_pe_live() -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        return round(float(j[0]["pe"]), 2)
    except Exception:
        c, _, _, _ = sp500_pe_mirror_latest()
        if not pd.isna(c):
            return round(c, 2)
        return 29.82


@st.cache_data(ttl=3600)
def live_hy_spread_details() -> Tuple[float, float]:
    s = fred_series("BAMLH0A0HYM2")
    if s.empty:
        return 317.0, 0.0
    s = s.dropna()
    latest = float(s.iloc[-1])
    if len(s) > 22:
        prev = float(s.iloc[-22])
    else:
        prev = float(s.iloc[0])
    delta = latest - prev
    return round(latest, 1), round(delta, 1)


@st.cache_data(ttl=3600)
def live_vix_level() -> float:
    try:
        data = yf.download("^VIX", period="10d", interval="1d", progress=False)
        closes = data["Close"].dropna()
        if closes.empty:
            return 15.0
        return round(float(closes.iloc[-1]), 2)
    except Exception:
        return 15.0


@st.cache_data(ttl=3600)
def live_spx_level_and_ath() -> Tuple[float, float]:
    try:
        data = yf.download("^GSPC", period="10y", interval="1d", progress=False)
        closes = data["Close"].dropna()
        if closes.empty:
            return 5000.0, 5000.0
        last = float(closes.iloc[-1])
        ath = float(closes.max())
        return round(last, 2), round(ath, 2)
    except Exception:
        return 5000.0, 5000.0


@st.cache_data(ttl=3600)
def live_spx_ytd_info() -> Tuple[bool, float]:
    try:
        today = dt.date.today()
        start = dt.date(today.year, 1, 1)
        data = yf.download("^GSPC", start=start, interval="1d", progress=False)
        closes = data["Close"].dropna()
        if closes.empty:
            return True, 0.0
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        ytd_ret = (last / first - 1.0) * 100.0
        return (ytd_ret >= 0.0), round(ytd_ret, 2)
    except Exception:
        return True, 0.0


@st.cache_data(ttl=3600)
def live_real_fed_rate_official() -> Tuple[float, float]:
    try:
        ff_series = fred_series("FEDFUNDS")
        cpi_series = fred_series("CPIAUCSL").pct_change(12) * 100.0
        df = pd.concat([ff_series, cpi_series], axis=1).dropna()
        if df.empty:
            return 1.07, 0.0
        df.columns = ["ff", "cpi_yoy"]
        df["real"] = df["ff"] - df["cpi_yoy"]
        latest = float(df["real"].iloc[-1])
        prev = float(df["real"].iloc[-2]) if len(df) > 1 else latest
        return round(latest, 2), round(prev, 2)
    except Exception:
        return 1.07, 0.0


# =============================================================================
# LIVE DATA — LONG-TERM (gold, real 30y, total debt/gdp, GPR, Gini, wage share, productivity)
# =============================================================================


@st.cache_data(ttl=3600)
def live_gold_price_usd() -> float:
    try:
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        )
        j = SESSION.get(url, timeout=10).json()
        rate = float(j["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        return round(rate, 0)
    except Exception:
        return 2400.0


@st.cache_data(ttl=10800)
def gold_ath_all_majors() -> Tuple[bool, dict]:
    symbols = {
        "USD": "XAUUSD=X",
        "EUR": "XAUEUR=X",
        "JPY": "XAUJPY=X",
        "CNY": "XAUCNY=X",
        "GBP": "XAUGBP=X",
        "CHF": "XAUCHF=X",
    }
    info = {}
    try:
        data = yf.download(list(symbols.values()), period="10y", interval="1d", progress=False)["Close"]
        for ccy, sym in symbols.items():
            s = data[sym].dropna()
            if s.empty:
                continue
            latest = float(s.iloc[-1])
            max_hist = float(s.max())
            at_ath = latest >= 0.995 * max_hist
            info[ccy] = {
                "latest": round(latest, 2),
                "max": round(max_hist, 2),
                "at_ath": at_ath,
            }
        all_at_ath = bool(info) and all(v["at_ath"] for v in info.values())
        return all_at_ath, info
    except Exception:
        return False, {}


@st.cache_data(ttl=3600)
def real_30y_extreme_months() -> Tuple[float, bool]:
    try:
        nom = fred_series("DGS30")
        cpi = fred_series("CPIAUCSL").pct_change(12) * 100.0
        df = pd.concat([nom, cpi], axis=1).dropna()
        if df.empty:
            return 1.82, False
        df.columns = ["nom", "cpi_yoy"]
        df["real30"] = df["nom"] - df["cpi_yoy"]
        latest = float(df["real30"].iloc[-1])
        recent = df["real30"].tail(60)
        if recent.empty:
            extreme = False
        else:
            extreme = (recent > 5.0).all() or (recent < -5.0).all()
        return round(latest, 2), bool(extreme)
    except Exception:
        return 1.82, False


@st.cache_data(ttl=43200)
def live_total_debt_gdp_ratio() -> Tuple[float, float]:
    try:
        debt = fred_series("TCMDO")
        gdp = fred_series("GDP")
        df = pd.concat([debt, gdp], axis=1).dropna()
        if df.empty:
            return 355.0, 0.0
        df.columns = ["debt", "gdp"]
        df["ratio"] = df["debt"] / df["gdp"] * 100.0
        ratio_latest = float(df["ratio"].iloc[-1])
        window = min(len(df), 12)
        recent = df["ratio"].tail(window)
        x = list(range(window))
        y = recent.values
        x_mean = sum(x) / window
        y_mean = float(sum(y) / window)
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den = sum((xi - x_mean) ** 2 for xi in x)
        slope = num / den if den != 0 else 0.0
        return round(ratio_latest, 1), round(slope, 3)
    except Exception:
        return 355.0, 0.0


@st.cache_data(ttl=43200)
def live_gpr_global_est() -> Tuple[float, bool]:
    try:
        df = pd.read_csv("https://www.policyuncertainty.com/media/GPR_Global_Data.csv")
        date_col = df.columns[0]
        val_col = df.columns[1]
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["date", val_col]).sort_values("date")
        series = df[val_col]
        if series.empty:
            return 180.0, False
        latest = float(series.iloc[-1])
        recent = series.tail(12)
        if len(recent) < 12:
            vertical = False
        else:
            first_half = float(recent.iloc[:6].mean())
            second_half = float(recent.iloc[6:].mean())
            vertical = (second_half - first_half) > 50.0
        return round(latest, 1), bool(vertical)
    except Exception:
        return 180.0, False


@st.cache_data(ttl=86400)
def live_gini_and_trend() -> Tuple[float, bool]:
    cur, prev, _, _ = wb_last_two("SI.POV.GINI", "USA")
    if pd.isna(cur):
        return 0.41, False
    return round(cur, 3), bool(cur > prev)


@st.cache_data(ttl=43200)
def live_wage_share_trend() -> Tuple[float, bool]:
    s = fred_series("LABSHPUSA156NRUG")
    if s.empty:
        return 52.0, False
    s = s.dropna()
    latest = float(s.iloc[-1])
    window = min(len(s), 12)
    recent = s.tail(window)
    x = list(range(window))
    y = recent.values
    x_mean = sum(x) / window
    y_mean = float(sum(y) / window)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    slope = num / den if den != 0 else 0.0
    downtrend = slope < 0.0
    return round(latest, 1), bool(downtrend)


@st.cache_data(ttl=43200)
def live_productivity_multi_year() -> Tuple[float, bool]:
    s = fred_series("OPHNFB")
    if s.empty:
        return 0.5, False
    s = s.dropna()
    if len(s) > 4:
        last = float(s.iloc[-1])
        base = float(s.iloc[-5])
        yoy = (last / base - 1.0) * 100.0 if base != 0 else 0.0
    else:
        yoy = 0.5
    neg_streak = 0
    for i in range(len(s) - 1, -1, -1):
        if i - 1 < 0:
            break
        g = s.iloc[i] - s.iloc[i - 1]
        if g < 0:
            neg_streak += 1
        else:
            break
    negative_years = neg_streak >= 6
    return round(yoy, 2), bool(negative_years)


@st.cache_data(ttl=43200)
def live_us10y_and_cpi() -> Tuple[float, float]:
    try:
        dgs10 = fred_series("DGS10")
        cpi = fred_series("CPIAUCSL").pct_change(12) * 100.0
        if dgs10.empty or cpi.empty:
            return 4.0, 3.0
        u = float(dgs10.dropna().iloc[-1])
        c = float(cpi.dropna().iloc[-1])
        return round(u, 2), round(c, 2)
    except Exception:
        return 4.0, 3.0


# =============================================================================
# RSS & CENTRAL BANK GOLD SIGNALS
# =============================================================================


@st.cache_data(ttl=3600)
def fetch_rss_keywords(url: str, keywords: List[str]) -> List[str]:
    hits: List[str] = []
    try:
        resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is None or not title_el.text:
                continue
            title = title_el.text.strip()
            low = title.lower()
            if any(k.lower() in low for k in keywords):
                hits.append(title)
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title_el = entry.find("{http://www.w3.org/2005/Atom}title")
            if title_el is None or not title_el.text:
                continue
            title = title_el.text.strip()
            low = title.lower()
            if any(k.lower() in low for k in keywords):
                hits.append(title)
    except Exception:
        return []
    return hits


@st.cache_data(ttl=43200)
def central_bank_gold_tonnage_increase_flag() -> Tuple[bool, str]:
    if not TE_KEY:
        return False, "TE_KEY missing — cannot check PBOC gold reserves; using RSS only."
    try:
        url = f"https://api.tradingeconomics.com/historical/country/china/indicator/gold%20reserves?c={TE_KEY}"
        df = pd.read_json(url)
        if df.empty:
            return False, "No data from TradingEconomics gold reserves."
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime", "Value"]).sort_values("DateTime")
        if len(df) < 2:
            return False, "Not enough data points for PBOC gold reserves."
        latest = float(df["Value"].iloc[-1])
        prev = float(df["Value"].iloc[-2])
        increased = latest > prev
        return bool(increased), f"PBOC gold reserves latest={latest}, prev={prev}."
    except Exception:
        return False, "TradingEconomics gold reserves fetch failed; rely on RSS gold-buying headlines."


@st.cache_data(ttl=3600)
def supercycle_news_alerts() -> Tuple[List[str], List[str]]:
    feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/worldNews",
        "https://www.ft.com/?format=rss",
    ]
    cb_keywords = ["central bank gold", "gold reserves", "buys gold", "adds gold", "gold purchases"]
    system_keywords = [
        "gold-backed currency",
        "gold backed currency",
        "BRICS currency",
        "new reserve currency",
        "de-dollarization",
        "dedollarisation",
    ]
    cb_hits: List[str] = []
    sys_hits: List[str] = []
    for url in feeds:
        cb_hits.extend(fetch_rss_keywords(url, cb_keywords))
        sys_hits.extend(fetch_rss_keywords(url, system_keywords))
    cb_hits = list(dict.fromkeys(cb_hits))
    sys_hits = list(dict.fromkeys(sys_hits))
    return cb_hits, sys_hits


# =============================================================================
# LIVE VALUES COMMON
# =============================================================================

# Short-term values
margin_gdp_cur, margin_gdp_delta = live_margin_gdp_details()
put_call_cur, put_call_avg5, put_call_last5 = live_put_call_details()
aaii_cur, aaii_last4, aaii_full = live_aaii_bulls_details()
pe_live = live_sp500_pe_live()
hy_spread_live, hy_spread_delta = live_hy_spread_details()
vix_level = live_vix_level()
real_fed_latest, real_fed_prev = live_real_fed_rate_official()
spx_last, spx_ath = live_spx_level_and_ath()
spx_green_ytd, spx_ytd_ret = live_spx_ytd_info()

if spx_ath <= 0:
    spx_drawdown_pct = 0.0
else:
    spx_drawdown_pct = (spx_last / spx_ath - 1.0) * 100.0
near_ath = spx_drawdown_pct >= -8.0

# Long-term values
gold_spot = live_gold_price_usd()
usd_vs_gold_ratio = 1000.0 / gold_spot if gold_spot else float("nan")
real_30y_latest, real30_extreme_months = real_30y_extreme_months()
total_debt_gdp_est, total_debt_slope = live_total_debt_gdp_ratio()
gpr_est, gpr_vertical = live_gpr_global_est()
gini_latest, gini_climbing = live_gini_and_trend()
wage_share_latest, wage_share_down = live_wage_share_trend()
prod_yoy_latest, prod_negative_years = live_productivity_multi_year()
us10y_yield, cpi_yoy = live_us10y_and_cpi()
gold_all_ath, gold_fx_info = gold_ath_all_majors()
cb_gold_increase, cb_gold_debug = central_bank_gold_tonnage_increase_flag()
cb_gold_titles, gold_system_titles = supercycle_news_alerts()

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
        "and pinned CSVs for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E. "
        "Threshold column shows the red/green line each indicator is measured against."
    )

    rows: List[Dict[str, object]] = []

    for ind in INDICATORS:
        unit = UNITS.get(ind, "")
        cur: float = float("nan")
        prev: float = float("nan")
        src: str = "—"

        if ind in WB_US:
            c, p, s, _h = wb_last_two(WB_US[ind], "USA")
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
            c, p, s, _h = mirror_latest_csv(path_pisa, "pisa_math_mean_usa", "year", numeric_time=True)
            if not pd.isna(c):
                cur, prev, src = c, p, "OECD PISA — " + s
        if ind.startswith("Power index (CINC"):
            path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
            c, p, s, _h = mirror_latest_csv(path_cinc, "cinc_usa", "year", numeric_time=True)
            if not pd.isna(c):
                cur, prev, src = c, p, "CINC — " + s
        if ind.startswith("Military losses (UCDP"):
            path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
            c, p, s, _h = mirror_latest_csv(
                path_ucdp, "ucdp_battle_deaths_global", "year", numeric_time=True
            )
            if not pd.isna(c):
                cur, prev, src = c, p, "UCDP — " + s
        if ind.startswith("Reserve currency usage"):
            c, p, s, _h = cofer_usd_share_latest()
            if not pd.isna(c):
                cur, prev, src = c, p, s
        if ind == "P/E ratios":
            c, p, s, _h = sp500_pe_mirror_latest()
            if not pd.isna(c):
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
                src = "FRED (mirror/online)"

        threshold_txt = THRESHOLDS.get(ind, "—")
        signal_icon, _signal_cls = evaluate_signal(cur, threshold_txt)
        seed_badge = (
            " <span class='badge seed'>Pinned seed</span>" if "Pinned seed" in src else ""
        )

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

# =============================================================================
# LONG-TERM TAB — super-cycle dashboard with DARK RED + POINT OF NO RETURN
# =============================================================================

with tab_long:
    st.markdown("### 🌍 Long-Term Debt Super-Cycle — Live (40–70 years)")
    st.caption("Updates hourly • Official sources only • No daily noise • DARK RED = final stage")

    total_debt_dark = (total_debt_gdp_est >= 400.0) or (total_debt_slope <= 0.0)
    gold_dark = gold_all_ath
    usd_vs_gold_dark = usd_vs_gold_ratio < 0.10 if not pd.isna(usd_vs_gold_ratio) else False
    real30_dark = real30_extreme_months
    gpr_dark = (gpr_est > 300.0) and gpr_vertical
    gini_dark = (gini_latest > 0.50) and gini_climbing
    wage_share_dark = (wage_share_latest < 50.0) and wage_share_down
    productivity_dark = prod_negative_years

    long_rows = [
        {
            "Signal": "1. Total Debt/GDP (private + public + foreign)",
            "Current value": f"≈{total_debt_gdp_est:.1f}%",
            "Red-flag threshold": ">400–450% or stops rising",
            "Status": "🔴 DARK RED" if total_debt_dark else "🟡 Watch",
            "Why this matters": "Debt >4× GDP or flatlining from defaults/inflation = classic pre-reset zone.",
        },
        {
            "Signal": "2. Gold (real) vs major currencies",
            "Current value": f"${gold_spot:,.0f}/oz (ATH in majors: {'YES' if gold_all_ath else 'NO'})",
            "Red-flag threshold": "Breaks to ATH vs USD, EUR, JPY, CNY, GBP, CHF",
            "Status": "🔴 DARK RED" if gold_dark else "🟡 Watch",
            "Why this matters": "When gold rips vs every currency, people are exiting paper systems globally.",
        },
        {
            "Signal": "3. USD vs gold ratio",
            "Current value": f"{usd_vs_gold_ratio:.3f} oz per $1,000",
            "Red-flag threshold": "<0.10 oz per $1,000 (4:1 or worse)",
            "Status": "🔴 DARK RED" if usd_vs_gold_dark else "🟡 Watch",
            "Why this matters": "Shows how much real value the dollar still holds vs hard money.",
        },
        {
            "Signal": "4. Real 30-year yield",
            "Current value": f"{real_30y_latest:.2f}%",
            "Red-flag threshold": ">+5% or <−5% for months",
            "Status": "🔴 DARK RED" if real30_dark else "🟡 Watch",
            "Why this matters": "Extremes in real yields are how systems are reset (repression or blow-ups).",
        },
        {
            "Signal": "5. Geopolitical Risk Index (GPR)",
            "Current value": f"{gpr_est:.1f}",
            "Red-flag threshold": ">300 and rising vertically",
            "Status": "🔴 DARK RED" if gpr_dark else "🟡 Watch",
            "Why this matters": "Wars, trade wars, and currency wars together always accompany big resets.",
        },
        {
            "Signal": "6. Gini coefficient (inequality)",
            "Current value": f"{gini_latest:.3f}",
            "Red-flag threshold": ">0.50 and still climbing",
            "Status": "🔴 DARK RED" if gini_dark else "🟡 Watch",
            "Why this matters": "High and rising inequality = revolution territory.",
        },
        {
            "Signal": "7. Wage share of GDP",
            "Current value": f"{wage_share_latest:.1f}%",
            "Red-flag threshold": "<50% with downtrend",
            "Status": "🔴 DARK RED" if wage_share_dark else "🟡 Watch",
            "Why this matters": "Collapsing wage share means most gains go to capital, not workers.",
        },
        {
            "Signal": "8. Productivity growth",
            "Current value": f"{prod_yoy_latest:.2f}% YoY (neg-years={prod_negative_years})",
            "Red-flag threshold": "Negative for multiple years",
            "Status": "🔴 DARK RED" if productivity_dark else "🟡 Watch",
            "Why this matters": "Without productivity, you can’t grow out of debt — only default or inflate.",
        },
    ]

    df_long = pd.DataFrame(long_rows)
    st.dataframe(df_long, use_container_width=True, hide_index=True)

    dark_red_flags = [
        total_debt_dark,
        gold_dark,
        usd_vs_gold_dark,
        real30_dark,
        gpr_dark,
        gini_dark,
        wage_share_dark,
        productivity_dark,
    ]
    dark_red_count = sum(1 for b in dark_red_flags if b)

    us10y_forced_default = (us10y_yield >= 7.0) and (cpi_yoy >= 3.0)
    cb_gold_trigger = cb_gold_increase or bool(cb_gold_titles)
    gold_system_trigger = bool(gold_system_titles)
    no_return_trigger = us10y_forced_default or cb_gold_trigger or gold_system_trigger

    st.write(
        f"**Dark red signals active: {dark_red_count}/8 • No-return trigger present: "
        f"{'YES' if no_return_trigger else 'NO'}**"
    )

    super_cycle_reset_state = (dark_red_count >= 6) and no_return_trigger

    if super_cycle_reset_state:
        st.markdown(
            """
<div class="kill-box">
<b>SUPER-CYCLE POINT OF NO RETURN:</b><br>
6+ dark-red long-term signals <b>AND</b> at least one no-return trigger (central banks buying gold openly, 
G20 gold-backed system, or US 10Y >7–8% with high CPI).<br><br>
<b>Rule:</b> Go <b>80–100% gold/bitcoin/cash/hard assets</b> and avoid stocks/bonds for 5–15 years.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="info-box-soft">
Super-cycle not yet at final reset stage. Watch for <b>6+ DARK RED</b> long-term lights <b>plus</b> an official 
gold/currency regime change or US 10Y >7–8% with high CPI before calling the true reset.
</div>
""",
            unsafe_allow_html=True,
        )

    # === NEW COLLAPSIBLE SECTION (LONG-TERM) ===
    with st.expander("SUPER-CYCLE POINT OF NO RETURN (final 6-24 months before reset)", expanded=False):

        long_8_rows = [
            {
                "Signal": "Total Debt/GDP",
                "Current value": f"≈{total_debt_gdp_est:.1f}%",
                "DARK RED level": ">400–450% (or stops rising)",
                "Dark red?": "🔴" if total_debt_dark else "🟡",
            },
            {
                "Signal": "Gold ATH vs major currencies",
                "Current value": f"${gold_spot:,.0f}/oz (all-ccy ATH: {'YES' if gold_all_ath else 'NO'})",
                "DARK RED level": "Breaks new ATH vs EVERY major currency",
                "Dark red?": "🔴" if gold_dark else "🟡",
            },
            {
                "Signal": "USD vs gold ratio",
                "Current value": f"{usd_vs_gold_ratio:.3f} oz per $1,000",
                "DARK RED level": "<0.10 oz per $1,000",
                "Dark red?": "🔴" if usd_vs_gold_dark else "🟡",
            },
            {
                "Signal": "Real 30-year yield",
                "Current value": f"{real_30y_latest:.2f}%",
                "DARK RED level": ">+5% OR <−5% for months",
                "Dark red?": "🔴" if real30_dark else "🟡",
            },
            {
                "Signal": "Geopolitical Risk Index (GPR)",
                "Current value": f"{gpr_est:.1f}",
                "DARK RED level": ">300 and rising vertically",
                "Dark red?": "🔴" if gpr_dark else "🟡",
            },
            {
                "Signal": "Gini coefficient",
                "Current value": f"{gini_latest:.3f}",
                "DARK RED level": ">0.50 and climbing",
                "Dark red?": "🔴" if gini_dark else "🟡",
            },
            {
                "Signal": "Wage share < 50%",
                "Current value": f"{wage_share_latest:.1f}%",
                "DARK RED level": "<50% of GDP",
                "Dark red?": "🔴" if wage_share_dark else "🟡",
            },
            {
                "Signal": "Productivity growth negative for multiple years",
                "Current value": f"{prod_yoy_latest:.2f}% YoY (negative quarters: {prod_negative_years})",
                "DARK RED level": "Productivity negative ≥ 6 consecutive quarters",
                "Dark red?": "🔴" if productivity_dark else "🟡",
            },
        ]

        df_long8 = pd.DataFrame(long_8_rows)
        st.dataframe(df_long8, use_container_width=True, hide_index=True)

        dark_red_8_flags = [
            total_debt_dark,
            gold_dark,
            usd_vs_gold_dark,
            real30_dark,
            gpr_dark,
            gini_dark,
            wage_share_dark,
            productivity_dark,
        ]
        dark_red_8_count = sum(1 for b in dark_red_8_flags if b)

        # 3 "Point of No Return" alerts
        if cb_gold_trigger:
            st.markdown(
                """
<div class="kill-box">
<b>Point of No Return — Central banks openly buying gold:</b><br>
Recent data/news show official gold reserve increases or explicit gold-buying announcements.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="info-box-soft">
Central bank gold-buying trigger not clearly active yet (tonnage + headlines below for manual review).
</div>
""",
                unsafe_allow_html=True,
            )

        if gold_system_trigger:
            st.markdown(
                """
<div class="kill-box">
<b>Point of No Return — G20/BRICS proposing new gold-backed system:</b><br>
Detected headlines on gold-backed currencies / new reserve systems. Manual verification recommended.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="info-box-soft">
Gold-backed / new currency system trigger not fully confirmed yet (see RSS debug for details).
</div>
""",
                unsafe_allow_html=True,
            )

        if us10y_forced_default:
            st.markdown(
                """
<div class="kill-box">
<b>Point of No Return — US 10Y >7–8% with high CPI:</b><br>
Bond market is pricing forced default / inflation dynamic. Classic final stage of debt super-cycle.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="info-box-soft">
US 10Y + CPI combination has not yet crossed the 7–8% forced-default zone.
</div>
""",
                unsafe_allow_html=True,
            )

        st.write(
            f"Dark red signals active: {dark_red_8_count}/8 + No-return trigger: "
            f"{'Yes' if no_return_trigger else 'No'}"
        )
        st.markdown(
            "**When 6+ dark red + one no-return trigger → go 80-100% gold/bitcoin/cash/hard assets "
            "and do not touch stocks/bonds for 5-15 years.**"
        )

    with st.expander("Debug logs — super-cycle RSS + yields", expanded=False):
        st.write("**US 10Y and CPI:**", {"us10y_yield": us10y_yield, "cpi_yoy": cpi_yoy})
        st.write("**PBOC gold reserves check:**", cb_gold_debug)
        if cb_gold_titles:
            st.write("**Central bank gold-buying headlines (RSS, manual verify):**")
            for t in cb_gold_titles:
                st.write(f"- {t}")
        if gold_system_titles:
            st.write("**Gold-backed / new currency system headlines (RSS, manual verify):**")
            for t in gold_system_titles:
                st.write(f"- {t}")
        if gold_fx_info:
            st.write("**Gold vs major currencies (FX info, last 10y):**")
            st.write(gold_fx_info)

# =============================================================================
# SHORT-TERM TAB — 6-of-8 KILL COMBO + STATE MACHINE
# =============================================================================

with tab_short:
    st.markdown("### ⚡ Short-Term Bubble Timing — Live (5–10 year cycle)")
    st.caption("Updates hourly • Official frequencies only • 6-of-8 kill combo + state machine")

    # AAII streak (>60% for 2+ weeks)
    aaii_streak = 0
    for v in aaii_full:
        if v > 60.0:
            aaii_streak += 1
        else:
            break
    aaii_kill = aaii_streak >= 2

    # Put/Call kill: at least 4 of last 5 days <0.65
    put_call_kill = False
    if len(put_call_last5) >= 5:
        put_call_kill = sum(1 for v in put_call_last5[:5] if v < 0.65) >= 4
    elif len(put_call_last5) >= 3:
        put_call_kill = all(v < 0.65 for v in put_call_last5[:3])

    # Margin kill: ≥3.5% and rolling over
    margin_kill = (margin_gdp_cur >= 3.5) and (margin_gdp_delta < 0.0)

    # Real Fed kill: ≥+1.5% and rising
    real_fed_kill = (real_fed_latest >= 1.5) and (real_fed_latest > real_fed_prev)

    # P/E kill: >30 AND first 4 kills active
    pe_kill = (pe_live > 30.0) and margin_kill and real_fed_kill and put_call_kill and aaii_kill

    # Insider kill: insider buying ratio <10% (via OpenInsider live)
    insider_buy_ratio = fetch_insider_ratio_openinsider() / 100.0
    insider_kill = insider_buy_ratio < 0.10

    # HY spread kill: <400 bps but widening 50+ bps in a month
    hy_kill = (hy_spread_live < 400.0) and (hy_spread_delta >= 50.0)

    # VIX kill: complacency <20
    vix_kill = vix_level < 20.0

    kill_signals = [
        {
            "ID": 1,
            "Kill name": "Margin Debt % GDP ≥3.5% and falling MoM",
            "Current": f"{margin_gdp_cur:.2f}% (Δ={margin_gdp_delta:+.2f} pp)",
            "Threshold": "≥3.5% and falling vs last month",
            "Kill active?": "🔴" if margin_kill else "🟢",
        },
        {
            "ID": 2,
            "Kill name": "Real Fed Funds Rate ≥+1.5% and rising",
            "Current": f"{real_fed_latest:+.2f}% (prev={real_fed_prev:+.2f}%)",
            "Threshold": "≥+1.5% and higher than last reading",
            "Kill active?": "🔴" if real_fed_kill else "🟢",
        },
        {
            "ID": 3,
            "Kill name": "CBOE Total Put/Call <0.65 for multiple days",
            "Current": f"{put_call_cur:.3f} (last5={put_call_last5})",
            "Threshold": "<0.65 for ≥4 of last 5 days",
            "Kill active?": "🔴" if put_call_kill else "🟢",
        },
        {
            "ID": 4,
            "Kill name": "AAII Bullish % >60% for 2+ weeks",
            "Current": f"{aaii_cur:.1f}% (streak={aaii_streak} weeks >60)",
            "Threshold": ">60% for ≥2 consecutive weeks",
            "Kill active?": "🔴" if aaii_kill else "🟢",
        },
        {
            "ID": 5,
            "Kill name": "S&P 500 trailing P/E >30 while 1–4 are flashing",
            "Current": f"{pe_live:.2f}x",
            "Threshold": ">30× AND kills 1–4 active",
            "Kill active?": "🔴" if pe_kill else "🟢",
        },
        {
            "ID": 6,
            "Kill name": "Insider buying ratio <10% (90%+ selling)",
            "Current": f"Buy ratio ≈{insider_buy_ratio*100:.1f}%",
            "Threshold": "<10% (90%+ selling)",
            "Kill active?": "🔴" if insider_kill else "🟢",
        },
        {
            "ID": 7,
            "Kill name": "High-yield spreads <400 bps but widening 50+ bps/month",
            "Current": f"{hy_spread_live:.1f} bps (Δ1m={hy_spread_delta:+.1f})",
            "Threshold": "<400 bps and Δ1m ≥50 bps",
            "Kill active?": "🔴" if hy_kill else "🟢",
        },
        {
            "ID": 8,
            "Kill name": "VIX still <20 (complacency)",
            "Current": f"{vix_level:.2f}",
            "Threshold": "<20 while bubble peaking",
            "Kill active?": "🔴" if vix_kill else "🟢",
        },
    ]

    df_kill = pd.DataFrame(kill_signals)
    st.dataframe(df_kill, use_container_width=True, hide_index=True)

    kill_count = sum(1 for row in kill_signals if row["Kill active?"] == "🔴")

    # Final top state: 6+ kills, S&P within -8% of ATH, and green YTD
    kill_top_state = (kill_count >= 6) and near_ath and spx_green_ytd

    # Panic bottom state: 6+ reds, market down ≥30%, VIX >50
    panic_bottom_state = (kill_count >= 6) and (spx_drawdown_pct <= -30.0) and (vix_level >= 50.0)

    if kill_top_state:
        short_state = "KILL_TOP"
    elif panic_bottom_state:
        short_state = "PANIC_BOTTOM"
    elif kill_count < 5:
        short_state = "FULLY_INVESTED"
    elif kill_count <= 4:
        short_state = "NEW_BULL"
    else:
        short_state = "INTERMEDIATE"

    st.write(
        f"**Current kill signals active: {kill_count}/8 • "
        f"S&P drawdown vs ATH: {spx_drawdown_pct:.1f}% • "
        f"Near ATH (>-8%): {'YES' if near_ath else 'NO'} • "
        f"YTD: {spx_ytd_ret:.1f}% ({'GREEN' if spx_green_ytd else 'RED'})**"
    )

    if short_state == "KILL_TOP":
        st.markdown(
            """
<div class="kill-box">
<b>Moment A — THE TOP:</b><br>
Short-term tab shows <b>6+ kill levels</b> while the S&P is still near its all-time high (within −8%) 
and green on the year.<br><br>
<b>Rule:</b> Sell <b>80–90% of stocks</b> this week and move new money into cash + gold/BTC. 
You will be out within ~5% of the final top.
</div>
""",
            unsafe_allow_html=True,
        )
    elif short_state == "PANIC_BOTTOM":
        st.markdown(
            """
<div class="kill-box">
<b>Moment B — THE PANIC BOTTOM:</b><br>
6–8 kill lights still red, but the market is already down 30–60% and VIX >50–80. 
Capitulation, forced liquidations, and panic headlines everywhere.<br><br>
<b>Rule:</b> Deploy <b>70–100% of the cash</b> raised in Moment A — buy stocks/commodities/BTC 
hand-over-fist at depressed prices.
</div>
""",
            unsafe_allow_html=True,
        )
    elif short_state == "FULLY_INVESTED":
        st.markdown(
            """
<div class="info-box-soft">
<b>Simple Timeline (1):</b> Short-term tab &lt;5 reds → stay fully invested in risk assets. 
You are still in the melt-up, not yet at the final 6-of-8 kill combo.
</div>
""",
            unsafe_allow_html=True,
        )
    elif short_state == "NEW_BULL":
        st.markdown(
            """
<div class="info-box-soft">
<b>Simple Timeline (4):</b> After the crash and panic bottom, when the short-term tab flips back to 
<b>4 or fewer reds</b> and the trend improves → new bull market confirmed → stay invested again.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="info-box-soft">
Interim zone between melt-up and full kill combo. Watch for the moment when 6+ kills flash while 
the S&P is still within −8% of ATH — that is the no-fake-out sell signal.
</div>
""",
            unsafe_allow_html=True,
        )

    # === NEW COLLAPSIBLE SECTION (SHORT-TERM) ===
    with st.expander("FINAL TOP KILL COMBO (6+ reds = sell 80-90% stocks this week)", expanded=False):
        st.dataframe(df_kill, use_container_width=True, hide_index=True)
        st.write(f"Current kill signals active: {kill_count}/8")
        st.markdown(
            """
<div class="kill-box">
<b>When 6+ are red AND S&P is within -8% of ATH → SELL 80-90% stocks this week.</b>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
Moment A (THE TOP): 6+ reds while market still high → sell instantly to cash/gold/BTC  
Moment B (THE BOTTOM): 6–18 months later, market down 30-60%, lights still red → buy aggressively with the cash
"""
        )

    with st.expander("Debug logs — short-term engines", expanded=False):
        st.write("**Raw values:**")
        st.write(
            {
                "margin_gdp_cur": margin_gdp_cur,
                "margin_gdp_delta": margin_gdp_delta,
                "put_call_cur": put_call_cur,
                "put_call_last5": put_call_last5,
                "aaii_cur": aaii_cur,
                "aaii_streak_weeks_over_60": aaii_streak,
                "pe_live": pe_live,
                "hy_spread_live_bps": hy_spread_live,
                "hy_spread_delta_bps": hy_spread_delta,
                "vix_level": vix_level,
                "real_fed_latest": real_fed_latest,
                "real_fed_prev": real_fed_prev,
                "spx_last": spx_last,
                "spx_ath": spx_ath,
                "spx_drawdown_pct": spx_drawdown_pct,
                "spx_ytd_ret": spx_ytd_ret,
                "spx_green_ytd": spx_green_ytd,
                "fed_balance_yoy_pct": live_fed_balance_yoy(),
                "spx_above_200ma_pct": live_spx_above_200ma(),
                "sofr_minus_fedfunds": live_sofr_spread(),
            }
        )
        st.write("**Kills active flags:**")
        st.write(
            {
                "margin_kill": margin_kill,
                "real_fed_kill": real_fed_kill,
                "put_call_kill": put_call_kill,
                "aaii_kill": aaii_kill,
                "pe_kill": pe_kill,
                "insider_kill": insider_kill,
                "hy_kill": hy_kill,
                "vix_kill": vix_kill,
                "kill_count": kill_count,
                "kill_top_state": kill_top_state,
                "panic_bottom_state": panic_bottom_state,
                "short_state": short_state,
            }
        )

# =============================================================================
# GLOBAL REGIME BANNER (BOTTOM) — uses actual kill_count & dark_red_count
# =============================================================================

short_total_signals = 8  # currently 8 kill signals implemented
long_total_signals = 8   # currently 8 long-term dark-red signals implemented

short_regime_text = "Late-stage melt-up"
if kill_top_state:
    short_regime_text = "FINAL TOP — sell 80–90% this week"
elif panic_bottom_state:
    short_regime_text = "PANIC BOTTOM — deploy raised cash aggressively"

long_regime_text = "Late debt super-cycle"
if super_cycle_reset_state:
    long_regime_text = "POINT OF NO RETURN — 80–100% hard assets"

st.markdown(
    f"""
<div class="regime-banner">
<b>Current regime snapshot:</b><br>
Short-term: {kill_count}/{short_total_signals} kill signals active — {short_regime_text}<br>
Long-term: {dark_red_count}/{long_total_signals} DARK RED super-cycle signals — {long_regime_text}
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Live data • Hourly refresh • Fallback mirrors • Built by Yinkaadx • Nov 2025")
