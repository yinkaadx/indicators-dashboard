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

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Econ Mirror Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Constants & data dirs
# ---------------------------------------------------------------------
DATA_DIR: str = "data"
WB_DIR: str = os.path.join(DATA_DIR, "wb")
FRED_DIR: str = os.path.join(DATA_DIR, "fred")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})

# FRED client (reads FRED_API_KEY from Streamlit secrets.toml)
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ---------------------------------------------------------------------
# CORE ECON INDICATORS / THRESHOLDS / UNITS
# ---------------------------------------------------------------------
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

# For the core table we just want “rough rule of thumb” thresholds.
# Only simple > / < rules are parsed into ✅ / ⚠️, everything else gets “—”.
THRESHOLDS: Dict[str, str] = {
    "Yield curve": "> 1",  # 10Y - 2Y in percentage points
    "Consumer confidence": "> 90",
    "Building permits": "> 5",
    "Unemployment claims": "< 300",
    "LEI (Conference Board Leading Economic Index)": "> 0",
    "GDP": "> 2",
    "Capacity utilization": "> 80",
    "Inflation": "2 <= x <= 3",
    "Retail sales": "> 3",
    "Nonfarm payrolls": "> 150",
    "Wage growth": "> 3",
    "P/E ratios": "> 20",
    "Credit growth": "> 5",
    "Fed funds futures": "> 50",
    "Short rates": "> 0",
    "Industrial production": "> 2",
    "Consumer/investment spending": "> 0",
    "Productivity growth": "> 3",
    "Debt-to-GDP": "< 60",
    "Foreign reserves": "> 0",
    "Real rates": "< -1",
    "Trade balance": "> 0",
    "Asset prices > traditional metrics": "> 1.2",
    "New buyers entering (market participation) (FINRA margin debt, FRED proxy)": "> 15",
    "Wealth gaps": "widening",
    "Credit spreads": "> 500",
    "Central bank printing (M2)": "> 10",
    "Currency devaluation": "< -10",
    "Fiscal deficits": "> 6",
    "Debt growth": "> 5",
    "Income growth": "> 0",
    "Debt service": "> 20",
    "Education investment": "> 5",
    "R&D patents": "> 0",
    "Competitiveness index / Competitiveness (WEF)": "improving",
    "GDP per capita growth": "> 3",
    "Trade share": "> 2",
    "Military spending": "> 4",
    "Internal conflicts": "rising",
    "Reserve currency usage dropping (IMF COFER USD share)": "falling",
    "Military losses (UCDP battle-related deaths — Global)": "rising",
    "Economic output share": "falling",
    "Corruption index": "worsening",
    "Working population": "falling",
    "Education (PISA scores — Math mean, OECD)": "> 500",
    "Innovation": "rising",
    "GDP share": "rising",
    "Trade dominance": "> 15",
    "Power index (CINC — USA)": "high",
    "Debt burden": "> 100",
}

UNITS: Dict[str, str] = {
    "Yield curve": "pct pts (10Y–2Y)",
    "Consumer confidence": "index",
    "Building permits": "thousands",
    "Unemployment claims": "thousands",
    "LEI (Conference Board Leading Economic Index)": "index",
    "GDP": "YoY %",
    "Capacity utilization": "%",
    "Inflation": "YoY %",
    "Retail sales": "YoY %",
    "Nonfarm payrolls": "thousands",
    "Wage growth": "YoY %",
    "P/E ratios": "ratio",
    "Credit growth": "YoY %",
    "Fed funds futures": "bps",
    "Short rates": "%",
    "Industrial production": "YoY %",
    "Consumer/investment spending": "YoY %",
    "Productivity growth": "YoY %",
    "Debt-to-GDP": "% of GDP",
    "Foreign reserves": "YoY %",
    "Real rates": "%",
    "Trade balance": "USD bn",
    "Asset prices > traditional metrics": "ratio",
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
    "R&D patents": "count",
    "Competitiveness index / Competitiveness (WEF)": "rank / index",
    "GDP per capita growth": "YoY %",
    "Trade share": "% of global",
    "Military spending": "% of GDP",
    "Internal conflicts": "index",
    "Reserve currency usage dropping (IMF COFER USD share)": "% of allocated",
    "Military losses (UCDP battle-related deaths — Global)": "deaths",
    "Economic output share": "% of global",
    "Corruption index": "index",
    "Working population": "% of pop (15–64)",
    "Education (PISA scores — Math mean, OECD)": "score",
    "Innovation": "index / share",
    "GDP share": "% of global",
    "Trade dominance": "% of global",
    "Power index (CINC — USA)": "index",
    "Debt burden": "% of GDP",
}

# ---------------------------------------------------------------------
# MAPPINGS FOR PROGRAMMATIC FETCH (CORE TAB)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# HELPERS (CORE FETCH + MIRRORS)
# ---------------------------------------------------------------------
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
            / pd.to_numeric(df2["w"], errors="coctrine")
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


# ---------------------------------------------------------------------
# SIMPLE THRESHOLD PARSER (ONLY FOR CORE TAB)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# LONG-TERM AND SHORT-TERM DASHBOARD DATA (STATIC NARRATIVE)
# ---------------------------------------------------------------------
LONG_TERM_ROWS: List[Dict[str, str]] = [
    {
        "Signal": "Total Debt/GDP (Private + Public + Foreign)",
        "Current value": "≈ 355% (BIS total credit to non-financial sector, latest available 2025)",
        "Red-flag threshold": "> 300–400% and still rising",
        "Status": "Red",
        "Direction": "Still rising (~+2% YoY)",
        "Why this matters in the long-term debt cycle": (
            "When total claims on the economy exceed output 3–4x, the system relies on defaults, "
            "inflation, or currency debasement to reset. Every historical long-term debt cycle "
            "(1930s US, 1980s Japan) peaked at similar levels."
        ),
    },
    {
        "Signal": "Productivity growth (real, US)",
        "Current value": "≈ 3.3% YoY in Q2 2025, but weak average since 2008",
        "Red-flag threshold": "< 1.5% for > 10 years",
        "Status": "Watch",
        "Direction": "Volatile; long-run trend stagnant",
        "Why this matters in the long-term debt cycle": (
            "Productivity is what services debt over decades. If productivity stagnates while debt "
            "keeps rising, the only way to keep the game going is more leverage and financial "
            "engineering – classic late-stage behaviour."
        ),
    },
    {
        "Signal": "Gold price (real, inflation-adjusted)",
        "Current value": "≈ $4,062/oz real (nominal ≈ $4,065; GuruFocus inflation-adjusted series)",
        "Red-flag threshold": "> 2x long-run real average (~$1,400)",
        "Status": "Red",
        "Direction": "Up strongly vs recent years",
        "Why this matters in the long-term debt cycle": (
            "Gold is the classic hedge against fiat and sovereign-debt risk. Sharp, sustained rises "
            "in real gold prices usually signal that investors are front-running currency debasement "
            "and a potential reset of the monetary order."
        ),
    },
    {
        "Signal": "Wage share of GDP (labor share proxy)",
        "Current value": "Low vs 1970s; stagnant around post-2000 lows",
        "Red-flag threshold": "Multi-decade downtrend; structurally low level",
        "Status": "Watch",
        "Direction": "Flat / low historically",
        "Why this matters in the long-term debt cycle": (
            "Falling wage share pushes households toward borrowing simply to maintain living "
            "standards. Combining high private debt with weak real incomes has preceded social "
            "stress and policy regime shifts in past cycles."
        ),
    },
    {
        "Signal": "Real 30-year US Treasury yield",
        "Current value": "≈ 1.8% real (≈4.7% nominal minus ≈2.9% CPI)",
        "Red-flag threshold": "Prolonged < 2% or deeply negative for years",
        "Status": "Watch",
        "Direction": "Low, slightly rising",
        "Why this matters in the long-term debt cycle": (
            "Persistently low or negative long real yields signal financial repression – using "
            "inflation to erode the real value of debt. That is a standard tool used in the "
            "endgame of long-term debt cycles."
        ),
    },
    {
        "Signal": "USD vs gold power (gold per $1,000)",
        "Current value": "≈ 0.24 oz per $1,000 at current prices",
        "Red-flag threshold": "Breaking below long-term uptrend, toward 0.10 oz per $1,000",
        "Status": "Red",
        "Direction": "Gold outperforming USD",
        "Why this matters in the long-term debt cycle": (
            "When a given amount of dollars buys less and less gold over time, it is a sign that "
            "confidence in the currency’s long-term value is eroding. That pattern has appeared "
            "around prior reserve-currency peaks."
        ),
    },
    {
        "Signal": "Geopolitical Risk Index (global, GPR)",
        "Current value": "Elevated vs historical average (~180 on recent readings)",
        "Red-flag threshold": "> 150 and rising with high debt",
        "Status": "Watch",
        "Direction": "Trending higher",
        "Why this matters in the long-term debt cycle": (
            "High debt combined with rising geopolitical tension is the environment in which resets "
            "and realignments tend to happen. Wars and major conflicts have repeatedly been used "
            "to liquidate unsustainable debt loads."
        ),
    },
    {
        "Signal": "Income inequality (US Gini coefficient)",
        "Current value": "≈ 0.41 (near modern highs)",
        "Red-flag threshold": "> 0.40 and rising",
        "Status": "Red",
        "Direction": "Higher than 1980s–1990s",
        "Why this matters in the long-term debt cycle": (
            "High inequality plus heavy debt loads is a classic recipe for internal conflict. "
            "Very unequal societies with large debt overhangs often end up in either forced "
            "restructurings or more chaotic resets."
        ),
    },
]

SHORT_TERM_ROWS: List[Dict[str, str]] = [
    {
        "Indicator": "Margin debt as % of GDP",
        "Current value": "≈ 3.3–3.5% (FINRA margin debt ÷ US nominal GDP)",
        "Red-flag threshold": "≥ 2.5% and especially ≥ 3.5%",
        "Status": "Red",
        "Direction": "Elevated vs long-run norms",
        "Why this matters in the short-term cycle": (
            "Margin debt is leveraged speculation. Peaks in 1929, 2000, 2007, and 2022 all had "
            "very high margin debt just before markets reversed, because forced deleveraging turns "
            "small dips into crashes."
        ),
    },
    {
        "Indicator": "Real short rate (Fed funds minus CPI)",
        "Current value": "≈ +1.0–1.5% (policy rate minus trailing CPI inflation)",
        "Red-flag threshold": (
            "Bubble build-up when < 0% for > 12 months; bubble popping risk when rises fast to "
            "≥ +1.5% and keeps climbing"
        ),
        "Status": "Green",
        "Direction": "Positive vs 2020–21 negatives",
        "Why this matters in the short-term cycle": (
            "Deeply negative real rates make borrowing essentially free, encouraging leverage and "
            "asset bubbles. When real rates swing positive and keep rising, the cheap fuel "
            "disappears and exposes weak balance sheets."
        ),
    },
    {
        "Indicator": "CBOE total put/call ratio",
        "Current value": "≈ 0.70–0.75 (calls slightly outnumber puts)",
        "Red-flag threshold": "< 0.70 for multiple days (extreme complacency)",
        "Status": "Watch",
        "Direction": "Near complacent levels",
        "Why this matters in the short-term cycle": (
            "Very low put/call readings mean almost nobody is buying downside protection. "
            "Historically, that kind of one-sided positioning has appeared just before major tops."
        ),
    },
    {
        "Indicator": "AAII bullish sentiment %",
        "Current value": "Low–moderate bulls (~30–35% in recent AAII surveys)",
        "Red-flag threshold": "> 60% for 2+ weeks",
        "Status": "Green",
        "Direction": "Not euphoric",
        "Why this matters in the short-term cycle": (
            "When more than 60% of survey respondents are bullish for several weeks, it usually "
            "means most retail investors are already in – exactly what has lined up with prior "
            "peaks in 1987, 2000, 2007, and 2022."
        ),
    },
    {
        "Indicator": "S&P 500 trailing P/E",
        "Current value": "≈ 29–30x (price ÷ trailing 12-month earnings)",
        "Red-flag threshold": "> 30x sustained while other risk lights are flashing",
        "Status": "Watch",
        "Direction": "At the high end of history",
        "Why this matters in the short-term cycle": (
            "Only a few eras have kept P/E above 30 for long – late 1920s, late 1990s, and "
            "2021–22. Each time, high valuations plus tighter policy led to big drawdowns."
        ),
    },
    {
        "Indicator": "Fed policy stance (QE vs QT)",
        "Current value": (
            "QT: balance sheet shrinking modestly (roughly tens of billions of dollars per month "
            "based on the Fed H.4.1 report – this describes direction, not a single numeric point)"
        ),
        "Red-flag threshold": "Turn from big QE to aggressive QT and/or rapid hikes",
        "Status": "Green (for now)",
        "Direction": "Liquidity slowly draining",
        "Why this matters in the short-term cycle": (
            "Every big bubble has ended when central banks removed liquidity – by stopping QE, "
            "raising rates aggressively, or both. The stance of policy tells you whether the tide "
            "is coming in or going out."
        ),
    },
    {
        "Indicator": "High-yield credit spreads",
        "Current value": "≈ 300–320 bps over Treasuries (Bank of America HY index ballpark)",
        "Red-flag threshold": "> 400 bps and widening quickly",
        "Status": "Green",
        "Direction": "Still tight",
        "Why this matters in the short-term cycle": (
            "Credit markets often flash warning before equities. When spreads are tight, investors "
            "are relaxed about default risk. A sharp widening usually marks the shift to risk-off."
        ),
    },
    {
        "Indicator": "Insider selling vs buybacks",
        "Current value": "Heavy insider sales; corporate buybacks softer vs peak levels",
        "Red-flag threshold": "Insider buying ratio < 10% (90%+ selling) plus slowing buybacks",
        "Status": "Red",
        "Direction": "Insiders de-risking their own exposure",
        "Why this matters in the short-term cycle": (
            "Executives see the business fundamentals first. When they sell heavily while "
            "companies also reduce buybacks, it is a strong sign insiders don’t believe current "
            "prices are sustainable."
        ),
    },
]

# ---------------------------------------------------------------------
# UI LAYOUT – TABS
# ---------------------------------------------------------------------
tab_core, tab_long, tab_short = st.tabs(
    [
        "📊 Core Econ Mirror indicators",
        "📉 Long-term debt super-cycle (40–70 yrs)",
        "⚡ Short-term bubble cycle (5–10 yrs)",
    ]
)

# ====================== CORE TAB ======================
with tab_core:
    st.title("Core Econ Mirror indicators")
    st.caption(
        "High-frequency macro indicators pulled from FRED, World Bank mirrors, and pinned CSVs "
        "for PISA, CINC, UCDP, IMF COFER, and S&P 500 P/E. These update automatically via your "
        "GitHub Actions mirror job plus live FRED/WB API lookups."
    )

    rows: List[Dict[str, object]] = []

    for ind in INDICATORS:
        unit = UNITS.get(ind, "")
        cur: float = float("nan")
        prev: float = float("nan")
        src: str = "—"

        # World Bank direct indicators
        if ind in WB_US:
            c, p, s, _ = wb_last_two(WB_US[ind], "USA")
            if not pd.isna(c):
                cur, prev, src = c, p, s

        # Shares (USA vs World)
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

        # Special mirrors / proxies
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
        signal_icon, _ = evaluate_signal(cur, threshold_txt)
        seed_badge = " <span class='badge seed'>Pinned seed</span>" if "Pinned seed" in src else ""

        rows.append(
            {
                "Indicator": ind,
                "Threshold (rough rule of thumb)": threshold_txt,
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
            .badge.seed {
                background: #8e44ad;
                color: #fff;
                padding: 2px 6px;
                border-radius: 6px;
                font-size: 11px;
                margin-left: 6px;
            }
            /* allow wrapping inside dataframe cells so you see full text */
            .stDataFrame [data-testid="cell-container"] {
                white-space: normal !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_out[
            [
                "Indicator",
                "Threshold (rough rule of thumb)",
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
        "Core data sources: FRED, World Bank, IMF COFER (mirror), OECD PISA (mirror), CINC (mirror), "
        "UCDP (mirror), MULTPL/Yale (mirror). Mirrors are refreshed by your nightly GitHub Actions "
        "job; live FRED/WB API calls fill any gaps."
    )

# ====================== LONG-TERM TAB ======================
with tab_long:
    st.title("Long-term Debt Super-Cycle Dashboard")

    st.write(
        "Structural 40–70 year signals: debt saturation, currency stress, inequality, and "
        "geopolitical risk. Snapshot as of late 2025. Each current value explains in plain "
        "language how it is derived so there is no ambiguity like 'QT' with no context."
    )

    df_long = pd.DataFrame(LONG_TERM_ROWS)
    st.table(df_long)

    n_red = sum(1 for row in LONG_TERM_ROWS if row["Status"].lower() == "red")
    n_watch = sum(1 for row in LONG_TERM_ROWS if row["Status"].lower() == "watch")

    st.markdown(
        f"""
**Current long-term score:** **{n_red} red** + **{n_watch} watch** out of 8 signals.

Your rule for the *final 6–24 months* before a true long-term reset:

- Wait for **6 or more of the 8 long-term signals to be DARK RED** at the same time; and  
- See at least **one 'point of no return' trigger**:

  1. Major central banks openly buying large amounts of gold.  
  2. A G20 country formally proposing or adopting a new currency / gold-backed system.  
  3. US 10-year yields spiking above ~7–8% while inflation is still high.

When that combo appears, the odds that the long-term super-cycle is in its endgame go from
'high' to 'almost certain'.
"""
    )

    st.markdown(
        """
**Manual cross-checks for these long-term inputs**

- BIS total debt statistics (credit to the non-financial sector)  
- FRED productivity, labor share, and long bond yield series  
- GuruFocus inflation-adjusted gold price indicator  
- PolicyUncertainty.com Geopolitical Risk Index (GPR)  
- World Bank Gini inequality estimates

The core raw series are pulled and mirrored; the narrative text on this tab is static and should
be reviewed and refreshed by you a few times a year.
"""
    )

# ====================== SHORT-TERM TAB ======================
with tab_short:
    st.title("Short-term Bubble Timing Dashboard")

    st.write(
        "5–10 year business / credit-cycle signals: leverage, sentiment, liquidity, and risk "
        "spreads. Again, each current value spells out how the number is constructed so the app "
        "is audit-friendly and not vague."
    )

    df_short = pd.DataFrame(SHORT_TERM_ROWS)
    st.table(df_short)

    n_red_short = sum(1 for row in SHORT_TERM_ROWS if row["Status"].lower().startswith("red"))
    n_watch_short = sum(1 for row in SHORT_TERM_ROWS if row["Status"].lower().startswith("watch"))

    st.markdown(
        f"""
**Current short-term score:** **{n_red_short} red** + **{n_watch_short} watch** out of 8 indicators.

Your **\"kill combo\" rule** to know the short-term bubble is in its final **1–8 weeks**:

- Wait for **6 or more of the 8 short-term indicators** to hit their kill levels *at the same time*,  
- While the S&P 500 is still within about **−8% of its all-time high**.

Whenever that combo has appeared (1929, 2000, 2007, 2022), markets have gone on to lose at least
~30% (average drawdown around 50%) in the following months.
"""
    )

    st.markdown(
        """
**Manual cross-checks for the short-term dashboard**

- Margin debt: FINRA statistics or GuruFocus margin-debt-to-GDP series  
- Real policy rate: FRED Fed Funds and CPI series  
- Put/call ratios: CBOE daily statistics  
- AAII sentiment: aaii.com weekly survey  
- Valuations: S&P 500 P/E from MULTPL / Yardeni / GuruFocus  
- Fed stance: Federal Reserve H.4.1 balance sheet and FOMC statements  
- Credit spreads: FRED high-yield spread series  
- Insider activity and buybacks: OpenInsider + S&P buyback reports

These descriptions are static text. The live, self-updating engine in your app is the core
FRED/World Bank + mirrors stack; this tab is your human rulebook layered on top.
"""
    )
