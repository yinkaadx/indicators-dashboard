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

# FRED client (reads FRED_API_KEY from Streamlit secrets)
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ---------------------------------------------------------------------
# Indicators / thresholds / units
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

# ---------------------------------------------------------------------
# Mappings for programmatic fetch
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
# Helpers
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
    return (
        pd.to_numeric(s.tail(n).values, errors="coerce")
        .astype(float)
        .tolist()
    )


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
        hist = (
            pd.to_numeric(t["val"], errors="coerce")
            .tail(24)
            .astype(float)
            .tolist()
        )
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
    hist = (
        pd.to_numeric(df[value_col], errors="coerce")
        .tail(24)
        .astype(float)
        .tolist()
    )
    return cur, prev, src, hist


def cofer_usd_share_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "imf_cofer_usd_share.csv")
    return mirror_latest_csv(path, "usd_share", "date", numeric_time=False)


def sp500_pe_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "pe_sp500.csv")
    return mirror_latest_csv(path, "pe", "date", numeric_time=False)


# ---------------------------------------------------------------------
# Threshold helpers
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
# Static long-term & short-term cycle dashboards
# (snapshot values, roughly as of late 2025)
# ---------------------------------------------------------------------
def build_long_term_df() -> pd.DataFrame:
    rows: List[Dict[str, str]] = [
        {
            "Signal": "Total Debt/GDP (Private + Public + Foreign)",
            "Current value": "≈ 355%",
            "Red-flag threshold": "> 300–400%",
            "Status": "🔴 Red",
            "Direction": "Still rising (~+2% YoY)",
            "Why this matters in the long-term debt cycle": (
                "When total claims on the economy exceed output by ~3–4x, "
                "systems hit debt saturation. Historic peaks (1929 US, "
                "Japan 1980s) preceded long deleveraging or restructuring."
            ),
        },
        {
            "Signal": "Productivity growth (real, US)",
            "Current value": "≈ 3.3% Q2 2025, but weak trend since 2008",
            "Red-flag threshold": "< 1.5% for > 10 years",
            "Status": "🟡 Watch",
            "Direction": "Volatile; long-run trend stagnant",
            "Why this matters in the long-term debt cycle": (
                "Productivity is the engine to service debt. Stagnant productivity "
                "with rising debt usually leads to money printing and currency "
                "debasement instead of real growth."
            ),
        },
        {
            "Signal": "Gold price (real, inflation-adjusted)",
            "Current value": "≈ $4,062/oz real (nominal ~$4,065)",
            "Red-flag threshold": "> 2× long-run real average (~$1,400)",
            "Status": "🔴 Red",
            "Direction": "Up strongly vs recent years",
            "Why this matters in the long-term debt cycle": (
                "Gold is the classic hedge against fiat and sovereign-debt stress. "
                "Sustained real breakouts in gold have aligned with major "
                "re-pricings of monetary regimes."
            ),
        },
        {
            "Signal": "Wage share of GDP (labor share proxy)",
            "Current value": "Low vs 1970s; stagnant",
            "Red-flag threshold": "Multi-decade downtrend; structurally low level",
            "Status": "🟡 Watch",
            "Direction": "Flat/low historically",
            "Why this matters in the long-term debt cycle": (
                "Falling wage share pushes households toward borrowing to maintain "
                "consumption, inflating the long debt cycle and feeding political "
                "tension and instability."
            ),
        },
        {
            "Signal": "Real 30-year Treasury yield",
            "Current value": "≈ 1.8% (4.7% nominal – ~2.9% CPI)",
            "Red-flag threshold": "Prolonged < 2% (or negative)",
            "Status": "🟡 Watch",
            "Direction": "Low, slightly rising",
            "Why this matters in the long-term debt cycle": (
                "Persistently low or negative long real yields signal financial "
                "repression and fiscal dominance — typical in late-cycle regimes "
                "struggling with high debt stocks."
            ),
        },
        {
            "Signal": "USD vs gold power (gold per $1,000)",
            "Current value": "≈ 0.24 oz per $1,000",
            "Red-flag threshold": "Breaking below long-term uptrend",
            "Status": "🔴 Red",
            "Direction": "Gold outperforming USD",
            "Why this matters in the long-term debt cycle": (
                "Weaker USD vs gold reflects erosion in monetary credibility. "
                "Patterns like this appeared in prior reserve-currency transitions "
                "and during Bretton Woods breakdown."
            ),
        },
        {
            "Signal": "Geopolitical Risk Index (global)",
            "Current value": "Elevated vs historical average",
            "Red-flag threshold": "> 150 and rising with high debt",
            "Status": "🟡 Watch",
            "Direction": "Trending higher",
            "Why this matters in the long-term debt cycle": (
                "High debt combined with rising geopolitical risk increases the "
                "chance of conflict-driven resets, restructuring, or order changes."
            ),
        },
        {
            "Signal": "Income inequality (US Gini coefficient)",
            "Current value": "≈ 0.41 (near modern highs)",
            "Red-flag threshold": "> 0.40 and rising",
            "Status": "🔴 Red",
            "Direction": "Higher than 1980s–1990s",
            "Why this matters in the long-term debt cycle": (
                "High inequality plus heavy debt loads is a classic setup for "
                "populism, policy shocks, and regime change — seen in the 1930s, "
                "1970s, and other major turning points."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_short_term_df() -> pd.DataFrame:
    rows: List[Dict[str, str]] = [
        {
            "Indicator": "Margin debt as % of GDP",
            "Current value": "≈ 3.3–3.5%",
            "Red-flag threshold": "> 2.5%",
            "Status": "🔴 Red",
            "Direction": "Elevated vs long-run norms",
            "Why this matters in the short-term cycle": (
                "Margin debt is leveraged speculation. Peaks in 1929, 2000, 2007, "
                "and 2021–22 coincided with major tops. When it rolls over, forced "
                "selling amplifies downturns."
            ),
        },
        {
            "Indicator": "Real short rate (Fed funds – CPI)",
            "Current value": "≈ +1.0–1.5%",
            "Red-flag threshold": "Bubble build-up: < 0% for > 12 months",
            "Status": "🟢 Green",
            "Direction": "Positive vs 2020–21 negatives",
            "Why this matters in the short-term cycle": (
                "Deep negative real rates make borrowing essentially free, fueling "
                "asset bubbles. The flip back to positive removes that fuel."
            ),
        },
        {
            "Indicator": "CBOE total put/call ratio",
            "Current value": "≈ 0.7–0.75",
            "Red-flag threshold": "< 0.70 (extreme complacency)",
            "Status": "🟡 Watch",
            "Direction": "Near complacent levels",
            "Why this matters in the short-term cycle": (
                "Very low put/call readings mean almost nobody is hedging. "
                "Readings under ~0.7 have often aligned with late-stage "
                "melt-ups and subsequent corrections."
            ),
        },
        {
            "Indicator": "AAII bullish sentiment %",
            "Current value": "Low–moderate bulls (~30–35%)",
            "Red-flag threshold": "> 60% for multiple weeks",
            "Status": "🟢 Green",
            "Direction": "Not euphoric",
            "Why this matters in the short-term cycle": (
                "When > 60% of retail survey respondents are bullish for weeks, "
                "there are few incremental buyers left. Historically this has "
                "marked exhaustion zones."
            ),
        },
        {
            "Indicator": "S&P 500 trailing P/E",
            "Current value": "≈ 29–30×",
            "Red-flag threshold": "> 30× sustained",
            "Status": "🟡 Watch",
            "Direction": "At the high end of history",
            "Why this matters in the short-term cycle": (
                "Only a few eras sustained P/E > 30 (late 1920s, late 1990s, "
                "2020–21). Each was followed by significant drawdowns as valuation "
                "mean reversion kicked in."
            ),
        },
        {
            "Indicator": "Fed policy stance (QE vs QT)",
            "Current value": "QT / modest balance-sheet shrink",
            "Red-flag threshold": "Turn from QE to QT / rapid hikes",
            "Status": "🟢 Green (for now)",
            "Direction": "Liquidity slowly draining",
            "Why this matters in the short-term cycle": (
                "Every big bubble ended when central banks removed liquidity "
                "or hiked aggressively: 1929, 2000, 2007, 2022. Liquidity is the "
                "core driver of risk appetite."
            ),
        },
        {
            "Indicator": "High-yield credit spreads",
            "Current value": "≈ 300–320 bps",
            "Red-flag threshold": "> 400 bps and widening",
            "Status": "🟢 Green",
            "Direction": "Still tight",
            "Why this matters in the short-term cycle": (
                "Credit markets often flash warning before equities. Spread "
                "widening signals rising default risk and tightening conditions."
            ),
        },
        {
            "Indicator": "Insider selling vs buybacks",
            "Current value": "Heavy insider selling; buybacks softer",
            "Red-flag threshold": "Rising insider sales + slowing buybacks",
            "Status": "🔴 Red",
            "Direction": "Insiders de-risking into strength",
            "Why this matters in the short-term cycle": (
                "Executives see fundamentals first. When they sell heavily while "
                "corporate buybacks slow, it has preceded multiple major tops."
            ),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("Econ Mirror Dashboard")
st.caption(
    "Core macro indicators plus long-term debt cycle and short-term bubble dashboards. "
    "Data pulled from FRED, World Bank mirrors, and pinned CSVs (PISA, CINC, UCDP, IMF COFER, S&P 500 P/E)."
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
                to_float(series.iloc[-2]["share"])
                if len(series) > 1
                else float("nan")
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
                to_float(series.iloc[-2]["share"])
                if len(series) > 1
                else float("nan")
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

    # Build table row
    threshold_txt = THRESHOLDS.get(ind, "—")
    signal_icon, signal_cls = evaluate_signal(cur, threshold_txt)
    seed_badge = " <span class='badge seed'>Pinned seed</span>" if "Pinned seed" in src else ""
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

# Final main table
df_out = pd.DataFrame(rows)

# Styling
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

# Tabs: core indicators + two cycle dashboards
tab_main, tab_long, tab_short = st.tabs(
    [
        "📊 Core Econ Mirror indicators",
        "🧭 Long-term debt super-cycle (40–70 yrs)",
        "📈 Short-term bubble cycle (5–10 yrs)",
    ]
)

with tab_main:
    st.subheader("Core Macro & System Indicators")
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
        "Sources: FRED, World Bank mirrors, IMF COFER (mirror), OECD PISA (mirror), "
        "CINC (mirror), UCDP (mirror), MULTPL/Yale (mirror)."
    )

with tab_long:
    st.subheader("Long-term Debt Super-Cycle Dashboard")
    st.markdown(
        "Structural 40–70 year signals: debt saturation, currency stress, "
        "inequality, and geopolitical risk. Snapshot as of late 2025."
    )
    long_df = build_long_term_df()
    st.dataframe(long_df, use_container_width=True, hide_index=True)

with tab_short:
    st.subheader("Short-term Bubble Timing Dashboard")
    st.markdown(
        "5–10 year business/credit-cycle signals: leverage, sentiment, "
        "liquidity, and risk spreads. Snapshot as of late 2025."
    )
    short_df = build_short_term_df()
    st.dataframe(short_df, use_container_width=True, hide_index=True)
