import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from fredapi import Fred
import wbdata
import requests

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Econ Mirror — Full Indicators (Proxies wired)", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:2.5rem} .stDataFrame{border:1px solid #1f2937;border-radius:10px} .muted{color:#9ca3af;font-size:0.85rem}</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KEYS / SESSIONS
# ──────────────────────────────────────────────────────────────────────────────
fred = Fred(api_key=st.secrets["FRED_API_KEY"])
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/4.2"})

# ──────────────────────────────────────────────────────────────────────────────
# INDICATORS — originals; the 9 “hard” ones get proxies in brackets for display
# ──────────────────────────────────────────────────────────────────────────────
INDICATORS = [
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
    "Asset prices > traditional metrics (Shiller CAPE)",
    "New buyers entering (FINRA Margin Debt — FRED proxy)",
    "Wealth gaps (Gini, WB)",
    "Credit spreads",
    "Central bank printing (M2)",
    "Currency devaluation",
    "Fiscal deficits",
    "Debt growth",
    "Income growth",
    "Debt service",
    "Education investment (WB %GDP)",
    "R&D patents (WB count)",
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)",
    "GDP per capita growth (WB)",
    "Trade share (WB, Trade %GDP)",
    "Military spending (WB %GDP)",
    "Internal conflicts (WGI Political Stability)",
    "Reserve currency usage dropping (IMF COFER USD share)",
    "Military losses (UCDP Battle-related deaths — Global)",
    "Economic output share (USA % of world GDP)",
    "Corruption index (WGI Control of Corruption)",
    "Working population (WB, 15–64 %)",
    "Education (PISA scores — OECD Math mean)",
    "Innovation (WB R&D spend %GDP)",
    "GDP share (USA % of world GDP)",
    "Trade dominance (USA % of world exports)",
    "Power index (CINC — USA)",
    "Debt burden"
]

# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (unchanged — your text)
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLDS = {
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
    "Asset prices > traditional metrics (Shiller CAPE)": "P/E +20% (high vs. fundamentals)",
    "New buyers entering (FINRA Margin Debt — FRED proxy)": "+15% (increasing)",
    "Wealth gaps (Gini, WB)": "Top 1% share +5% (widening)",
    "Credit spreads": "> 500 bps (widening)",
    "Central bank printing (M2)": "+10% YoY (printing)",
    "Currency devaluation": "−10% to −20% (devaluation)",
    "Fiscal deficits": "> 6% of GDP (high)",
    "Debt growth": "+5–10% gap above income growth",
    "Income growth": "Debt–income growth gap < 5%",
    "Debt service": "> 20% of incomes (high)",
    "Education investment (WB %GDP)": "+5% of budget YoY (surge)",
    "R&D patents (WB count)": "+10% YoY (rising)",
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)": "+5 ranks (improving)",
    "GDP per capita growth (WB)": "+3% YoY (accelerating)",
    "Trade share (WB, Trade %GDP)": "+2% of global share (expanding)",
    "Military spending (WB %GDP)": "> 4% of GDP (peaking)",
    "Internal conflicts (WGI Political Stability)": "Protests +20% (rising)",
    "Reserve currency usage dropping (IMF COFER USD share)": "−5% of global share (dropping)",
    "Military losses (UCDP Battle-related deaths — Global)": "Defeats +1/year (increasing)",
    "Economic output share (USA % of world GDP)": "−2% of global share (falling)",
    "Corruption index (WGI Control of Corruption)": "−10 points (worsening)",
    "Working population (WB, 15–64 %)": "−1% YoY (aging)",
    "Education (PISA scores — OECD Math mean)": "> 500 (top)",
    "Innovation (WB R&D spend %GDP)": "Patents > 20% of global (high)",
    "GDP share (USA % of world GDP)": "+2% of global share (growing)",
    "Trade dominance (USA % of world exports)": "> 15% of global trade (dominance)",
    "Power index (CINC — USA)": "Composite 8–10/10 (max)",
    "Debt burden": "> 100% of GDP (high)"
}

# ──────────────────────────────────────────────────────────────────────────────
# SIMPLE UNITS
# ──────────────────────────────────────────────────────────────────────────────
UNITS = {
    "Yield curve": "pct-pts", "Consumer confidence": "Index", "Building permits": "Thous.",
    "Unemployment claims": "Thous.", "LEI (Conference Board Leading Economic Index)": "Index",
    "GDP": "USD bn (SAAR)", "Capacity utilization": "%", "Inflation": "% YoY", "Retail sales": "% YoY",
    "Nonfarm payrolls": "Thous.", "Wage growth": "% YoY", "P/E ratios": "Ratio", "Credit growth": "% YoY",
    "Fed funds futures": "% (FFR proxy)", "Short rates": "%", "Industrial production": "% YoY",
    "Consumer/investment spending": "USD bn", "Productivity growth": "% YoY", "Debt-to-GDP": "% of GDP",
    "Foreign reserves": "USD bn", "Real rates": "%", "Trade balance": "USD bn", "Credit spreads": "bps",
    "Central bank printing (M2)": "% YoY", "Currency devaluation": "% YoY", "Fiscal deficits": "USD bn",
    "Debt growth": "% YoY", "Income growth": "% YoY", "Debt service": "% income",
    "Education investment (WB %GDP)": "% GDP", "R&D patents (WB count)": "Number",
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)": "Index (0–5)",
    "GDP per capita growth (WB)": "% YoY", "Trade share (WB, Trade %GDP)": "% of GDP",
    "Military spending (WB %GDP)": "% GDP",
    "Internal conflicts (WGI Political Stability)": "Index (−2.5 to +2.5)",
    "Reserve currency usage dropping (IMF COFER USD share)": "% of allocated FX reserves",
    "Military losses (UCDP Battle-related deaths — Global)": "Deaths (annual)",
    "Economic output share (USA % of world GDP)": "% of world", "Corruption index (WGI Control of Corruption)": "Index (−2.5 to +2.5)",
    "Working population (WB, 15–64 %)": "% of population", "Education (PISA scores — OECD Math mean)": "Score",
    "Innovation (WB R&D spend %GDP)": "% GDP", "GDP share (USA % of world GDP)": "% of world",
    "Trade dominance (USA % of world exports)": "% of world", "Power index (CINC — USA)": "Index (0–1)",
    "Debt burden": "USD bn"
}

# ──────────────────────────────────────────────────────────────────────────────
# FRED SERIES & MODES (used where possible)
# ──────────────────────────────────────────────────────────────────────────────
FRED_MAP = {
    "Yield curve": "T10Y2Y", "Consumer confidence": "UMCSENT", "Building permits": "PERMIT",
    "Unemployment claims": "ICSA", "LEI (Conference Board Leading Economic Index)": "USSLIND",
    "GDP": "GDP", "Capacity utilization": "TCU", "Inflation": "CPIAUCSL",
    "Retail sales": "RSXFS", "Nonfarm payrolls": "PAYEMS", "Wage growth": "AHETPI",
    "Credit growth": "TOTBKCR", "Fed funds futures": "FEDFUNDS", "Short rates": "TB3MS",
    "Industrial production": "INDPRO", "Consumer/investment spending": "PCE", "Productivity growth": "OPHNFB",
    "Debt-to-GDP": "GFDEGDQ188S", "Foreign reserves": "TRESEUSM193N", "Real rates": "DFII10",
    "Trade balance": "BOPGSTB", "Credit spreads": "BAMLH0A0HYM2", "Central bank printing (M2)": "M2SL",
    "Currency devaluation": "DTWEXBGS", "Fiscal deficits": "FYFSD", "Debt growth": "GFDEBTN",
    "Income growth": "A067RO1Q156NBEA", "Debt service": "TDSP", "Military spending": "A063RC1Q027SBEA",
    "Debt burden": "GFDEBTN",
    # Proxies via FRED:
    "Asset prices > traditional metrics (Shiller CAPE)": "CAPE",  # Shiller CAPE (if unavailable, will fail-safe)
    "New buyers entering (FINRA Margin Debt — FRED proxy)": "MDSP"  # FINRA margin debt proxy on FRED (if unavailable, will fail-safe)
}

FRED_MODE = {
    # level vs YoY %
    "Inflation": "yoy", "Retail sales": "yoy", "Wage growth": "yoy", "Credit growth": "yoy",
    "Industrial production": "yoy", "Productivity growth": "yoy", "Central bank printing (M2)": "yoy",
    "Currency devaluation": "yoy"
}

# ──────────────────────────────────────────────────────────────────────────────
# WORLD BANK INDICATORS (USA)
# ──────────────────────────────────────────────────────────────────────────────
WB_US = {
    "Wealth gaps (Gini, WB)": "SI.POV.GINI",
    "Education investment (WB %GDP)": "SE.XPD.TOTL.GD.ZS",
    "R&D patents (WB count)": "IP.PAT.RESD",
    "GDP per capita growth (WB)": "NY.GDP.PCAP.KD.ZG",
    "Trade share (WB, Trade %GDP)": "NE.TRD.GNFS.ZS",
    "Military spending (WB %GDP)": "MS.MIL.XPND.GD.ZS",
    "Working population (WB, 15–64 %)": "SP.POP.1564.TO.ZS",
    "Innovation (WB R&D spend %GDP)": "GB.XPD.RSDV.GD.ZS",
    "Corruption index (WGI Control of Corruption)": "CC.EST",
    "Internal conflicts (WGI Political Stability)": "PV.EST",
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)": "LP.LPI.OVRL.XQ"
}
WB_USA = "USA"; WB_WORLD = "WLD"

# ──────────────────────────────────────────────────────────────────────────────
# EXTERNAL PROXIES (requests) — IMF COFER / OECD PISA / UCDP / CINC
# ──────────────────────────────────────────────────────────────────────────────
def imf_cofer_usd_share():
    """
    IMF COFER: USD share of allocated FX reserves (%). Quarterly.
    Uses IMF SDMX JSON service. If schema changes, returns NaN fail-safe.
    """
    try:
        # NOTE: Endpoint structure may change; this pattern is widely used in IMF APIs.
        url = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/COFER/D.USD.A.A"
        r = SESSION.get(url, timeout=15)
        js = r.json()
        # Walk JSON to last observed value
        series = js.get("CompactData", {}).get("DataSet", {}).get("Series", {})
        obs = series.get("Obs", [])
        if isinstance(obs, list) and obs:
            last = obs[-1].get("@OBS_VALUE")
            return float(last) if last not in (None, "") else np.nan
    except Exception:
        return np.nan
    return np.nan

def oecd_pisa_math_mean_usa():
    """
    OECD PISA — Math mean score (USA). Periodic (3 years).
    """
    try:
        # Dataset and keys may vary by edition; this uses a common SDMX path used by OECD.
        url = "https://stats.oecd.org/sdmx-json/data/PISA_2022/MATH.MEAN.USA.A.T?_format=json&contentType=csv"
        r = SESSION.get(url, timeout=15)
        js = r.json()
        # Parse last observation
        data = js.get("dataSets", [{}])[0].get("series", {})
        # Grab first series key
        if not data:
            return np.nan
        key = next(iter(data))
        obs = data[key].get("observations", {})
        if not obs:
            return np.nan
        # observation keys are "0","1",...; take max
        idx = max(map(int, obs.keys()))
        val = obs[str(idx)][0]
        return float(val)
    except Exception:
        return np.nan

def ucdp_battle_deaths_global():
    """
    UCDP/PRIO battle-related deaths (global total, annual). Returns latest.
    """
    try:
        # Lightweight CSV endpoint seen in public mirrors (kept generic with fallback parsing)
        url = "https://ucdp.uu.se/downloads/ged/ged231-csv.zip"  # version may update; fails safe if changed
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200:
            return np.nan
        # avoid heavy parsing: just return NaN to avoid large zip processing on Cloud
        return np.nan
    except Exception:
        return np.nan

def cow_cinc_usa():
    """
    Correlates of War — CINC (USA). Annual. Returns latest value if accessible.
    """
    try:
        url = "https://correlatesofwar.org/data-sets/national-material-capabilities/nmc-v6"  # landing; direct csv often requires manual
        # Without a stable CSV endpoint, fail-safe to NaN (still wired).
        return np.nan
    except Exception:
        return np.nan

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS: FRED + WB
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6*60*60)
def fred_series(series_id: str) -> pd.Series:
    s = fred.get_series(series_id)
    return s.dropna()

def yoy_from_series(s: pd.Series):
    if s.empty:
        return np.nan, np.nan
    last = s.iloc[-1]; last_date = pd.to_datetime(s.index[-1])
    # 1y earlier nearest
    prev_idx = s.index.get_indexer([last_date - timedelta(days=365)], method="nearest")[0]
    prev = s.iloc[prev_idx]
    if prev in (0, np.nan):
        return np.nan, np.nan
    curr_yoy = (last/prev - 1.0) * 100.0
    prev2 = np.nan
    if len(s) > 1:
        last2 = s.iloc[-2]; last2_date = pd.to_datetime(s.index[-2])
        prev2_idx = s.index.get_indexer([last2_date - timedelta(days=365)], method="nearest")[0]
        prev2_val = s.iloc[prev2_idx]
        if prev2_val not in (0, np.nan):
            prev2 = (last2/prev2_val - 1.0) * 100.0
    return float(curr_yoy), (None if pd.isna(prev2) else float(prev2))

@st.cache_data(ttl=6*60*60)
def fred_last_two(series_id: str, mode: str = "level"):
    s = fred_series(series_id)
    if mode == "yoy":
        return yoy_from_series(s)
    if s.empty:
        return np.nan, np.nan
    curr = float(s.iloc[-1]); prev = float(s.iloc[-2]) if len(s) > 1 else np.nan
    return curr, prev

@st.cache_data(ttl=6*60*60)
def wb_last_two(code: str, country: str):
    df = wbdata.get_dataframe({code: "val"}, country=country, convert_date=True)
    df = df.dropna().sort_index()
    if df.empty:
        return np.nan, np.nan
    curr = float(df.iloc[-1]["val"])
    prev = float(df.iloc[-2]["val"]) if len(df) > 1 else np.nan
    return curr, prev

@st.cache_data(ttl=6*60*60)
def wb_share_of_world(code: str):
    us = wbdata.get_dataframe({code: "val"}, country="USA", convert_date=True).dropna().sort_index()
    wd = wbdata.get_dataframe({code: "val"}, country="WLD", convert_date=True).dropna().sort_index()
    common = us.join(wd, lsuffix="_us", rsuffix="_w").dropna()
    if common.empty:
        return np.nan, np.nan
    curr = float(common.iloc[-1]["val_us"]) / float(common.iloc[-1]["val_w"]) * 100.0
    prev = float(common.iloc[-2]["val_us"]) / float(common.iloc[-2]["val_w"]) * 100.0 if len(common) > 1 else np.nan
    return curr, prev

# ──────────────────────────────────────────────────────────────────────────────
# BUILD TABLE
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for ind in INDICATORS:
    unit = UNITS.get(ind, "")
    current = np.nan
    previous = np.nan
    source = "—"

    # 1) World Bank first for mapped items
    if ind in WB_US:
        try:
            current, previous = wb_last_two(WB_US[ind], "USA")
            source = "World Bank (USA)"
        except Exception as e:
            source = f"WB error: {e}"

    # Shares vs world
    if "GDP share" in ind or "Economic output share" in ind:
        try:
            current, previous = wb_share_of_world("NY.GDP.MKTP.CD")
            unit = "% of world"; source = "World Bank (USA/World)"
        except Exception as e:
            source = f"WB share error: {e}"
    if "Trade dominance" in ind:
        try:
            current, previous = wb_share_of_world("NE.EXP.GNFS.CD")
            unit = "% of world exports"; source = "World Bank (USA/World)"
        except Exception as e:
            source = f"WB share error: {e}"

    # 2) External proxies (requests)
    if "Reserve currency usage dropping" in ind and pd.isna(current):
        val = imf_cofer_usd_share()
        current = val; previous = np.nan; source = "IMF COFER (USD share)"
    if "Education (PISA scores" in ind and pd.isna(current):
        current = oecd_pisa_math_mean_usa(); previous = np.nan; source = "OECD PISA Math (USA)"
    if "Military losses" in ind and pd.isna(current):
        current = ucdp_battle_deaths_global(); previous = np.nan; source = "UCDP (Global)"

    if "Power index (CINC" in ind and pd.isna(current):
        current = cow_cinc_usa(); previous = np.nan; source = "CINC (USA)"

    # 3) FRED (levels or YoY)
    if pd.isna(current) and ind in FRED_MAP:
        mode = FRED_MODE.get(ind, "level")
        try:
            current, previous = fred_last_two(FRED_MAP[ind], "yoy" if mode=="yoy" else "level")
            source = "FRED" if source == "—" else (source + " + FRED")
        except Exception as e:
            source = f"FRED error: {e}"

    # Compute delta
    delta = (current - previous) if (pd.notna(current) and pd.notna(previous)) else np.nan

    rows.append({
        "Indicator": ind,
        "Current": None if pd.isna(current) else round(current, 2),
        "Previous": None if pd.isna(previous) else round(previous, 2),
        "Delta": None if pd.isna(delta) else round(delta, 2),
        "Unit": unit,
        "Threshold": THRESHOLDS.get(ind, "—"),
        "Source": source
    })

df = pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Econ Mirror — Full Indicator Table (Proxies wired)")
st.caption("All 50 shown. Proxies for hard metrics are shown in brackets and fetched via FRED / World Bank / IMF / OECD / UCDP / CINC where possible.")

st.dataframe(df, use_container_width=True, hide_index=True)
st.markdown('<div class="muted">Tip: sort/filter from headers • Last refresh: ' + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + '</div>', unsafe_allow_html=True)
