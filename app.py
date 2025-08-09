import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from fredapi import Fred
import wbdata

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Econ Mirror — Full Indicators", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:2.5rem} .stDataFrame{border:1px solid #1f2937;border-radius:10px} .muted{color:#9ca3af;font-size:0.85rem}</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SECRETS / KEYS
# ──────────────────────────────────────────────────────────────────────────────
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ──────────────────────────────────────────────────────────────────────────────
# INDICATORS (your exact list, in order)
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
    "Asset prices > traditional metrics",
    "New buyers entering (market participation)",
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
    "Reserve currency usage dropping",
    "Military losses",
    "Economic output share",
    "Corruption index",
    "Working population",
    "Education (PISA scores)",
    "Innovation",
    "GDP share",
    "Trade dominance",
    "Power index",
    "Debt burden"
]

# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (your exact text)
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
    "Asset prices > traditional metrics": "P/E +20% (high vs. fundamentals)",
    "New buyers entering (market participation)": "+15% (increasing)",
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
    "Reserve currency usage dropping": "−5% of global share (dropping)",
    "Military losses": "Defeats +1/year (increasing)",
    "Economic output share": "−2% of global share (falling)",
    "Corruption index": "−10 points (worsening)",
    "Working population": "−1% YoY (aging)",
    "Education (PISA scores)": "> 500 (top)",
    "Innovation": "Patents > 20% of global (high)",
    "GDP share": "+2% of global share (growing)",
    "Trade dominance": "> 15% of global trade (dominance)",
    "Power index": "Composite 8–10/10 (max)",
    "Debt burden": "> 100% of GDP (high)"
}

# ──────────────────────────────────────────────────────────────────────────────
# UNITS (kept simple)
# ──────────────────────────────────────────────────────────────────────────────
UNITS = {
    "Yield curve": "pct-pts",
    "Consumer confidence": "Index",
    "Building permits": "Thous.",
    "Unemployment claims": "Thous.",
    "LEI (Conference Board Leading Economic Index)": "Index",
    "GDP": "USD bn (SAAR)",
    "Capacity utilization": "%",
    "Inflation": "% YoY",
    "Retail sales": "% YoY",
    "Nonfarm payrolls": "Thous.",
    "Wage growth": "% YoY",
    "P/E ratios": "Ratio",
    "Credit growth": "% YoY",
    "Fed funds futures": "% (proxy: FFR)",
    "Short rates": "%",
    "Industrial production": "% YoY",
    "Consumer/investment spending": "USD bn",
    "Productivity growth": "% YoY",
    "Debt-to-GDP": "% of GDP",
    "Foreign reserves": "USD bn",
    "Real rates": "%",
    "Trade balance": "USD bn",
    "Credit spreads": "bps",
    "Central bank printing (M2)": "% YoY",
    "Currency devaluation": "% YoY (USD index)",
    "Fiscal deficits": "USD bn",
    "Debt growth": "% YoY",
    "Income growth": "% YoY",
    "Debt service": "% income",
    "Education investment": "% GDP",
    "R&D patents": "Number",
    "GDP per capita growth": "% YoY",
    "Trade share": "% of GDP",
    "Military spending": "% GDP",
    "Working population": "% of total",
    "Innovation": "% GDP (R&D spend)",
    "GDP share": "% of world",
    "Trade dominance": "% of world exports",
    "Debt burden": "USD bn"
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA SOURCES MAPS
# ──────────────────────────────────────────────────────────────────────────────
# FRED series (levels OR used to compute YoY)
FRED_MAP = {
    "Yield curve": "T10Y2Y",                     # pct-pts
    "Consumer confidence": "UMCSENT",            # index
    "Building permits": "PERMIT",                # thousands (level)
    "Unemployment claims": "ICSA",               # thousands (level, weekly)
    "LEI (Conference Board Leading Economic Index)": "USSLIND",
    "GDP": "GDP",                                # USD bn SAAR (level)
    "Capacity utilization": "TCU",               # %
    "Inflation": "CPIAUCSL",                     # will compute YoY %
    "Retail sales": "RSXFS",                     # YoY %
    "Nonfarm payrolls": "PAYEMS",                # thousands (level)
    "Wage growth": "AHETPI",                     # YoY %
    "Credit growth": "TOTBKCR",                  # YoY %
    "Fed funds futures": "FEDFUNDS",             # proxy: effective rate level
    "Short rates": "TB3MS",                      # %
    "Industrial production": "INDPRO",           # YoY %
    "Consumer/investment spending": "PCE",       # USD bn (level)
    "Productivity growth": "OPHNFB",             # YoY %
    "Debt-to-GDP": "GFDEGDQ188S",                # %
    "Foreign reserves": "TRESEUSM193N",          # USD (approx)
    "Real rates": "DFII10",                      # 10Y TIPS real yield %
    "Trade balance": "BOPGSTB",                  # USD bn
    "Credit spreads": "BAMLH0A0HYM2",            # bps
    "Central bank printing (M2)": "M2SL",        # YoY %
    "Currency devaluation": "DTWEXBGS",          # USD Broad Index -> YoY % (neg = deval)
    "Fiscal deficits": "FYFSD",                  # USD bn
    "Debt growth": "GFDEBTN",                    # YoY %
    "Income growth": "A067RO1Q156NBEA",          # YoY %
    "Debt service": "TDSP",                      # % income
    "Military spending": "A063RC1Q027SBEA",      # (we'll prefer WB %GDP below if available)
    "Debt burden": "GFDEBTN"                     # USD bn level
}

# World Bank codes (USA) for items not great on FRED
WB_US = {
    "Education investment": "SE.XPD.TOTL.GD.ZS",   # % GDP
    "R&D patents": "IP.PAT.RESD",                  # number
    "GDP per capita growth": "NY.GDP.PCAP.KD.ZG",  # % YoY
    "Trade share": "NE.TRD.GNFS.ZS",               # Trade (% of GDP)
    "Military spending": "MS.MIL.XPND.GD.ZS",      # % GDP (prefer this)
    "Working population": "SP.POP.1564.TO.ZS",     # % of total
    "Innovation": "GB.XPD.RSDV.GD.ZS",             # R&D spend % GDP
    "Wealth gaps": "SI.POV.GINI"                   # Gini index
}

# Helper: WB country codes
WB_USA = "USA"
WB_WORLD = "WLD"

# ──────────────────────────────────────────────────────────────────────────────
# CACHING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6*60*60)
def fred_series(series_id: str) -> pd.Series:
    s = fred.get_series(series_id)
    return s.dropna()

def _nearest_year_value(s: pd.Series, ref_date: pd.Timestamp):
    target = ref_date - timedelta(days=365)
    if len(s.index) == 0:
        return np.nan
    # pick closest date not after ref_date-365 by nearest match
    pos = s.index.get_indexer([target], method="nearest")[0]
    return float(s.iloc[pos])

@st.cache_data(ttl=6*60*60)
def fred_last_two(series_id: str, mode: str):
    """
    mode: 'level' — last and previous observation (raw)
          'yoy_pct' — YoY percent (last vs ~1yr earlier)
    """
    s = fred_series(series_id)
    if s.empty:
        return np.nan, np.nan
    if mode == "level":
        curr = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) > 1 else np.nan
        return curr, prev
    else:
        last_date = pd.to_datetime(s.index[-1])
        last_val = float(s.iloc[-1])
        prev_val = _nearest_year_value(s, last_date)
        if np.isnan(prev_val) or prev_val == 0:
            return np.nan, np.nan
        curr_yoy = (last_val/prev_val - 1.0) * 100.0
        # previous YoY one observation earlier
        if len(s) > 1:
            prev_date = pd.to_datetime(s.index[-2])
            prev_last = float(s.iloc[-2])
            prev_prev = _nearest_year_value(s, prev_date)
            prev_yoy = (prev_last/prev_prev - 1.0) * 100.0 if prev_prev not in (0, np.nan) else np.nan
        else:
            prev_yoy = np.nan
        return curr_yoy, prev_yoy

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
    """Return USA% of world for the given code (e.g., GDP, exports)."""
    us = wbdata.get_dataframe({code: "val"}, country=WB_USA, convert_date=True).dropna().sort_index()
    wd = wbdata.get_dataframe({code: "val"}, country=WB_WORLD, convert_date=True).dropna().sort_index()
    if us.empty or wd.empty:
        return np.nan, np.nan
    # align years
    common = us.join(wd, lsuffix="_us", rsuffix="_w").dropna()
    if common.empty:
        return np.nan, np.nan
    curr = float(common.iloc[-1]["val_us"]) / float(common.iloc[-1]["val_w"]) * 100.0
    prev = float(common.iloc[-2]["val_us"]) / float(common.iloc[-2]["val_w"]) * 100.0 if len(common) > 1 else np.nan
    return curr, prev

# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMS per indicator
# ──────────────────────────────────────────────────────────────────────────────
# How to compute current/previous for FRED items
FRED_MODE = {
    "Yield curve": "level",
    "Consumer confidence": "level",
    "Building permits": "level",
    "Unemployment claims": "level",
    "LEI (Conference Board Leading Economic Index)": "level",
    "GDP": "level",                     # keep level to match Unit
    "Capacity utilization": "level",
    "Inflation": "yoy_pct",
    "Retail sales": "yoy_pct",
    "Nonfarm payrolls": "level",
    "Wage growth": "yoy_pct",
    "Credit growth": "yoy_pct",
    "Fed funds futures": "level",       # proxy: FEDFUNDS level
    "Short rates": "level",
    "Industrial production": "yoy_pct",
    "Consumer/investment spending": "level",
    "Productivity growth": "yoy_pct",
    "Debt-to-GDP": "level",
    "Foreign reserves": "level",
    "Real rates": "level",
    "Trade balance": "level",
    "Credit spreads": "level",
    "Central bank printing (M2)": "yoy_pct",
    "Currency devaluation": "yoy_pct",
    "Fiscal deficits": "level",
    "Debt growth": "yoy_pct",
    "Income growth": "yoy_pct",
    "Debt service": "level",
    "Military spending": "level",       # may be replaced by WB %GDP
    "Debt burden": "level"
}

# Items we enrich via World Bank (USA)
WB_ITEMS = set(WB_US.keys()) | {"GDP share", "Trade dominance", "Economic output share", "Wealth gaps"}

# ──────────────────────────────────────────────────────────────────────────────
# BUILD TABLE (all 50 rows, thresholds always visible)
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for ind in INDICATORS:
    unit = UNITS.get(ind, "")
    current = np.nan
    previous = np.nan
    source = "—"

    # 1) Prefer World Bank for certain indicators (USA values where applicable)
    if ind in WB_US:
        try:
            current, previous = wb_last_two(WB_US[ind], WB_USA)
            source = "World Bank (USA)"
            # Make sure unit aligns for Innovation/Military spending/Trade share etc.
        except Exception as e:
            source = f"WB error: {e}"

    # 2) Derived shares vs world (GDP share, Trade dominance, Economic output share)
    elif ind in ("GDP share", "Economic output share"):
        try:
            current, previous = wb_share_of_world("NY.GDP.MKTP.CD")  # Current USD
            source = "World Bank (USA/World)"
            unit = "% of world"
        except Exception as e:
            source = f"WB share error: {e}"

    elif ind == "Trade dominance":
        try:
            current, previous = wb_share_of_world("NE.EXP.GNFS.CD")  # Exports current USD
            source = "World Bank (USA/World)"
            unit = "% of world exports"
        except Exception as e:
            source = f"WB share error: {e}"

    # 3) FRED for everything mapped
    if np.isnan(current) and ind in FRED_MAP:
        mode = FRED_MODE.get(ind, "level")
        try:
            current, previous = fred_last_two(FRED_MAP[ind], mode)
            source = "FRED" if source == "—" else (source + " + FRED")
        except Exception as e:
            source = f"FRED error: {e}"

    # Some items intentionally have no stable public series; they still show thresholds.
    # (Asset prices > traditional metrics, New buyers entering, Internal conflicts,
    #  Reserve currency usage dropping, Military losses, Corruption index,
    #  Education (PISA scores), Power index, Competitiveness index/WEF)

    # Compute delta
    delta = (current - previous) if (pd.notna(current) and pd.notna(previous)) else np.nan

    # Append row
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
st.markdown("## 📊 Econ Mirror — Full Indicator Table")
st.caption("All indicators shown. Thresholds always visible. Data fused from FRED + World Bank (USA) where reliable; others show thresholds only.")

st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown('<div class="muted">Tip: sort/filter from the column headers • Last refresh: ' + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + ' UTC</div>', unsafe_allow_html=True)
