import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Econ Mirror — Indicators Table", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:2.5rem} .stDataFrame{border:1px solid #1f2937;border-radius:10px} .muted{color:#9ca3af;font-size:0.85rem}</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SECRETS
# ──────────────────────────────────────────────────────────────────────────────
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# ──────────────────────────────────────────────────────────────────────────────
# INDICATORS (exactly your list, in order)
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
# THRESHOLDS (exact text you provided)
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
# UNITS (minimal; blank means not applicable/unknown)
# ──────────────────────────────────────────────────────────────────────────────
UNITS = {
    "Yield curve": "pct-pts",
    "Consumer confidence": "Index",
    "Building permits": "Thous.",
    "Unemployment claims": "Thous.",
    "LEI (Conference Board Leading Economic Index)": "Index",
    "GDP": "USD bn (SAAR)",
    "Capacity utilization": "%",
    "Inflation": "Index",
    "Retail sales": "USD mn",
    "Nonfarm payrolls": "Thous.",
    "Wage growth": "Index",
    "P/E ratios": "Ratio",
    "Credit growth": "%",
    "Fed funds futures": "%",
    "Short rates": "%",
    "Industrial production": "Index",
    "Consumer/investment spending": "USD bn",
    "Productivity growth": "%",
    "Debt-to-GDP": "%",
    "Foreign reserves": "USD bn",
    "Real rates": "%",
    "Trade balance": "USD bn",
    "Credit spreads": "bps",
    "Central bank printing (M2)": "USD bn",
    "Currency devaluation": "%",
    "Fiscal deficits": "% GDP",
    "Debt growth": "%",
    "Income growth": "%",
    "Debt service": "% income",
    "Military spending": "% GDP",
    "Debt burden": "%"
}

# ──────────────────────────────────────────────────────────────────────────────
# FRED SERIES MAP (US-centric proxies where possible)
# ──────────────────────────────────────────────────────────────────────────────
FRED_MAP = {
    "Yield curve": "T10Y2Y",
    "Consumer confidence": "UMCSENT",
    "Building permits": "PERMIT",
    "Unemployment claims": "ICSA",
    "LEI (Conference Board Leading Economic Index)": "USSLIND",
    "GDP": "GDP",
    "Capacity utilization": "TCU",
    "Inflation": "CPIAUCSL",
    "Retail sales": "RSXFS",
    "Nonfarm payrolls": "PAYEMS",
    "Wage growth": "AHETPI",
    "Credit growth": "TOTBKCR",
    "Fed funds futures": "FEDFUNDS",  # proxy
    "Short rates": "TB3MS",
    "Industrial production": "INDPRO",
    "Consumer/investment spending": "PCE",
    "Productivity growth": "OPHNFB",
    "Debt-to-GDP": "GFDEGDQ188S",
    "Foreign reserves": "TRESEUSM193N",
    "Real rates": "REAINTRATREARAT1YE",
    "Trade balance": "BOPGSTB",
    "Credit spreads": "BAMLH0A0HYM2",
    "Central bank printing (M2)": "M2SL",
    "Fiscal deficits": "FYFSD",
    "Debt growth": "GFDEBTN",
    "Income growth": "A067RO1Q156NBEA",
    "Debt service": "TDSP",
    "Military spending": "A063RC1Q027SBEA",
    "Debt burden": "GFDEBTN"
    # Others intentionally left unmapped to keep the app fast; they'll still show with thresholds.
}

# ──────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6*60*60)
def fred_last_two(series_id: str):
    s = fred.get_series(series_id)
    s = s.dropna()
    if s.empty:
        return np.nan, np.nan
    curr = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else np.nan
    return curr, prev

@st.cache_data(ttl=6*60*60)
def spx_trailing_pe():
    try:
        t = yf.Ticker("^GSPC")
        pe = t.info.get("trailingPE", np.nan)
        if pe is None:
            return np.nan
        return float(pe)
    except Exception:
        return np.nan

# ──────────────────────────────────────────────────────────────────────────────
# BUILD TABLE (no selections, all 50 rows)
# ──────────────────────────────────────────────────────────────────────────────
rows = []
for ind in INDICATORS:
    unit = UNITS.get(ind, "")
    current = np.nan
    previous = np.nan
    source = "—"

    if ind == "P/E ratios":
        pe = spx_trailing_pe()
        if not np.isnan(pe):
            current = pe
            previous = np.nan
            source = "yfinance (^GSPC trailingPE)"
    elif ind in FRED_MAP:
        try:
            current, previous = fred_last_two(FRED_MAP[ind])
            source = "FRED"
        except Exception as e:
            source = f"FRED error: {e}"

    delta = (current - previous) if not (np.isnan(current) or np.isnan(previous)) else np.nan

    rows.append({
        "Indicator": ind,
        "Current": None if np.isnan(current) else round(current, 2),
        "Previous": None if np.isnan(previous) else round(previous, 2),
        "Delta": None if np.isnan(delta) else round(delta, 2),
        "Unit": unit,
        "Threshold": THRESHOLDS.get(ind, "—"),
        "Source": source
    })

df = pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Econ Mirror — Full Indicator Table")
st.caption("All indicators shown below with thresholds. Data populated where reliable US proxies exist; others show thresholds only.")

st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown('<div class="muted">Tip: Click a column header to sort. Use the column menu to filter.</div>', unsafe_allow_html=True)
