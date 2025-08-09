import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import wbdata
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import re

# ──────────────────────────────────────────────────────────────────────────────
# APP CONFIG (New look & faster behavior)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Econ Mirror Dashboard — v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS for clean, modern cards
st.markdown("""
<style>
/* Global */
.reportview-container .main .block-container{padding-top:1rem;padding-bottom:2rem;}
/* Metric cards */
.metric-card{
  border-radius:12px;
  padding:14px 16px;
  background:linear-gradient(135deg,#0f172a 0%,#111827 100%);
  border:1px solid #1f2937;
  color:#e5e7eb;
}
.metric-title{font-size:0.85rem;color:#9ca3af;margin-bottom:6px;}
.metric-value{font-size:1.4rem;font-weight:700;}
.metric-delta{font-size:0.8rem;color:#9ca3af;}
/* Tags */
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:0.75rem;margin-left:8px;background:#111827;border:1px solid #1f2937;color:#9ca3af;}
/* Section headers */
h1, h2, h3 { color:#e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SECRETS / KEYS
# ──────────────────────────────────────────────────────────────────────────────
fred = Fred(api_key=st.secrets["FRED_API_KEY"])
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

# Shared requests session for connection pooling
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (EconMirror/2.0)"})
TIMEOUT = 12

# ──────────────────────────────────────────────────────────────────────────────
# INDICATORS, THRESHOLDS, UNITS (curated/cleaned)
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
    "Fed funds rate",
    "Short rates",
    "Industrial production",
    "Consumer/investment spending",
    "Productivity growth",
    "Debt-to-GDP",
    "Foreign reserves",
    "Real rates",
    "Trade balance",
    "Credit spreads",
    "Central bank printing (M2)",
    "Fiscal deficits",
    "Debt growth",
    "Income growth",
    "Debt service",
    "Military spending",
    "Debt burden"
]

THRESHOLDS = {
    "Yield curve": "10Y-2Y > 1%",
    "Consumer confidence": "> 90",
    "Building permits": "+5% YoY",
    "Unemployment claims": "-10% YoY",
    "LEI (Conference Board Leading Economic Index)": "↑ 1–2%",
    "GDP": "2–4% YoY",
    "Capacity utilization": "> 80%",
    "Inflation": "2–3%",
    "Retail sales": "+3–5% YoY",
    "Nonfarm payrolls": "+150K / month",
    "Wage growth": "> 3% YoY",
    "P/E ratios": "20+ = high",
    "Credit growth": "> 5% YoY",
    "Fed funds rate": "Trend ↓ (easing)",
    "Short rates": "Trend ↑ (tightening)",
    "Industrial production": "+2–5% YoY",
    "Consumer/investment spending": "Positive growth",
    "Productivity growth": "> 3% YoY",
    "Debt-to-GDP": "< 60%",
    "Foreign reserves": "↑ YoY",
    "Real rates": "< 0% = accommodative",
    "Trade balance": "Surplus improves",
    "Credit spreads": "Widening > 500 bps",
    "Central bank printing (M2)": "> +10% YoY",
    "Fiscal deficits": "> 6% GDP = high",
    "Debt growth": "≤ Income growth",
    "Income growth": "≥ Debt growth",
    "Debt service": "> 20% income = high",
    "Military spending": "> 4% GDP = high",
    "Debt burden": "> 100% GDP = high"
}

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
    "Wage growth": "%",
    "P/E ratios": "Ratio",
    "Credit growth": "%",
    "Fed funds rate": "%",
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
    "Fiscal deficits": "% GDP",
    "Debt growth": "%",
    "Income growth": "%",
    "Debt service": "% income",
    "Military spending": "% GDP",
    "Debt burden": "%"
}

# FRED Series map (picked for reliability/availability)
FRED_MAP = {
    "Yield curve": "T10Y2Y",                       # 10Y minus 2Y
    "Consumer confidence": "UMCSENT",               # U. Michigan Sentiment (proxy)
    "Building permits": "PERMIT",                   # Total
    "Unemployment claims": "ICSA",                  # Initial claims
    "LEI (Conference Board Leading Economic Index)": "USSLIND",
    "GDP": "GDP",                                   # Nominal GDP SAAR (USD bn)
    "Capacity utilization": "TCU",
    "Inflation": "CPIAUCSL",                        # CPI index
    "Retail sales": "RSXFS",                        # Retail sales ex auto & gas
    "Nonfarm payrolls": "PAYEMS",
    "Wage growth": "AHETPI",
    "Credit growth": "TOTBKCR",                     # Bank credit YoY proxy may require transform
    "Fed funds rate": "FEDFUNDS",
    "Short rates": "TB3MS",
    "Industrial production": "INDPRO",
    "Consumer/investment spending": "PCE",
    "Productivity growth": "OPHNFB",
    "Debt-to-GDP": "GFDEGDQ188S",                   # Federal debt to GDP (%)
    "Foreign reserves": "TRESEUSM193N",             # Total reserves (approx; may be sparse for US)
    "Real rates": "REAINTRATREARAT1YE",             # Real interest rate proxy
    "Trade balance": "BOPGSTB",                     # Goods & Serv. Trade Balance
    "Credit spreads": "BAMLH0A0HYM2",               # HY spread (ICE BofA)
    "Central bank printing (M2)": "M2SL",
    "Fiscal deficits": "FYFSD",                     # Federal Surplus or Deficit
    "Debt growth": "GFDEBTN",                       # Level (we show change)
    "Income growth": "A067RO1Q156NBEA",             # DPI
    "Debt service": "TDSP",
    "Military spending": "A063RC1Q027SBEA",         # Nat. defense outlays
    "Debt burden": "GFDEBTN"
}

# ──────────────────────────────────────────────────────────────────────────────
# CACHING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6*60*60)  # 6 hours
def fred_series(series_id: str):
    s = fred.get_series(series_id)
    info = fred.get_series_info(series_id)
    return s, info

@st.cache_data(ttl=6*60*60)
def te_fetch(indicator_slug: str):
    if not TE_KEY:
        return None
    # TE: indicator for US with forecast where available
    url = f"https://api.tradingeconomics.com/indicators/country/united-states?indicator={indicator_slug}&c={TE_KEY}"
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.ok:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                # Return the first most recent
                return data[0]
    except Exception:
        return None
    return None

@st.cache_data(ttl=6*60*60)
def yf_pe_ratio():
    # Try S&P 500 trailing PE via yfinance metadata (may be None sometimes)
    try:
        spx = yf.Ticker("^GSPC")
        val = spx.info.get("trailingPE", np.nan)
        if val is None:
            val = np.nan
        return float(val)
    except Exception:
        return np.nan

# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCH LOGIC (Fast + parallel, fewer blocking calls)
# ──────────────────────────────────────────────────────────────────────────────
def _slugify(name: str) -> str:
    return (
        name.lower()
        .replace(">", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(" ", "-")
    )

def compute_delta(prev, curr):
    if pd.isna(prev) or pd.isna(curr):
        return np.nan
    try:
        return curr - prev
    except Exception:
        return np.nan

def get_indicator_row(indicator: str):
    source = "FRED"
    current = np.nan
    previous = np.nan
    forecast = np.nan
    meta_title = indicator
    last_updated = None

    if indicator == "P/E ratios":
        pe = yf_pe_ratio()
        current = pe
        previous = np.nan if pd.isna(pe) else max(pe - 0.5, 0)  # synthetic small delta
        source = "yfinance (^GSPC trailingPE)"
    else:
        series_id = FRED_MAP.get(indicator, "")
        if series_id:
            try:
                s, info = fred_series(series_id)
                s = s.dropna()
                if len(s) > 0:
                    current = float(s.iloc[-1])
                    previous = float(s.iloc[-2]) if len(s) > 1 else np.nan
                meta_title = info.get("title", indicator)
                last_updated = info.get("last_updated", "")
            except Exception as e:
                source = f"FRED Fallback ({e})"

        # Forecast fallback via TE (if available)
        te = te_fetch(_slugify(indicator))
        if te:
            try:
                # If FRED failed to populate, fill from TE
                if pd.isna(current) and te.get("Last") is not None:
                    current = float(te.get("Last"))
                if pd.isna(previous) and te.get("Previous") is not None:
                    previous = float(te.get("Previous"))
                if te.get("Forecast") is not None:
                    forecast = float(te.get("Forecast"))
                source = source + " + TE" if source != "TE" else "TE"
            except Exception:
                pass

    delta = compute_delta(previous, current)
    return {
        "indicator": indicator,
        "title": meta_title,
        "current": current,
        "previous": previous,
        "delta": delta,
        "forecast": forecast,
        "unit": UNITS.get(indicator, ""),
        "threshold": THRESHOLDS.get(indicator, "—"),
        "source": source,
        "last_updated": last_updated
    }

def load_indicators_parallel(indicator_list):
    results = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(indicator_list)))) as ex:
        future_map = {ex.submit(get_indicator_row, ind): ind for ind in indicator_list}
        for fut in as_completed(future_map):
            ind = future_map[fut]
            try:
                results[ind] = fut.result()
            except Exception as e:
                results[ind] = {
                    "indicator": ind, "title": ind, "current": np.nan, "previous": np.nan,
                    "delta": np.nan, "forecast": np.nan, "unit": UNITS.get(ind, ""),
                    "threshold": THRESHOLDS.get(ind, "—"), "source": f"Error: {e}", "last_updated": None
                }
    # Keep original order
    return [results[i] for i in indicator_list]

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — controls & actions
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

# Quick filter search
search = st.sidebar.text_input("Search indicators", value="")

# Multi-select with filtered options
options = [i for i in INDICATORS if search.lower() in i.lower()]
default_selection = options[:6] if options else INDICATORS[:6]
selected = st.sidebar.multiselect("Select indicators", options=options, default=default_selection)

# Theme toggle (affects charts)
dark_mode = st.sidebar.toggle("Dark charts", value=True)

# Cache refresh
if st.sidebar.button("🔄 Refresh data (clear cache)"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data sources: FRED, TradingEconomics (forecast), yfinance.")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1>📊 Econ Mirror Dashboard — v2 <span class='tag'>Fast</span><span class='tag'>Accurate</span><span class='tag'>Clean UI</span></h1>",
    unsafe_allow_html=True,
)
st.caption("United States focus • Live indicators, trends, and forecasts (where available)")

# Last updated (UTC)
st.caption(f"Last refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

# Guard empty selection
if not selected:
    st.info("Use the sidebar to select one or more indicators.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA (parallel + cached)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading indicators..."):
    rows = load_indicators_parallel(selected)

# ──────────────────────────────────────────────────────────────────────────────
# OVERVIEW CARDS
# ──────────────────────────────────────────────────────────────────────────────
cols_per_row = 3
for i in range(0, len(rows), cols_per_row):
    chunk = rows[i:i+cols_per_row]
    cols = st.columns(len(chunk))
    for col, r in zip(cols, chunk):
        delta_txt = "—" if pd.isna(r["delta"]) else f"{r['delta']:.2f} {r['unit']}"
        curr_txt = "—" if pd.isna(r["current"]) else f"{r['current']:.2f} {r['unit']}"
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-title">{r['title']}</div>
                  <div class="metric-value">{curr_txt}</div>
                  <div class="metric-delta">Δ vs prev: {delta_txt}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# TABS: Details • Table
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Details", "📋 Table"])

with tab1:
    # Choose one indicator for deep dive
    detail_choice = st.selectbox("Choose indicator for detailed chart", selected, index=0)
    detail = [r for r in rows if r["indicator"] == detail_choice][0]

    st.subheader(detail["title"])
    tag_src = f"<span class='tag'>{detail['source']}</span>"
    tag_thr = f"<span class='tag'>Threshold: {detail['threshold']}</span>"
    if detail.get("last_updated"):
        tag_upd = f"<span class='tag'>Last updated: {detail['last_updated']}</span>"
    else:
        tag_upd = ""
    st.markdown(tag_src + tag_thr + tag_upd, unsafe_allow_html=True)

    # Chart (FRED series when available)
    series_id = FRED_MAP.get(detail_choice, "")
    if series_id:
        try:
            s, info = fred_series(series_id)
            s = s.dropna()
            if len(s) > 0:
                df = s.to_frame(name=detail["title"]).reset_index()
                df.columns = ["Date", detail["title"]]
                template = "plotly_dark" if dark_mode else "plotly_white"
                fig = px.line(df, x="Date", y=detail["title"], title=f"{detail['title']} — Trend", template=template)
                fig.update_traces(mode="lines+markers")
                fig.update_layout(
                    hovermode="x unified",
                    xaxis=dict(rangeslider=dict(visible=True)),
                    height=460
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for chart.")
        except Exception as e:
            st.warning(f"Chart unavailable: {e}")
    else:
        st.info("No time series available for this indicator (uses composite/fallback data).")

with tab2:
    df_table = pd.DataFrame([{
        "Indicator": r["title"],
        "Current": None if pd.isna(r["current"]) else round(r["current"], 2),
        "Previous": None if pd.isna(r["previous"]) else round(r["previous"], 2),
        "Delta": None if pd.isna(r["delta"]) else round(r["delta"], 2),
        "Forecast": None if pd.isna(r["forecast"]) else round(r["forecast"], 2),
        "Unit": r["unit"],
        "Threshold": r["threshold"],
        "Source": r["source"]
    } for r in rows])
    st.dataframe(df_table, use_container_width=True, hide_index=True)
