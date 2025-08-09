import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import wbdata
import requests
import yfinance as yf
import plotly.express as px
from datetime import datetime, timezone
from functools import lru_cache

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG • clean, airy, no pre-selection
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Econ Mirror — Clean View", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Minimal CSS
st.markdown("""
<style>
/* spacing & fonts */
.block-container { padding-top: 1rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #e5e7eb; }
.small { color:#9ca3af; font-size:0.85rem; }
/* hero */
.hero {
  padding: 18px 20px; border:1px solid #1f2937; border-radius:14px;
  background: linear-gradient(135deg,#0b1020 0%,#0f172a 100%);
}
/* metric strip */
.kpi {
  border-radius:12px; padding:14px 16px; border:1px solid #1f2937;
  background:#0f172a; color:#e5e7eb;
}
.kpi .t { font-size:0.8rem; color:#9ca3af; margin-bottom:6px; }
.kpi .v { font-size:1.6rem; font-weight:700; }
.kpi .u { font-size:0.8rem; color:#9ca3af; }
/* section card */
.section {
  padding:14px 16px; border:1px solid #1f2937; border-radius:14px; background:#0b1020;
}
.tag { display:inline-block; padding:2px 8px; border:1px solid #1f2937; border-radius:999px; font-size:0.75rem; color:#9ca3af; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KEYS & SESSIONS
# ──────────────────────────────────────────────────────────────────────────────
fred = Fred(api_key=st.secrets["FRED_API_KEY"])
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/3.0"})
TIMEOUT = 12

# Region toggle (no selection required)
region = st.segmented_control("Region", options=["United States", "Global (World)"], default="United States", help="Switch between U.S. view (FRED + TE) and Global aggregates (World Bank).")

# ──────────────────────────────────────────────────────────────────────────────
# INDICATORS & MAPPINGS
# ──────────────────────────────────────────────────────────────────────────────
# US indicators (curated to avoid congestion)
US_INDICATORS = [
    "GDP", "Inflation", "Unemployment rate", "Nonfarm payrolls",
    "Retail sales", "Fed funds rate", "Industrial production", "Credit spreads",
    "Yield curve", "Building permits", "Wage growth", "Capacity utilization",
    "Debt-to-GDP", "Debt service", "M2 (money supply)"
]

# Global indicators (use World Bank aggregates)
GLOBAL_INDICATORS = [
    "GDP", "Inflation", "Unemployment rate", "Debt-to-GDP",
    "Trade balance (% GDP)", "M2 growth", "GDP per capita growth", "Government spending growth"
]

# FRED map (US only)
FRED_MAP = {
    "GDP": "GDP",                                # USD bn SAAR
    "Inflation": "CPIAUCSL",                     # CPI index
    "Unemployment rate": "UNRATE",               # %
    "Nonfarm payrolls": "PAYEMS",                # thousands
    "Retail sales": "RSXFS",                     # USD mn
    "Fed funds rate": "FEDFUNDS",                # %
    "Industrial production": "INDPRO",           # index
    "Credit spreads": "BAMLH0A0HYM2",            # bps (ICE BofA HY)
    "Yield curve": "T10Y2Y",                     # pct-pts
    "Building permits": "PERMIT",                # thousands
    "Wage growth": "AHETPI",                     # dollars/hour idx
    "Capacity utilization": "TCU",               # %
    "Debt-to-GDP": "GFDEGDQ188S",                # %
    "Debt service": "TDSP",                      # % income
    "M2 (money supply)": "M2SL"                  # USD bn
}

# Display units
UNITS_US = {
    "GDP": "USD bn (SAAR)", "Inflation": "Index", "Unemployment rate": "%", "Nonfarm payrolls": "Thous.",
    "Retail sales": "USD mn", "Fed funds rate": "%", "Industrial production": "Index", "Credit spreads": "bps",
    "Yield curve": "pct-pts", "Building permits": "Thous.", "Wage growth": "Index", "Capacity utilization": "%",
    "Debt-to-GDP": "%", "Debt service": "% income", "M2 (money supply)": "USD bn"
}

# World Bank codes (Global aggregates)
WB_GLOBAL = {
    "GDP": "NY.GDP.MKTP.CD",                         # USD current
    "Inflation": "FP.CPI.TOTL.ZG",                   # % YoY
    "Unemployment rate": "SL.UEM.TOTL.ZS",           # % labor force
    "Debt-to-GDP": "GC.DOD.TOTL.GD.ZS",              # % of GDP
    "Trade balance (% GDP)": "NE.RSB.GNFS.ZS",       # % of GDP
    "M2 growth": "FM.LBL.BMNY.ZG",                   # % YoY
    "GDP per capita growth": "NY.GDP.PCAP.KD.ZG",    # % YoY
    "Government spending growth": "NE.CON.GOVT.KD.ZG" # % YoY
}
WB_UNITS = {
    "GDP": "USD tn", "Inflation": "% YoY", "Unemployment rate": "%", "Debt-to-GDP": "% of GDP",
    "Trade balance (% GDP)": "% of GDP", "M2 growth": "% YoY", "GDP per capita growth": "% YoY",
    "Government spending growth": "% YoY"
}
WB_COUNTRY = {"United States": "USA", "Global (World)": "WLD"}

# Thresholds (concise badges)
THRESHOLDS = {
    "GDP": "2–4% YoY (healthy)", "Inflation": "2–3% (target-ish)", "Unemployment rate": "3–5% (low)",
    "Nonfarm payrolls": "+150K/m (steady)", "Retail sales": "+3–5% YoY", "Fed funds rate": "Cutting=easing",
    "Industrial production": "+2–5% YoY", "Credit spreads": ">500 bps = stress",
    "Yield curve": ">+1 pct-pt = steep", "Building permits": "+5% YoY", "Wage growth": ">3% YoY",
    "Capacity utilization": ">80%", "Debt-to-GDP": "<60%", "Debt service": "<=20% income",
    "M2 (money supply)": "+10% YoY = easing", "Trade balance (% GDP)": "Higher=improving",
    "M2 growth": "3–8% YoY", "GDP per capita growth": "≥2% YoY", "Government spending growth": "Stable"
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS • Caching without threads
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6*60*60)
def fred_series(series_id: str):
    s = fred.get_series(series_id)
    return s

@st.cache_data(ttl=6*60*60)
def fred_series_last2(series_id: str):
    s = fred_series(series_id).dropna()
    if len(s) == 0:
        return np.nan, np.nan
    curr = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else np.nan
    return curr, prev

@st.cache_data(ttl=6*60*60)
def wb_last2(code: str, country_code: str):
    df = wbdata.get_dataframe({code: "val"}, country=country_code, convert_date=True)
    df = df.dropna().sort_index()
    if df.empty:
        return np.nan, np.nan
    curr = float(df.iloc[-1]["val"])
    prev = float(df.iloc[-2]["val"]) if len(df) > 1 else np.nan
    return curr, prev

@st.cache_data(ttl=6*60*60)
def te_fetch(indicator_slug: str, country_slug: str):
    if not TE_KEY:
        return None
    url = f"https://api.tradingeconomics.com/indicators/country/{country_slug}?indicator={indicator_slug}&c={TE_KEY}"
    try:
        r = SESSION.get(url, timeout=12)
        if r.ok:
            js = r.json()
            if isinstance(js, list) and js:
                return js[0]
    except Exception:
        return None
    return None

def slugify(name: str):
    return name.lower().replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "").replace(">", "")

def fmt(value: float, unit: str, region: str, indicator: str):
    if pd.isna(value):
        return "—"
    # Special format for GDP (Global to trillions)
    if indicator == "GDP" and region == "Global (World)":
        return f"{value/1e12:.2f} {WB_UNITS['GDP']}"
    if indicator == "GDP" and region == "United States":
        return f"{value:.0f} {UNITS_US['GDP']}"
    if unit in ("USD mn","USD bn (SAAR)","USD bn"):
        return f"{value:,.0f} {unit}"
    if unit in ("Index",):
        return f"{value:.2f} {unit}"
    if "pct-pts" in unit:
        return f"{value:.2f} {unit}"
    if "%" in unit:
        return f"{value:.2f} {unit}"
    if unit in ("Thous.",):
        return f"{value:.0f} {unit}"
    return f"{value:.2f} {unit}"

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOAD (sequential, cached)
# ──────────────────────────────────────────────────────────────────────────────
if region == "United States":
    indicators = US_INDICATORS
else:
    indicators = GLOBAL_INDICATORS

rows = []
country_slug = "united-states" if region == "United States" else "world"
wb_country = WB_COUNTRY[region]

for ind in indicators:
    unit = UNITS_US.get(ind, WB_UNITS.get(ind, ""))
    current = np.nan
    previous = np.nan
    forecast = np.nan
    source = ""
    chart_series_id = None

    if region == "United States":
        if ind in FRED_MAP:
            chart_series_id = FRED_MAP[ind]
            try:
                current, previous = fred_series_last2(FRED_MAP[ind])
                source = "FRED"
            except Exception as e:
                source = f"FRED error: {e}"
        # modest forecast try via TE
        te = te_fetch(slugify(ind), country_slug)
        if te:
            try:
                if pd.isna(current) and te.get("Last") is not None:
                    current = float(te["Last"])
                if pd.isna(previous) and te.get("Previous") is not None:
                    previous = float(te["Previous"])
                if te.get("Forecast") is not None:
                    forecast = float(te["Forecast"])
                source = "FRED + TE" if "FRED" in source else "TE"
            except Exception:
                pass
        # PE example removed to avoid unreliable noise; could be added as separate tile if needed.

    else:  # Global
        code = WB_GLOBAL.get(ind)
        if code:
            try:
                current, previous = wb_last2(code, wb_country)
                source = "World Bank"
            except Exception as e:
                source = f"World Bank error: {e}"
        else:
            source = "—"

    delta = (current - previous) if not (pd.isna(current) or pd.isna(previous)) else np.nan
    rows.append({
        "indicator": ind,
        "unit": unit,
        "current": current,
        "previous": previous,
        "delta": delta,
        "forecast": forecast,
        "source": source,
        "series_id": chart_series_id,
    })

# ──────────────────────────────────────────────────────────────────────────────
# HERO • clean headline
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>📊 Econ Mirror — {region}</h1>
  <div class="small">Live macro overview. No clicks required. Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
</div>
""", unsafe_allow_html=True)
st.write("")

# ──────────────────────────────────────────────────────────────────────────────
# TOP STRIP • 6 key KPIs always visible (no selection)
# ──────────────────────────────────────────────────────────────────────────────
priority = ["GDP","Inflation","Unemployment rate","Retail sales","Fed funds rate","Industrial production"]
priority = [p for p in priority if p in [r["indicator"] for r in rows]]

cols = st.columns(len(priority))
for col, name in zip(cols, priority):
    r = next(x for x in rows if x["indicator"] == name)
    curr_txt = fmt(r["current"], r["unit"], region, name)
    delta_txt = "—" if pd.isna(r["delta"]) else f"{r['delta']:.2f}"
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="t">{name}</div>
          <div class="v">{curr_txt}</div>
          <div class="u">Δ vs prev: {delta_txt}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ──────────────────────────────────────────────────────────────────────────────
# TRENDS • one large clean chart (defaults to GDP, no selection needed)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Trends")
default_chart = "GDP" if any(r["indicator"] == "GDP" for r in rows) else rows[0]["indicator"]
r0 = next(x for x in rows if x["indicator"] == default_chart)

if region == "United States" and r0["series_id"]:
    try:
        s = fred_series(r0["series_id"]).dropna()
        df = s.to_frame(name=default_chart).reset_index()
        df.columns = ["Date", default_chart]
        fig = px.line(df, x="Date", y=default_chart, title=f"{default_chart} — historical trend (US)", template="plotly_dark")
        fig.update_traces(mode="lines+markers")
        fig.update_layout(height=420, hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Chart unavailable: {e}")
elif region == "Global (World)" and default_chart in WB_GLOBAL:
    try:
        code = WB_GLOBAL[default_chart]
        # fetch full series for chart
        df = wbdata.get_dataframe({code: default_chart}, country=WB_COUNTRY[region], convert_date=True).dropna().sort_index()
        out = df.reset_index().rename(columns={"date":"Date"})
        fig = px.line(out, x="Date", y=default_chart, title=f"{default_chart} — historical trend (Global)", template="plotly_dark")
        fig.update_traces(mode="lines+markers")
        fig.update_layout(height=420, hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Chart unavailable: {e}")
else:
    st.info("No time-series available for the default chart in this mode.")

# ──────────────────────────────────────────────────────────────────────────────
# MORE INDICATORS • decongested (collapsible)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("More indicators", expanded=False):
    others = [r for r in rows if r["indicator"] not in priority]
    if not others:
        st.write("No additional indicators in this mode.")
    else:
        grid = st.columns(3)
        for i, r in enumerate(others):
            curr_txt = fmt(r["current"], r["unit"], region, r["indicator"])
            delta_txt = "—" if pd.isna(r["delta"]) else f"{r['delta']:.2f}"
            with grid[i % 3]:
                st.markdown(f"""
                <div class="section">
                  <div><span class="tag">{r['source'] or '—'}</span><span class="tag">{THRESHOLDS.get(r['indicator'],'')}</span></div>
                  <h4 style="margin-top:8px;">{r['indicator']}</h4>
                  <div class="small">Current</div>
                  <div style="font-size:1.2rem; font-weight:700;">{curr_txt}</div>
                  <div class="small">Δ vs prev: {delta_txt} • Forecast: {"—" if pd.isna(r["forecast"]) else round(r["forecast"],2)}</div>
                </div>
                """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABLE • clean summary
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Table")
df_table = pd.DataFrame([{
    "Indicator": r["indicator"],
    "Current": None if pd.isna(r["current"]) else (round(r["current"]/1e12,2) if (region=="Global (World)" and r["indicator"]=="GDP") else round(r["current"],2)),
    "Previous": None if pd.isna(r["previous"]) else (round(r["previous"]/1e12,2) if (region=="Global (World)" and r["indicator"]=="GDP") else round(r["previous"],2)),
    "Delta": None if pd.isna(r["delta"]) else round(r["delta"],2),
    "Forecast": None if pd.isna(r["forecast"]) else round(r["forecast"],2),
    "Unit": (WB_UNITS.get(r["indicator"], UNITS_US.get(r["indicator"], "")) if region=="Global (World)" else UNITS_US.get(r["indicator"], "")),
    "Source": r["source"]
} for r in rows])

st.dataframe(df_table, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER ACTIONS
# ──────────────────────────────────────────────────────────────────────────────
colA, colB = st.columns([1,1])
with colA:
    if st.button("🔄 Refresh data (clear cache)"):
        st.cache_data.clear()
        st.rerun()
with colB:
    st.caption("Sources: FRED, World Bank, TradingEconomics (forecast where available).")
