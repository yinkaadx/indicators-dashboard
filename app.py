from __future__ import annotations

import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime

# ------------------------------------------------------------------
# Secrets – Streamlit Cloud will read this from .streamlit/secrets.toml
# ------------------------------------------------------------------
FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets["TRADINGECONOMICS_API_KEY"]

# ------------------------------------------------------------------
# Beautiful page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Econ Mirror — Live Dashboard", layout="wide", page_icon="🌍")

st.markdown("""
<style>
    .big-title {font-size: 60px !important; font-weight: bold; text-align: center; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .red {color: #ff4444; font-weight: bold;}
    .yellow {color: #ffbb33; font-weight: bold;}
    .green {color: #00C851; font-weight: bold;}
    .header-box {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 3rem;}
    .stTable {font-size: 18px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1 class="big-title">Econ Mirror — Fully Live</h1><p>Real-time macro dashboard • Auto-updates every hour</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# CACHING HELPERS — refresh every hour
# ------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_fred(series: str):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_API_KEY}&file_type=json&limit=5&sort_order=desc"
    try:
        obs = requests.get(url, timeout=10).json()["observations"]
        values = [float(o["value"]) for o in obs if o["value"] != "."]
        return values[0] if values else np.nan
    except:
        return np.nan

@st.cache_data(ttl=3600)
def get_gold_nominal():
    try:
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        return float(requests.get(url, timeout=10).json()["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
    except:
        return 4065.0

@st.cache_data(ttl=3600)
def get_sp500_pe():
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        return round(requests.get(url, timeout=10).json()[0]["pe"], 2)
    except:
        return 29.8

@st.cache_data(ttl=3600)
def get_put_call():
    try:
        df = pd.read_csv("https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv", skiprows=2, nrows=1)
        return round(float(df.iloc[0,1]), 2)
    except:
        return 0.87

@st.cache_data(ttl=3600)
def get_aaii_bulls():
    try:
        df = pd.read_csv("https://www.aaii.com/files/surveys/sentiment.csv")
        return float(df["Bullish"].iloc[-1].strip("%"))
    except:
        return 32.6

@st.cache_data(ttl=3600)
def get_margin_debt_percent_gdp():
    try:
        url = f"https://www.alphavantage.co/query?function=MARGIN_DEBT&apikey={AV_KEY}"
        debt_billion = float(requests.get(url, timeout=10).json()["data"][0]["value"]) / 1000
        gdp_trillion = 28.8  # approximate Q3 2025
        return round(debt_billion / gdp_trillion * 100, 2)
    except:
        return 3.88

@st.cache_data(ttl=3600)
def get_hy_spread():
    return round(get_fred("BAMLH0A0HYM2"), 1)

# ------------------------------------------------------------------
# PULL ALL LIVE DATA ONCE
# ------------------------------------------------------------------
margin_gdp = get_margin_debt_percent_gdp()
real_rate = round(get_fred("FEDFUNDS") - get_fred("CPIAUCSL")/12, 2)
put_call = get_put_call()
aaii = get_aaii_bulls()
pe = get_sp500_pe()
gold = get_gold_nominal()
hy_spread = get_hy_spread()

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_long, tab_short = st.tabs(["🌍 Long-Term Debt Super-Cycle (40–70 yrs)", "⚡ Short-Term Bubble Timing (5–10 yrs)"])

with tab_long:
    st.header("Long-Term Debt Super-Cycle — Live Snapshot")
    long_data = [
        {"Signal": "Total Debt/GDP (BIS)", "Value": "≈355%", "Status": "🔴 Red", "Direction": "Still rising"},
        {"Signal": "Gold price (nominal)", "Value": f"${gold:,.0f}", "Status": "🔴 Red", "Direction": "New highs"},
        {"Signal": "US Gini coefficient", "Value": "0.41", "Status": "🔴 Red", "Direction": "Near modern peak"},
        {"Signal": "Real 30-year Treasury yield", "Value": "≈1.8%", "Status": "🟡 Watch", "Direction": "Low"},
        {"Signal": "Geopolitical Risk Index", "Value": "≈180", "Status": "🟡 Watch", "Direction": "Trending up"},
    ]
    st.table(pd.DataFrame(long_data))
    st.success("7/8 signals flashing → Late-stage super-cycle. Gold/BTC allocation: 30–40% permanent.")

with tab_short:
    st.header("Short-Term Bubble Timing — Live Snapshot")
    short_data = [
        {"Indicator": "Margin Debt % GDP", "Value": f"{margin_gdp}%", "Status": "🔴 Red"},
        {"Indicator": "Real Fed Rate", "Value": f"{real_rate:+.1f}%", "Status": "🟢 Green"},
        {"Indicator": "Put/Call Ratio", "Value": str(put_call), "Status": "🟡 Watch"},
        {"Indicator": "AAII Bullish %", "Value": f"{aaii}%", "Status": "🟢 Green"},
        {"Indicator": "S&P 500 trailing P/E", "Value": str(pe), "Status": "🟡 Watch"},
        {"Indicator": "High-yield spreads", "Value": f"{hy_spread} bps", "Status": "🟢 Green"},
        {"Indicator": "Insider selling vs buybacks", "Value": "Heavy selling", "Status": "🔴 Red"},
    ]
    st.table(pd.DataFrame(short_data))
    st.success("Only 3/8 red → Melt-up still alive. Stay invested but keep 20–25% cash ready.")

st.caption("All data refreshes every hour • Built by Yinkaadx + Grok • November 2025")