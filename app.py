import os
import io
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from fredapi import Fred

st.set_page_config(page_title="Econ Mirror", layout="wide")

if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)

cofer_path = "data/imf_cofer_usd_share.csv"
if not os.path.exists(cofer_path):
    cofer_default = """
date,usd_share
2023-12-31,58.4
2024-03-31,58.1
2024-06-30,57.8
2024-09-30,57.2
2025-06-30,56.3
"""
    with open(cofer_path, "w") as f:
        f.write(cofer_default.strip() + "\n")

pe_path = "data/pe_sp500.csv"
if not os.path.exists(pe_path):
    pe_default = """
date,pe
2025-11-26,30.57
"""
    with open(pe_path, "w") as f:
        f.write(pe_default.strip() + "\n")

margin_path = "data/margin_finra.csv"
if not os.path.exists(margin_path):
    margin_default = """
date,debit_bil
2025-10-31,1180
"""
    with open(margin_path, "w") as f:
        f.write(margin_default.strip() + "\n")

breadth_path = "data/spx_percent_above_200dma.csv"
if not os.path.exists(breadth_path):
    breadth_default = """
date,value
2024-12-31,68.4
2025-01-31,72.1
2025-02-28,78.9
2025-03-31,81.2
2025-04-30,79.5
2025-05-31,74.3
2025-06-30,69.8
2025-07-31,65.2
2025-08-31,58.7
2025-09-30,52.1
2025-10-31,45.6
2025-11-26,16.0
"""
    with open(breadth_path, "w") as f:
        f.write(breadth_default.strip() + "\n")

fred = Fred(api_key=st.secrets["FRED_API_KEY"])
fmp_key = st.secrets["FMP_API_KEY"]
av_key = st.secrets["ALPHAVANTAGE_API_KEY"]

@st.cache_data(ttl=1800)
def fetch_fmp(endpoint):
    url = f"https://financialmodelingprep.com/api/v3/{endpoint}&apikey={fmp_key}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except:
        return None

@st.cache_data(ttl=1800)
def fetch_fred(series):
    try:
        df = fred.get_series_all_releases(series)
        if df is None or len(df) == 0:
            return None
        return df
    except:
        return None

@st.cache_data(ttl=1800)
def fetch_alpha_ts(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={av_key}"
    r = requests.get(url, timeout=30)
    try:
        return r.json()
    except:
        return None
def safe_read_csv(path, date_col):
    try:
        df = pd.read_csv(path)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
        df = df.dropna()
        if df.empty:
            return None
        return df.sort_values(date_col)
    except:
        return None

def mirror_latest_csv(path, time_col, value_col):
    df = safe_read_csv(path, time_col)
    if df is None or df.empty:
        return float("nan"), float("nan"), "Mirror empty", []
    if df.empty:
        return float("nan"), float("nan"), "Mirror empty", []
    df = df.dropna().sort_values(time_col)
    if df.empty:
        return float("nan"), float("nan"), "Mirror empty", []
    latest = df.iloc[-1][value_col]
    prev = df.iloc[-2][value_col] if len(df) > 1 else float("nan")
    return latest, prev, "OK", df.to_dict("records")

def get_fmp_index(symbol):
    data = fetch_fmp(f"historical-price-full/{symbol}?serietype=line")
    if not data or "historical" not in data:
        return None
    try:
        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna()
        df = df.sort_values("date")
        return df
    except:
        return None

def get_fmp_quote(symbol):
    data = fetch_fmp(f"quote/{symbol}?")
    if not data:
        return None
    try:
        return data[0]
    except:
        return None

def get_spx_data():
    return get_fmp_index("%5EGSPC")

def get_vix_data():
    return get_fmp_index("%5EVIX")

def spx_close_latest():
    df = get_spx_data()
    if df is None or df.empty:
        return float("nan")
    return float(df.iloc[-1]["close"])

def spx_ath_value():
    df = get_spx_data()
    if df is None or df.empty:
        return float("nan")
    return float(df["close"].max())

def spx_drawdown_pct():
    c = spx_close_latest()
    a = spx_ath_value()
    if np.isnan(c) or np.isnan(a) or a == 0:
        return float("nan")
    return (c / a - 1) * 100

def get_gold_usd():
    q = get_fmp_quote("XAUUSD")
    if not q:
        return float("nan")
    return float(q["price"])

def get_oil_wti():
    q = get_fmp_quote("WTIUSD")
    if not q:
        return float("nan")
    return float(q["price"])

def get_btc():
    q = get_fmp_quote("BTCUSD")
    if not q:
        return float("nan")
    return float(q["price"])

def real_assets_index():
    g = get_gold_usd()
    o = get_oil_wti()
    b = get_btc()
    if np.isnan(g) or np.isnan(o) or np.isnan(b):
        return float("nan")
    return (g / 2000) + (o / 90) + (b / 60000)

def fed_balance_yoy():
    df = fetch_fred("WALCL")
    if df is None:
        return float("nan")
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        if len(s) < 53:
            return float("nan")
        latest = s.iloc[-1]["value"]
        prev_yr = s.iloc[-53]["value"]
        return (latest - prev_yr) / prev_yr * 100
    except:
        return float("nan")

def sofr_spread():
    df = fetch_fred("SOFR")
    if df is None:
        return float("nan")
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        if len(s) < 2:
            return float("nan")
        latest = s.iloc[-1]["value"]
        prev = s.iloc[-2]["value"]
        return latest - prev
    except:
        return float("nan")

def spx_breadth():
    latest, prev, status, hist = mirror_latest_csv("data/spx_percent_above_200dma.csv", "date", "value")
    return latest
def cofer_usd_share():
    latest, prev, status, hist = mirror_latest_csv("data/imf_cofer_usd_share.csv", "date", "usd_share")
    return latest

def margin_debt():
    latest, prev, status, hist = mirror_latest_csv("data/margin_finra.csv", "date", "debit_bil")
    return latest

def spx_pe():
    latest, prev, status, hist = mirror_latest_csv("data/pe_sp500.csv", "date", "pe")
    return latest

def kill_1_vix_spike():
    df = get_vix_data()
    if df is None:
        return 0
    latest = df.iloc[-1]["close"]
    prev = df.iloc[-2]["close"] if len(df) > 1 else latest
    return 1 if latest > prev * 1.15 else 0

def kill_2_spx_drawdown():
    d = spx_drawdown_pct()
    return 1 if not np.isnan(d) and d > -5 else 0

def kill_3_spx_ytd():
    df = get_spx_data()
    if df is None or df.empty:
        return 0
    this_year = df[df["date"].dt.year == datetime.now().year]
    if this_year.empty:
        return 0
    start = this_year.iloc[0]["close"]
    end = this_year.iloc[-1]["close"]
    if start == 0:
        return 0
    y = (end - start) / start * 100
    return 1 if y > 10 else 0

def kill_4_margin_debt_spike():
    m = margin_debt()
    return 1 if not np.isnan(m) and m > 1000 else 0

def kill_5_hy_spread():
    df = fetch_fred("BAMLH0A0HYM2")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest < 4 else 0
    except:
        return 0

def kill_6_real_fed_rate():
    ff = fetch_fred("DFF")
    cpi = fetch_fred("CPIAUCSL")
    if ff is None or cpi is None:
        return 0
    try:
        f = ff.copy()
        f["date"] = pd.to_datetime(f["date"], format="%Y-%m-%d", errors="coerce")
        f = f.dropna().sort_values("date")
        c = cpi.copy()
        c["date"] = pd.to_datetime(c["date"], format="%Y-%m-%d", errors="coerce")
        c = c.dropna().sort_values("date")
        f = f.rename(columns={"value": "ff"})
        c = c.rename(columns={"value": "cpi"})
        merged = pd.merge_asof(f, c, on="date")
        merged["cpi_yoy"] = merged["cpi"].pct_change(12) * 100
        merged["real"] = merged["ff"] - merged["cpi_yoy"]
        latest = merged.iloc[-1]["real"]
        return 1 if latest < 1 else 0
    except:
        return 0

def kill_7_put_call():
    df = fetch_fred("PC_RATIO")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest < 0.9 else 0
    except:
        return 0

def kill_8_aaii():
    df = fetch_fred("AAIIBULL")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest > 35 else 0
    except:
        return 0

def kill_9_breadth():
    b = spx_breadth()
    return 1 if not np.isnan(b) and b < 25 else 0

def kill_10_fed_balance_sofr():
    y = fed_balance_yoy()
    s = sofr_spread()
    if np.isnan(y) or np.isnan(s):
        return 0
    return 1 if y > 5 and s > 0 else 0
def short_term_kill_count():
    k1 = kill_1_vix_spike()
    k2 = kill_2_spx_drawdown()
    k3 = kill_3_spx_ytd()
    k4 = kill_4_margin_debt_spike()
    k5 = kill_5_hy_spread()
    k6 = kill_6_real_fed_rate()
    k7 = kill_7_put_call()
    k8 = kill_8_aaii()
    k9 = kill_9_breadth()
    k10 = kill_10_fed_balance_sofr()
    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "k5": k5,
        "k6": k6,
        "k7": k7,
        "k8": k8,
        "k9": k9,
        "k10": k10,
        "count": k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 + k10
    }

def dark_1_real_rates():
    ff = fetch_fred("DFF")
    cpi = fetch_fred("CPIAUCSL")
    if ff is None or cpi is None:
        return 0
    try:
        f = ff.copy()
        f["date"] = pd.to_datetime(f["date"], format="%Y-%m-%d", errors="coerce")
        f = f.dropna().sort_values("date")
        c = cpi.copy()
        c["date"] = pd.to_datetime(c["date"], format="%Y-%m-%d", errors="coerce")
        c = c.dropna().sort_values("date")
        f = f.rename(columns={"value": "ff"})
        c = c.rename(columns={"value": "cpi"})
        merged = pd.merge_asof(f, c, on="date")
        merged["cpi_yoy"] = merged["cpi"].pct_change(12) * 100
        merged["real"] = merged["ff"] - merged["cpi_yoy"]
        latest = merged.iloc[-1]["real"]
        return 1 if latest < 1 else 0
    except:
        return 0

def dark_2_debt_to_gdp():
    df = fetch_fred("GFDEGDQ188S")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest > 110 else 0
    except:
        return 0

def dark_3_wealth_gap():
    df = fetch_fred("WFRBLB50207")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest > 2.5 else 0
    except:
        return 0

def dark_4_labor_share():
    df = fetch_fred("LABSHPUSA156NRUG")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest < 60 else 0
    except:
        return 0

def dark_5_productivity():
    df = fetch_fred("OPHNFB")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest < 2 else 0
    except:
        return 0

def dark_6_cb_gold_buying():
    df = fetch_fred("GOLDAMGBD228NLBM")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest > 1800 else 0
    except:
        return 0

def dark_7_currency_wars():
    dxy = get_fmp_quote("DXY")
    if not dxy:
        return 0
    try:
        v = float(dxy["price"])
        return 1 if v > 110 else 0
    except:
        return 0

def dark_8_real_assets():
    v = real_assets_index()
    return 1 if not np.isnan(v) and v > 3 else 0

def dark_9_cofer():
    v = cofer_usd_share()
    return 1 if not np.isnan(v) and v < 57 else 0

def dark_10_pe():
    p = spx_pe()
    return 1 if not np.isnan(p) and p > 28 else 0

def dark_11_credit_spread():
    df = fetch_fred("BAMLH0A0HYM2")
    if df is None:
        return 0
    try:
        s = df[df["value"].notna()].copy()
        s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d", errors="coerce")
        s = s.dropna().sort_values("date")
        latest = s.iloc[-1]["value"]
        return 1 if latest < 4 else 0
    except:
        return 0
def long_term_dark_count(reset_flag):
    d1 = dark_1_real_rates()
    d2 = dark_2_debt_to_gdp()
    d3 = dark_3_wealth_gap()
    d4 = dark_4_labor_share()
    d5 = dark_5_productivity()
    d6 = dark_6_cb_gold_buying()
    d7 = dark_7_currency_wars()
    d8 = dark_8_real_assets()
    d9 = dark_9_cofer()
    d10 = dark_10_pe()
    d11 = dark_11_credit_spread()
    dark_total = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11
    nr1 = 1 if d6 == 1 else 0
    nr2 = 1 if d9 == 1 else 0
    nr3 = 1 if reset_flag else 0
    no_return_total = nr1 + nr2 + nr3
    return {
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "d5": d5,
        "d6": d6,
        "d7": d7,
        "d8": d8,
        "d9": d9,
        "d10": d10,
        "d11": d11,
        "dark_total": dark_total,
        "nr1": nr1,
        "nr2": nr2,
        "nr3": nr3,
        "no_return_total": no_return_total
    }

def regime_box(kill_count, near_ath, dark_total, no_return_total):
    if kill_count >= 7 and near_ath:
        return "FINAL TOP: SELL 80–90% THIS WEEK"
    if dark_total >= 8 and no_return_total >= 2:
        return "POINT OF NO RETURN: 80–100% HARD ASSETS FOREVER"
    return "NEUTRAL"

def compute_regime(reset_flag):
    st_kills = short_term_kill_count()
    kill_total = st_kills["count"]
    close = spx_close_latest()
    ath = spx_ath_value()
    near_ath = False
    if not np.isnan(close) and not np.isnan(ath) and ath > 0:
        if close > ath * 0.92:
            near_ath = True
    lt_dark = long_term_dark_count(reset_flag)
    dark_total = lt_dark["dark_total"]
    no_return_total = lt_dark["no_return_total"]
    box = regime_box(kill_total, near_ath, dark_total, no_return_total)
    return kill_total, dark_total, no_return_total, box
def top_banner(kill_total, dark_total):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"<div style='background:#111;padding:20px;border-radius:10px;font-size:28px;color:white;text-align:center;'>CURRENT REGIME<br>{kill_total}/10 short-term kill • {dark_total}/11 long-term dark-red</div>",
            unsafe_allow_html=True
        )
    with c2:
        spx = spx_close_latest()
        dd = spx_drawdown_pct()
        ath = spx_ath_value()
        t = f"SPX: {spx:.2f} | Drawdown: {dd:.2f}% | ATH: {ath:.2f}"
        st.markdown(
            f"<div style='background:#111;padding:20px;border-radius:10px;font-size:22px;color:white;text-align:center;'>{t}</div>",
            unsafe_allow_html=True
        )

def core_tab():
    c1, c2, c3 = st.columns(3)
    spx = spx_close_latest()
    vix_df = get_vix_data()
    vix = vix_df.iloc[-1]["close"] if vix_df is not None and not vix_df.empty else float("nan")
    gold = get_gold_usd()
    oil = get_oil_wti()
    btc = get_btc()
    pe = spx_pe()
    margin = margin_debt()
    cofer = cofer_usd_share()
    breadth = spx_breadth()
    with c1:
        st.metric("S&P 500", f"{spx:.2f}")
        st.metric("VIX", f"{vix:.2f}")
        st.metric("Breadth <200DMA%", f"{breadth:.2f}")
    with c2:
        st.metric("Gold (USD)", f"{gold:.2f}")
        st.metric("Oil WTI", f"{oil:.2f}")
        st.metric("BTC", f"{btc:.2f}")
    with c3:
        st.metric("S&P PE", f"{pe:.2f}")
        st.metric("Margin Debt (B)", f"{margin:.2f}")
        st.metric("COFER USD%", f"{cofer:.2f}")

def short_term_tab():
    k = short_term_kill_count()
    df = pd.DataFrame([
        ["1. VIX Spike", k["k1"]],
        ["2. SPX Drawdown", k["k2"]],
        ["3. SPX YTD", k["k3"]],
        ["4. Margin Debt Spike", k["k4"]],
        ["5. HY Spread", k["k5"]],
        ["6. Real Fed Rate", k["k6"]],
        ["7. Put/Call", k["k7"]],
        ["8. AAII Bulls", k["k8"]],
        ["9. Breadth Collapse", k["k9"]],
        ["10. Fed Balance + SOFR", k["k10"]],
    ], columns=["Indicator", "Signal"])
    st.dataframe(df, use_container_width=True)
    st.markdown(f"### Total: {k['count']}/10")

def long_term_tab(reset_flag):
    d = long_term_dark_count(reset_flag)
    df = pd.DataFrame([
        ["1. Real Rates <1%", d["d1"]],
        ["2. Debt/GDP >110%", d["d2"]],
        ["3. Wealth Gap", d["d3"]],
        ["4. Labor Share <60", d["d4"]],
        ["5. Productivity Weak", d["d5"]],
        ["6. CB Gold Buying", d["d6"]],
        ["7. Currency Wars", d["d7"]],
        ["8. Real Assets Index", d["d8"]],
        ["9. COFER Collapse", d["d9"]],
        ["10. PE >28", d["d10"]],
        ["11. Credit Spread <4", d["d11"]],
    ], columns=["Indicator", "Dark Red"])
    st.dataframe(df, use_container_width=True)
    st.markdown(f"### Dark Red: {d['dark_total']}/11")
    st.markdown(f"### No-Return: {d['no_return_total']}/3")
st.set_page_config(page_title="Econ Mirror", layout="wide")

if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)

auto_create("data/imf_cofer_usd_share.csv", "date,usd_share\n2023-12-31,58.4\n2024-03-31,58.1\n2024-06-30,57.8\n2024-09-30,57.2\n2025-06-30,56.3")
auto_create("data/pe_sp500.csv", "date,pe\n2025-11-26,30.57")
auto_create("data/margin_finra.csv", "date,debit_bil\n2025-10-31,1180")
auto_create("data/spx_percent_above_200dma.csv",
            "date,value\n2024-12-31,68.4\n2025-01-31,72.1\n2025-02-28,78.9\n2025-03-31,81.2\n2025-04-30,79.5\n2025-05-31,74.3\n2025-06-30,69.8\n2025-07-31,65.2\n2025-08-31,58.7\n2025-09-30,52.1\n2025-10-31,45.6\n2025-11-26,16.0")

k = short_term_kill_count()
d = long_term_dark_count(get_reset_flag())

top_banner(k["count"], d["dark_total"])

tab1, tab2, tab3 = st.tabs(["Core Mirror", "Short-Term Combo", "Long-Term Supercycle"])

with tab1:
    core_tab()

with tab2:
    short_term_tab()

with tab3:
    long_term_tab(get_reset_flag())
st.caption("Econ Mirror — Immortal Edition")
