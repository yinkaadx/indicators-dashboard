import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
import wbdata
import requests
import os
from typing import Any, Tuple

st.set_page_config(page_title="Econ Mirror — Mirrored & Auto-Updated", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:2.5rem} .stDataFrame{border:1px solid #1f2937;border-radius:10px} .muted{color:#9ca3af;font-size:0.85rem}</style>", unsafe_allow_html=True)

fred = Fred(api_key=st.secrets["FRED_API_KEY"])
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/4.5-Mirrors"})
DATA_DIR = "data"
WB_DIR = os.path.join(DATA_DIR, "wb")
FRED_DIR = os.path.join(DATA_DIR, "fred")

# ---------- safe utils ----------
def to_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)): return np.nan
        if isinstance(x, (list, tuple)) and len(x)==1: x = x[0]
        return float(x)
    except Exception:
        return np.nan

def is_nan(x: Any) -> bool:
    return pd.isna(x)

def latest_prev_from_df(df: pd.DataFrame, col_value: str, col_time: str|None=None) -> Tuple[float,float]:
    if df is None or df.empty or col_value not in df.columns: return np.nan, np.nan
    if col_time and col_time in df.columns: df = df.sort_values(col_time)
    else: df = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df
    cur = to_float(df.iloc[-1][col_value]); prev = to_float(df.iloc[-2][col_value]) if len(df)>1 else np.nan
    return cur, prev

# ---------- indicators / thresholds / units (unchanged from your last good set) ----------
INDICATORS = [
    "Yield curve","Consumer confidence","Building permits","Unemployment claims",
    "LEI (Conference Board Leading Economic Index)","GDP","Capacity utilization","Inflation","Retail sales",
    "Nonfarm payrolls","Wage growth","P/E ratios","Credit growth","Fed funds futures","Short rates",
    "Industrial production","Consumer/investment spending","Productivity growth","Debt-to-GDP","Foreign reserves",
    "Real rates","Trade balance","Asset prices > traditional metrics (Shiller CAPE)",
    "New buyers entering (FINRA Margin Debt — FRED proxy)","Wealth gaps (Gini, WB)","Credit spreads",
    "Central bank printing (M2)","Currency devaluation","Fiscal deficits","Debt growth","Income growth","Debt service",
    "Education investment (WB %GDP)","R&D patents (WB count)","Competitiveness index / Competitiveness (WEF) (WB LPI overall)",
    "GDP per capita growth (WB)","Trade share (WB, Trade %GDP)","Military spending (WB %GDP)","Internal conflicts (WGI Political Stability)",
    "Reserve currency usage dropping (IMF COFER USD share)","Military losses (UCDP Battle-related deaths — Global)",
    "Economic output share (USA % of world GDP)","Corruption index (WGI Control of Corruption)","Working population (WB, 15–64 %)",
    "Education (PISA scores — OECD Math mean)","Innovation (WB R&D spend %GDP)","GDP share (USA % of world GDP)",
    "Trade dominance (USA % of world exports)","Power index (CINC — USA)","Debt burden"
]
THRESHOLDS = {
    "Yield curve":"10Y–2Y > 1% (steepens)","Consumer confidence":"> 90 index (rising)","Building permits":"+5% YoY (increasing)",
    "Unemployment claims":"−10% YoY (falling)","LEI (Conference Board Leading Economic Index)":"Up 1–2% (positive)","GDP":"2–4% YoY (rising)",
    "Capacity utilization":"> 80% (high)","Inflation":"2–3% (moderate)","Retail sales":"+3–5% YoY (rising)","Nonfarm payrolls":"+150K/month (steady)",
    "Wage growth":"> 3% YoY (rising)","P/E ratios":"20+ (high)","Credit growth":"> 5% YoY (increasing)","Fed funds futures":"Hikes implied +0.5%+",
    "Short rates":"Rising (tightening)","Industrial production":"+2–5% YoY (increasing)","Consumer/investment spending":"Positive growth (high)",
    "Productivity growth":"> 3% YoY (rising)","Debt-to-GDP":"< 60% (low)","Foreign reserves":"+10% YoY (increasing)","Real rates":"< −1% (falling)",
    "Trade balance":"Surplus > 2% of GDP (improving)","Asset prices > traditional metrics (Shiller CAPE)":"P/E +20% (high vs. fundamentals)",
    "New buyers entering (FINRA Margin Debt — FRED proxy)":"+15% (increasing)","Wealth gaps (Gini, WB)":"Top 1% share +5% (widening)","Credit spreads":"> 500 bps (widening)",
    "Central bank printing (M2)":"+10% YoY (printing)","Currency devaluation":"−10% to −20% (devaluation)","Fiscal deficits":"> 6% of GDP (high)",
    "Debt growth":"+5–10% gap above income growth","Income growth":"Debt–income growth gap < 5%","Debt service":"> 20% of incomes (high)",
    "Education investment (WB %GDP)":"+5% of budget YoY (surge)","R&D patents (WB count)":"+10% YoY (rising)","Competitiveness index / Competitiveness (WEF) (WB LPI overall)":"+5 ranks (improving)",
    "GDP per capita growth (WB)":"+3% YoY (accelerating)","Trade share (WB, Trade %GDP)":"+2% of global share (expanding)","Military spending (WB %GDP)":"> 4% of GDP (peaking)",
    "Internal conflicts (WGI Political Stability)":"Protests +20% (rising)","Reserve currency usage dropping (IMF COFER USD share)":"−5% of global share (dropping)",
    "Military losses (UCDP Battle-related deaths — Global)":"Defeats +1/year (increasing)","Economic output share (USA % of world GDP)":"−2% of global share (falling)",
    "Corruption index (WGI Control of Corruption)":"−10 points (worsening)","Working population (WB, 15–64 %)":"−1% YoY (aging)","Education (PISA scores — OECD Math mean)":" > 500 (top)",
    "Innovation (WB R&D spend %GDP)":"Patents > 20% of global (high)","GDP share (USA % of world GDP)":"+2% of global share (growing)","Trade dominance (USA % of world exports)":"> 15% of global trade (dominance)",
    "Power index (CINC — USA)":"Composite 8–10/10 (max)","Debt burden":"> 100% of GDP (high)"
}
UNITS = {
    "Yield curve":"pct-pts","Consumer confidence":"Index","Building permits":"Thous.","Unemployment claims":"Thous.","LEI (Conference Board Leading Economic Index)":"Index",
    "GDP":"USD bn (SAAR)","Capacity utilization":"%","Inflation":"% YoY","Retail sales":"% YoY","Nonfarm payrolls":"Thous.","Wage growth":"% YoY","P/E ratios":"Ratio",
    "Credit growth":"% YoY","Fed funds futures":"% (FFR proxy)","Short rates":"%","Industrial production":"% YoY","Consumer/investment spending":"USD bn","Productivity growth":"% YoY",
    "Debt-to-GDP":"% of GDP","Foreign reserves":"USD bn","Real rates":"%","Trade balance":"USD bn","Credit spreads":"bps","Central bank printing (M2)":"% YoY","Currency devaluation":"% YoY",
    "Fiscal deficits":"USD bn","Debt growth":"% YoY","Income growth":"% YoY","Debt service":"% income","Education investment (WB %GDP)":"% GDP","R&D patents (WB count)":"Number",
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)":"Index (0–5)","GDP per capita growth (WB)":"% YoY","Trade share (WB, Trade %GDP)":"% of GDP","Military spending (WB %GDP)":"% GDP",
    "Internal conflicts (WGI Political Stability)":"Index (−2.5 to +2.5)","Reserve currency usage dropping (IMF COFER USD share)":"% of allocated FX reserves","Military losses (UCDP Battle-related deaths — Global)":"Deaths (annual)",
    "Economic output share (USA % of world GDP)":"% of world","Corruption index (WGI Control of Corruption)":"Index (−2.5 to +2.5)","Working population (WB, 15–64 %)":"% of population",
    "Education (PISA scores — OECD Math mean)":"Score","Innovation (WB R&D spend %GDP)":"% GDP","GDP share (USA % of world GDP)":"% of world","Trade dominance (USA % of world exports)":"% of world",
    "Power index (CINC — USA)":"Index (0–1)","Debt burden":"USD bn"
}

# ---------- series maps ----------
FRED_MAP = {
    "Yield curve":"T10Y2Y","Consumer confidence":"UMCSENT","Building permits":"PERMIT","Unemployment claims":"ICSA",
    "LEI (Conference Board Leading Economic Index)":"USSLIND","GDP":"GDP","Capacity utilization":"TCU","Inflation":"CPIAUCSL","Retail sales":"RSXFS",
    "Nonfarm payrolls":"PAYEMS","Wage growth":"AHETPI","Credit growth":"TOTBKCR","Fed funds futures":"FEDFUNDS","Short rates":"TB3MS",
    "Industrial production":"INDPRO","Consumer/investment spending":"PCE","Productivity growth":"OPHNFB","Debt-to-GDP":"GFDEGDQ188S","Foreign reserves":"TRESEUSM193N",
    "Real rates":"DFII10","Trade balance":"BOPGSTB","Credit spreads":"BAMLH0A0HYM2","Central bank printing (M2)":"M2SL","Currency devaluation":"DTWEXBGS",
    "Fiscal deficits":"FYFSD","Debt growth":"GFDEBTN","Income growth":"A067RO1Q156NBEA","Debt service":"TDSP","Military spending":"A063RC1Q027SBEA","Debt burden":"GFDEBTN",
    "Asset prices > traditional metrics (Shiller CAPE)":"CAPE","New buyers entering (FINRA Margin Debt — FRED proxy)":"MDSP"
}
FRED_MODE = {"Inflation":"yoy","Retail sales":"yoy","Wage growth":"yoy","Credit growth":"yoy","Industrial production":"yoy","Productivity growth":"yoy","Central bank printing (M2)":"yoy","Currency devaluation":"yoy"}

WB_US = {
    "Wealth gaps (Gini, WB)":"SI.POV.GINI","Education investment (WB %GDP)":"SE.XPD.TOTL.GD.ZS","R&D patents (WB count)":"IP.PAT.RESD",
    "GDP per capita growth (WB)":"NY.GDP.PCAP.KD.ZG","Trade share (WB, Trade %GDP)":"NE.TRD.GNFS.ZS","Military spending (WB %GDP)":"MS.MIL.XPND.GD.ZS",
    "Working population (WB, 15–64 %)":"SP.POP.1564.TO.ZS","Innovation (WB R&D spend %GDP)":"GB.XPD.RSDV.GD.ZS","Corruption index (WGI Control of Corruption)":"CC.EST",
    "Internal conflicts (WGI Political Stability)":"PV.EST","Competitiveness index / Competitiveness (WEF) (WB LPI overall)":"LP.LPI.OVRL.XQ"
}
WB_USA, WB_WORLD = "USA", "WLD"

# ---------- mirror readers ----------
def load_wb_mirror(country: str, code: str):
    path = os.path.join(WB_DIR, f"{country}_{code}.csv")
    if not os.path.exists(path): return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "date" in df.columns and "val" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()

def load_fred_mirror(series_id: str) -> pd.Series:
    path = os.path.join(FRED_DIR, f"{series_id}.csv")
    if not os.path.exists(path): return pd.Series(dtype=float)
    try:
        df = pd.read_csv(path)
        # Expect columns: DATE, {series_id}
        vcol = series_id if series_id in df.columns else "VALUE"
        if vcol not in df.columns: vcol = df.columns[-1]
        s = pd.Series(df[vcol].values, index=pd.to_datetime(df["DATE"]), name=series_id)
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s
    except Exception:
        return pd.Series(dtype=float)

# ---------- data helpers (prefer mirrors) ----------
@st.cache_data(ttl=6*60*60)
def fred_series(series_id: str) -> pd.Series:
    # try mirror first
    s = load_fred_mirror(series_id)
    if not s.empty:
        return s
    # fallback to API
    s = fred.get_series(series_id)
    return s.dropna()

def yoy_from_series(s: pd.Series):
    if s.empty: return np.nan, np.nan
    last = to_float(s.iloc[-1]); ld = pd.to_datetime(s.index[-1])
    prev_idx = s.index.get_indexer([ld - timedelta(days=365)], method="nearest")[0]
    prev = to_float(s.iloc[prev_idx])
    if is_nan(prev) or prev == 0: return np.nan, np.nan
    curr = (last/prev - 1) * 100
    prev2 = np.nan
    if len(s) > 1:
        last2 = to_float(s.iloc[-2]); ld2 = pd.to_datetime(s.index[-2])
        prev2_idx = s.index.get_indexer([ld2 - timedelta(days=365)], method="nearest")[0]
        prev2_val = to_float(s.iloc[prev2_idx])
        if not is_nan(prev2_val) and prev2_val != 0:
            prev2 = (last2/prev2_val - 1) * 100
    return float(curr), (None if pd.isna(prev2) else float(prev2))

@st.cache_data(ttl=6*60*60)
def fred_last_two(series_id: str, mode: str="level"):
    s = fred_series(series_id)
    if mode == "yoy": return yoy_from_series(s)
    if s.empty: return np.nan, np.nan
    return to_float(s.iloc[-1]), (to_float(s.iloc[-2]) if len(s)>1 else np.nan)

@st.cache_data(ttl=6*60*60)
def wb_last_two(code: str, country: str):
    # mirror first
    m = load_wb_mirror(country, code)
    if not m.empty:
        return latest_prev_from_df(m.reset_index(), "val", "date")
    # fallback to API
    df = wbdata.get_dataframe({code:"val"}, country=country).dropna()
    if df.empty: return np.nan, np.nan
    df.index = pd.to_datetime(df.index); df = df.sort_index()
    return latest_prev_from_df(df, "val")

@st.cache_data(ttl=6*60*60)
def wb_share_of_world(code: str):
    # mirrors first
    usm = load_wb_mirror(WB_USA, code)
    wdm = load_wb_mirror(WB_WORLD, code)
    if not usm.empty and not wdm.empty:
        common = usm.join(wdm, lsuffix="_us", rsuffix="_w").dropna()
        if not common.empty:
            cur = to_float(common.iloc[-1]["val_us"]) / to_float(common.iloc[-1]["val_w"]) * 100
            prev = (to_float(common.iloc[-2]["val_us"]) / to_float(common.iloc[-2]["val_w"]) * 100) if len(common)>1 else np.nan
            return cur, prev
    # fallback to API
    us = wbdata.get_dataframe({code:"val"}, country=WB_USA).dropna()
    wd = wbdata.get_dataframe({code:"val"}, country=WB_WORLD).dropna()
    if us.empty or wd.empty: return np.nan, np.nan
    us.index = pd.to_datetime(us.index); wd.index = pd.to_datetime(wd.index)
    us = us.sort_index(); wd = wd.sort_index()
    common = us.join(wd, lsuffix="_us", rsuffix="_w").dropna()
    if common.empty: return np.nan, np.nan
    cur = to_float(common.iloc[-1]["val_us"]) / to_float(common.iloc[-1]["val_w"]) * 100
    prev = (to_float(common.iloc[-2]["val_us"]) / to_float(common.iloc[-2]["val_w"]) * 100) if len(common)>1 else np.nan
    return cur, prev

# ---------- UI data build ----------
rows=[]
for ind in INDICATORS:
    unit = UNITS.get(ind, ""); cur=np.nan; prev=np.nan; src="—"

    # WB mapped items
    if ind in WB_US:
        try:
            cur, prev = wb_last_two(WB_US[ind], WB_USA); src = "Mirror: WB (USA)" if not is_nan(cur) else src
        except Exception as e:
            src = f"WB error: {e}"

    # Shares vs world
    if ("GDP share" in ind or "Economic output share" in ind) and is_nan(cur):
        try:
            cur, prev = wb_share_of_world("NY.GDP.MKTP.CD"); unit = "% of world"; src = "Mirror: WB (USA/World)" if not is_nan(cur) else "World Bank (USA/World)"
        except Exception as e:
            src = f"WB share error: {e}"
    if "Trade dominance" in ind and is_nan(cur):
        try:
            cur, prev = wb_share_of_world("NE.EXP.GNFS.CD"); unit = "% of world exports"; src = "Mirror: WB (USA/World)" if not is_nan(cur) else "World Bank (USA/World)"
        except Exception as e:
            src = f"WB share error: {e}"

    # IMF COFER USD share (network only; small)
    if "Reserve currency usage dropping" in ind and is_nan(cur):
        try:
            js = SESSION.get("https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/COFER/D.USD.A.A", timeout=20).json()
            obs = js.get("CompactData",{}).get("DataSet",{}).get("Series",{}).get("Obs",[])
            if isinstance(obs, list) and obs:
                cur = to_float(obs[-1].get("@OBS_VALUE")); src = "IMF COFER (USD share)"
        except Exception:
            pass

    # FRED series (mirror first, then API)
    if is_nan(cur) and ind in FRED_MAP:
        try:
            mode = "yoy" if ind in FRED_MODE else "level"
            cur, prev = fred_last_two(FRED_MAP[ind], mode)
            src = "Mirror: FRED" if src=="—" else (src+" + Mirror FRED")
        except Exception as e:
            src = f"FRED error: {e}"

    delta = (cur - prev) if (not is_nan(cur) and not is_nan(prev)) else np.nan
    rows.append({"Indicator":ind,"Current":None if is_nan(cur) else round(cur,2),"Previous":None if is_nan(prev) else round(prev,2),"Delta":None if is_nan(delta) else round(delta,2),"Unit":unit,"Threshold":THRESHOLDS.get(ind,"—"),"Source":src})

df = pd.DataFrame(rows)
st.markdown("## 📊 Econ Mirror — Full Indicator Table (Mirrors + Nightly refresh)")
st.caption("Uses local mirrors first (WB & FRED). Nightly GitHub Action refresh keeps data current. Falls back to APIs if mirrors missing.")
st.dataframe(df, use_container_width=True, hide_index=True)
st.markdown('<div class="muted">Last refresh: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + "</div>", unsafe_allow_html=True)
