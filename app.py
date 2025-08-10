import streamlit as st, pandas as pd, numpy as np, os, requests
from datetime import datetime, timedelta
from fredapi import Fred
import wbdata

st.set_page_config(page_title="Econ Mirror — Full Table", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:2rem} .stDataFrame{border:1px solid #1f2937;border-radius:10px}</style>", unsafe_allow_html=True)

fred = Fred(api_key=st.secrets["FRED_API_KEY"])
SESSION = requests.Session(); SESSION.headers.update({"User-Agent":"EconMirror/App"})

DATA_DIR = "data"; WB_DIR = os.path.join(DATA_DIR,"wb"); FRED_DIR = os.path.join(DATA_DIR,"fred")
def to_float(x): 
    try: 
        if x is None or (isinstance(x,float) and pd.isna(x)): return np.nan
        return float(x)
    except: return np.nan
def is_seed(path): return os.path.exists(path + ".SEED")

def load_csv(path):
    try: return pd.read_csv(path)
    except: return pd.DataFrame()

def load_fred_mirror_series(series_id):
    path = os.path.join(FRED_DIR, f"{series_id}.csv")
    df = load_csv(path)
    if df.empty or "DATE" not in df.columns: return pd.Series(dtype=float)
    vcol = series_id if series_id in df.columns else (df.columns[-1] if len(df.columns)>1 else None)
    if vcol is None: return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values, index=pd.to_datetime(df["DATE"]), name=series_id)
    return s.dropna()

@st.cache_data(ttl=21600)
def fred_series(series_id): 
    s = load_fred_mirror_series(series_id)
    if not s.empty: return s
    return fred.get_series(series_id).dropna()

def yoy_from_series(s):
    if s.empty: return np.nan, np.nan
    last = to_float(s.iloc[-1]); ld = pd.to_datetime(s.index[-1])
    idx = s.index.get_indexer([ld - timedelta(days=365)], method="nearest")[0]
    base = to_float(s.iloc[idx]); 
    if pd.isna(base) or base==0: return np.nan, np.nan
    cur = (last/base - 1) * 100
    prv = np.nan
    if len(s)>1:
        last2 = to_float(s.iloc[-2]); ld2 = pd.to_datetime(s.index[-2])
        idx2 = s.index.get_indexer([ld2 - timedelta(days=365)], method="nearest")[0]
        base2 = to_float(s.iloc[idx2])
        if not pd.isna(base2) and base2!=0: prv = (last2/base2 - 1) * 100
    return float(cur), (None if pd.isna(prv) else float(prv))

@st.cache_data(ttl=21600)
def fred_last_two(series_id, mode="level"):
    s = fred_series(series_id)
    if mode=="yoy": return yoy_from_series(s)
    if s.empty: return np.nan, np.nan
    return to_float(s.iloc[-1]), (to_float(s.iloc[-2]) if len(s)>1 else np.nan)

# special mirrors
def mirror_latest_csv(path, value_col, time_col, numeric_time=False):
    df = load_csv(path)
    if df.empty or value_col not in df.columns: return (np.nan, np.nan, "—")
    if numeric_time:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna().sort_values(time_col)
    c = to_float(df.iloc[-1][value_col]); p = to_float(df.iloc[-2][value_col]) if len(df)>1 else np.nan
    src = "Pinned seed" if is_seed(path) else "Mirror"
    return c, p, src

def cofer_usd_share_latest():
    path = os.path.join(DATA_DIR,"imf_cofer_usd_share.csv")
    c,p,src = mirror_latest_csv(path,"usd_share","date",numeric_time=False)
    return c,p,("Mirror: IMF COFER " + src)

def sp500_pe_latest():
    path = os.path.join(DATA_DIR,"pe_sp500.csv")
    c,p,src = mirror_latest_csv(path,"pe","date",numeric_time=False)
    return c,p,("Mirror: P/E " + src)

# WB helpers
def wb_last_two(code, country):
    mpath = os.path.join(WB_DIR, f"{country}_{code}.csv")
    df = load_csv(mpath)
    if not df.empty and "val" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce"); df = df.dropna().sort_values("date")
        cur = to_float(df.iloc[-1]["val"]); prev = to_float(df.iloc[-2]["val"]) if len(df)>1 else np.nan
        return cur, prev, ("Mirror: WB (seed)" if is_seed(mpath) else "Mirror: WB")
    try:
        t = wbdata.get_dataframe({code:"val"}, country=country).dropna()
        if t.empty: return np.nan, np.nan, "—"
        t.index = pd.to_datetime(t.index); t = t.sort_index()
        cur = to_float(t.iloc[-1]["val"]); prev = to_float(t.iloc[-2]["val"]) if len(t)>1 else np.nan
        return cur, prev, "WB (online)"
    except Exception:
        return np.nan, np.nan, "—"

def wb_share_of_world(code):
    us = load_csv(os.path.join(WB_DIR, f"USA_{code}.csv"))
    wd = load_csv(os.path.join(WB_DIR, f"WLD_{code}.csv"))
    src = "Mirror: WB"
    if not us.empty and not wd.empty:
        us["date"] = pd.to_datetime(us["date"], errors="coerce"); wd["date"] = pd.to_datetime(wd["date"], errors="coerce")
        us = us.dropna().sort_values("date"); wd = wd.dropna().sort_values("date")
        common = pd.merge(us, wd, on="date", suffixes=("_us","_w")).dropna()
        if not common.empty:
            cur = to_float(common.iloc[-1]["val_us"])/to_float(common.iloc[-1]["val_w"])*100
            prev = (to_float(common.iloc[-2]["val_us"])/to_float(common.iloc[-2]["val_w"])*100) if len(common)>1 else np.nan
            if is_seed(os.path.join(WB_DIR, f"USA_{code}.csv")) or is_seed(os.path.join(WB_DIR, f"WLD_{code}.csv")):
                src = "Mirror: WB (seed)"
            return cur, prev, src
    try:
        us = wbdata.get_dataframe({code:"val"}, country="USA").dropna()
        wd = wbdata.get_dataframe({code:"val"}, country="WLD").dropna()
        us.index = pd.to_datetime(us.index); wd.index = pd.to_datetime(wd.index)
        common = us.join(wd, lsuffix="_us", rsuffix="_w").dropna()
        cur = to_float(common.iloc[-1]["val_us"])/to_float(common.iloc[-1]["val_w"])*100
        prev = (to_float(common.iloc[-2]["val_us"])/to_float(common.iloc[-2]["val_w"])*100) if len(common)>1 else np.nan
        return cur, prev, "WB (online)"
    except Exception:
        return np.nan, np.nan, "—"

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
    "Credit growth":"% YoY","Fed funds futures":"%","Short rates":"%","Industrial production":"% YoY","Consumer/investment spending":"USD bn","Productivity growth":"% YoY",
    "Debt-to-GDP":"% of GDP","Foreign reserves":"USD bn","Real rates":"%","Trade balance":"USD bn","Credit spreads":"bps","Central bank printing (M2)":"% YoY","Currency devaluation":"% YoY",
    "Fiscal deficits":"USD bn","Debt growth":"% YoY","Income growth":"% YoY","Debt service":"% income",
    "Education investment (WB %GDP)":"% GDP","R&D patents (WB count)":"Number","Competitiveness index / Competitiveness (WEF) (WB LPI overall)":"Index (0–5)",
    "GDP per capita growth (WB)":"% YoY","Trade share (WB, Trade %GDP)":"% of GDP","Military spending (WB %GDP)":"% GDP","Internal conflicts (WGI Political Stability)":"Index (−2.5..+2.5)",
    "Reserve currency usage dropping (IMF COFER USD share)":"% of allocated FX reserves","Military losses (UCDP Battle-related deaths — Global)":"Deaths (annual)",
    "Economic output share (USA % of world GDP)":"% of world","Corruption index (WGI Control of Corruption)":"Index (−2.5..+2.5)","Working population (WB, 15–64 %)":"% of population",
    "Education (PISA scores — OECD Math mean)":"Score","Innovation (WB R&D spend %GDP)":"% GDP","GDP share (USA % of world GDP)":"% of world","Trade dominance (USA % of world exports)":"% of world",
    "Power index (CINC — USA)":"Index (0–1)","Debt burden":"USD bn"
}
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

rows=[]
for ind in INDICATORS:
    unit = UNITS.get(ind,""); cur=np.nan; prev=np.nan; src="—"

    # WB direct
    if ind in WB_US:
        c,p,s = wb_last_two(WB_US[ind],"USA")
        if not pd.isna(c): cur,prev,src = c,p,s

    # Shares
    if ("GDP share" in ind or "Economic output share" in ind) and pd.isna(cur):
        c,p,s = wb_share_of_world("NY.GDP.MKTP.CD"); unit="% of world"
        if not pd.isna(c): cur,prev,src = c,p,s
    if "Trade dominance" in ind and pd.isna(cur):
        c,p,s = wb_share_of_world("NE.EXP.GNFS.CD"); unit="% of world exports"
        if not pd.isna(c): cur,prev,src = c,p,s

    # Special mirrors
    if ind.startswith("Education (PISA"):
        path = os.path.join(DATA_DIR,"pisa_math_usa.csv")
        c,p,s = mirror_latest_csv(path,"pisa_math_mean_usa","year",numeric_time=True)
        if not pd.isna(c): cur,prev,src = c,p,"OECD PISA — " + s
    if ind.startswith("Power index (CINC"):
        path = os.path.join(DATA_DIR,"cinc_usa.csv")
        c,p,s = mirror_latest_csv(path,"cinc_usa","year",numeric_time=True)
        if not pd.isna(c): cur,prev,src = c,p,"CINC — " + s
    if ind.startswith("Military losses (UCDP"):
        path = os.path.join(DATA_DIR,"ucdp_battle_deaths_global.csv")
        c,p,s = mirror_latest_csv(path,"ucdp_battle_deaths_global","year",numeric_time=True)
        if not pd.isna(c): cur,prev,src = c,p,"UCDP — " + s
    if ind.startswith("Reserve currency usage"):
        c,p,s = cofer_usd_share_latest()
        if not pd.isna(c): cur,prev,src = c,p,s
    if ind=="P/E ratios":
        c,p,s = sp500_pe_latest()
        if not pd.isna(c): cur,prev,src = c,p,s

    # FRED
    if pd.isna(cur) and ind in FRED_MAP:
        mode = "yoy" if ind in FRED_MODE else "level"
        try:
            c,p = fred_last_two(FRED_MAP[ind], mode)
            if not pd.isna(c): cur,prev,src = c,p,"Mirror: FRED"
        except Exception as e:
            src = f"FRED error: {e}"

    delta = (cur - prev) if (not pd.isna(cur) and not pd.isna(prev)) else np.nan
    rows.append({
        "Indicator": ind,
        "Current": None if pd.isna(cur) else round(cur, 2),
        "Previous": None if pd.isna(prev) else round(prev, 2),
        "Delta": None if pd.isna(delta) else round(delta, 2),
        "Unit": unit,
        "Threshold": THRESHOLDS.get(ind,"—"),
        "Source": src
    })

df = pd.DataFrame(rows)
st.markdown("## 📊 Econ Mirror — Full Indicator Table")
st.caption("Mirrors first (FRED/WB/OECD/CINC/UCDP/IMF). If a mirror is unreachable, a clearly‑labeled **Pinned seed** is shown and will be replaced by the nightly refresh.")
st.dataframe(df, use_container_width=True, hide_index=True)
st.caption("Last refresh: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
