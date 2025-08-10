import os
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import wbdata
from fredapi import Fred


# --------------------------- Page ---------------------------
st.set_page_config(
    page_title="Econ Mirror — Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding-top:1rem; padding-bottom:2rem;}
.stDataFrame {border:1px solid #1f2937;border-radius:10px}
.sticky-bar {position: sticky; top: 0; background: #0B1220; padding: .4rem 0 .6rem; z-index: 999;}
.badge {display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; margin-left:6px; background:#192742; color:#9fb7ff;}
.seed {background:#3b2a00; color:#ffd27a;}
.ok {color:#7CFFA2; font-weight:600;}
.warn {color:#FFAA7C; font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sticky-bar'><h2>📊 Econ Mirror — Full Indicator Table</h2></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Mirrors first (FRED/WB/OECD/CINC/UCDP/IMF). If a live source fails, a clearly‑"
    "labeled **Pinned seed** appears and is replaced by the nightly refresh."
)

# --------------------------- Infra ---------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/App"})

DATA_DIR = "data"
WB_DIR = os.path.join(DATA_DIR, "wb")
FRED_DIR = os.path.join(DATA_DIR, "fred")

fred_api_key = str(st.secrets.get("FRED_API_KEY", ""))
fred = Fred(api_key=fred_api_key)


def to_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, float) and pd.isna(x):
            return float("nan")
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

    if series_id in df.columns:
        vcol = series_id
    else:
        vcol = df.columns[-1] if len(df.columns) > 1 else None

    if vcol is None:
        return pd.Series(dtype=float)

    s = pd.Series(
        pd.to_numeric(df[vcol], errors="coerce").values,
        index=pd.to_datetime(df["DATE"]),
        name=series_id,
    )
    return s.dropna()


@st.cache_data(ttl=21600)
def fred_series(series_id: str) -> pd.Series:
    s = load_fred_mirror_series(series_id)
    if not s.empty:
        return s
    return fred.get_series(series_id).dropna()


def yoy_from_series(s: pd.Series) -> Tuple[float, float]:
    if s.empty:
        return float("nan"), float("nan")

    last = to_float(s.iloc[-1])
    ld = pd.to_datetime(s.index[-1])

    idx = s.index.get_indexer([ld - timedelta(days=365)], method="nearest")[0]
    base = to_float(s.iloc[idx])
    if pd.isna(base) or base == 0:
        return float("nan"), float("nan")

    cur = (last / base - 1) * 100.0
    prv = float("nan")
    if len(s) > 1:
        last2 = to_float(s.iloc[-2])
        ld2 = pd.to_datetime(s.index[-2])
        idx2 = s.index.get_indexer([ld2 - timedelta(days=365)], method="nearest")[0]
        base2 = to_float(s.iloc[idx2])
        if not pd.isna(base2) and base2 != 0:
            prv = (last2 / base2 - 1) * 100.0

    return float(cur), (float("nan") if pd.isna(prv) else float(prv))


@st.cache_data(ttl=21600)
def fred_last_two(series_id: str, mode: str = "level") -> Tuple[float, float]:
    s = fred_series(series_id)
    if mode == "yoy":
        return yoy_from_series(s)
    if s.empty:
        return float("nan"), float("nan")
    cur = to_float(s.iloc[-1])
    prev = to_float(s.iloc[-2]) if len(s) > 1 else float("nan")
    return cur, prev


def fred_history(series_id: str, mode: str = "level", n: int = 24) -> List[float]:
    s = fred_series(series_id)
    if s.empty:
        return []

    if mode == "yoy":
        vals: List[float] = []
        max_len = min(len(s), n * 2)
        for i in range(max_len):
            j = len(s) - i - 1
            if j <= 0:
                break
            ld = s.index[j]
            idx = s.index.get_indexer([ld - timedelta(days=365)], method="nearest")[0]
            base = to_float(s.iloc[idx])
            val = to_float(s.iloc[j])
            if not pd.isna(base) and base != 0:
                vals.append((val / base - 1) * 100.0)
        vals = [v for v in vals if v is not None]
        return list(reversed(vals))[:n]

    return list(s.tail(n).astype(float).values)


@st.cache_data(ttl=21600)
def wb_last_two(code: str, country: str) -> Tuple[float, float, str, List[float]]:
    mpath = os.path.join(WB_DIR, f"{country}_{code}.csv")
    df = load_csv(mpath)

    if not df.empty and "val" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna().sort_values("date")
        cur = to_float(df.iloc[-1]["val"])
        prev = to_float(df.iloc[-2]["val"]) if len(df) > 1 else float("nan")
        src = "Mirror: WB (seed)" if is_seed(mpath) else "Mirror: WB"
        hist = df["val"].tail(24).astype(float).tolist()
        return cur, prev, src, hist

    try:
        t = wbdata.get_dataframe({code: "val"}, country=country).dropna()
        if t.empty:
            return float("nan"), float("nan"), "—", []
        t.index = pd.to_datetime(t.index)
        t = t.sort_index()
        cur = to_float(t.iloc[-1]["val"])
        prev = to_float(t.iloc[-2]["val"]) if len(t) > 1 else float("nan")
        hist = t["val"].tail(24).astype(float).tolist()
        return cur, prev, "WB (online)", hist
    except Exception:
        return float("nan"), float("nan"), "—", []


@st.cache_data(ttl=21600)
def wb_share_series(code: str) -> Tuple[pd.DataFrame, str]:
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
            ) * 100.0
            src = (
                "Mirror: WB (seed)"
                if (
                    is_seed(os.path.join(WB_DIR, f"USA_{code}.csv"))
                    or is_seed(os.path.join(WB_DIR, f"WLD_{code}.csv"))
                )
                else "Mirror: WB"
            )
            return df[["date", "share"]].dropna(), src

    try:
        us_on = wbdata.get_dataframe({code: "us"}, country="USA").dropna()
        wd_on = wbdata.get_dataframe({code: "w"}, country="WLD").dropna()
        us_on.index = pd.to_datetime(us_on.index)
        wd_on.index = pd.to_datetime(wd_on.index)
        df = us_on.join(wd_on, how="inner").dropna()
        df["share"] = (
            pd.to_numeric(df["us"], errors="coerce")
            / pd.to_numeric(df["w"], errors="coerce")
        ) * 100.0
        out = df.reset_index().rename(columns={"index": "date"})[["date", "share"]]
        return out, "WB (online)"
    except Exception:
        return pd.DataFrame(), "—"


def mirror_latest_csv(
    path: str, value_col: str, time_col: str, numeric_time: bool = False
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
    hist = df[value_col].tail(24).astype(float).tolist()
    return cur, prev, src, hist


def cofer_usd_share_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "imf_cofer_usd_share.csv")
    cur, prev, src, hist = mirror_latest_csv(
        path, "usd_share", "date", numeric_time=False
    )
    return cur, prev, ("IMF COFER — " + src), hist


def sp500_pe_latest() -> Tuple[float, float, str, List[float]]:
    path = os.path.join(DATA_DIR, "pe_sp500.csv")
    cur, prev, src, hist = mirror_latest_csv(path, "pe", "date", numeric_time=False)
    return cur, prev, ("SP500 P/E — " + src), hist


# --------------------------- Indicators ---------------------------
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
    "Education (PISA scores — OECD Math mean)": " > 500 (top)",
    "Innovation (WB R&D spend %GDP)": "Patents > 20% of global (high)",
    "GDP share (USA % of world GDP)": "+2% of global share (growing)",
    "Trade dominance (USA % of world exports)": "> 15% of global trade (dominance)",
    "Power index (CINC — USA)": "Index rising (power)",
    "Debt burden": "> 100% of GDP (high)",
}

UNITS: Dict[str, str] = {
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
    "Fed funds futures": "%",
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
    "Currency devaluation": "% YoY",
    "Fiscal deficits": "USD bn",
    "Debt growth": "% YoY",
    "Income growth": "% YoY",
    "Debt service": "% income",
    "Education investment (WB %GDP)": "% GDP",
    "R&D patents (WB count)": "Number",
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)": "Index (0–5)",
    "GDP per capita growth (WB)": "% YoY",
    "Trade share (WB, Trade %GDP)": "% of GDP",
    "Military spending (WB %GDP)": "% GDP",
    "Internal conflicts (WGI Political Stability)": "Index (−2.5..+2.5)",
    "Reserve currency usage dropping (IMF COFER USD share)": "% of allocated FX reserves",
    "Military losses (UCDP Battle-related deaths — Global)": "Deaths (annual)",
    "Economic output share (USA % of world GDP)": "% of world",
    "Corruption index (WGI Control of Corruption)": "Index (−2.5..+2.5)",
    "Working population (WB, 15–64 %)": "% of population",
    "Education (PISA scores — OECD Math mean)": "Score",
    "Innovation (WB R&D spend %GDP)": "% GDP",
    "GDP share (USA % of world GDP)": "% of world",
    "Trade dominance (USA % of world exports)": "% of world",
    "Power index (CINC — USA)": "Index (0–1)",
    "Debt burden": "USD bn",
}

FRED_MAP: Dict[str, str] = {
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
    "Fed funds futures": "FEDFUNDS",
    "Short rates": "TB3MS",
    "Industrial production": "INDPRO",
    "Consumer/investment spending": "PCE",
    "Productivity growth": "OPHNFB",
    "Debt-to-GDP": "GFDEGDQ188S",
    "Foreign reserves": "TRESEUSM193N",
    "Real rates": "DFII10",
    "Trade balance": "BOPGSTB",
    "Credit spreads": "BAMLH0A0HYM2",
    "Central bank printing (M2)": "M2SL",
    "Currency devaluation": "DTWEXBGS",
    "Fiscal deficits": "FYFSD",
    "Debt growth": "GFDEBTN",
    "Income growth": "A067RO1Q156NBEA",
    "Debt service": "TDSP",
    "Military spending": "A063RC1Q027SBEA",
    "Debt burden": "GFDEBTN",
    "Asset prices > traditional metrics (Shiller CAPE)": "CAPE",
    "New buyers entering (FINRA Margin Debt — FRED proxy)": "MDSP",
}

FRED_MODE: Dict[str, str] = {
    "Inflation": "yoy",
    "Retail sales": "yoy",
    "Wage growth": "yoy",
    "Credit growth": "yoy",
    "Industrial production": "yoy",
    "Productivity growth": "yoy",
    "Central bank printing (M2)": "yoy",
    "Currency devaluation": "yoy",
}

WB_US: Dict[str, str] = {
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
    "Competitiveness index / Competitiveness (WEF) (WB LPI overall)": "LP.LPI.OVRL.XQ",
}


def parse_simple_threshold(txt: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(txt, str):
        return None, None
    m = re.search(r"([<>]=?)\s*([+-]?\d+(?:\.\d+)?)", txt.replace("−", "-"))
    if not m:
        return None, None
    comp = m.group(1)
    num = float(m.group(2))
    return comp, num


def evaluate_signal(current: float, threshold_text: Optional[str]) -> Tuple[str, str]:
    comp, val = parse_simple_threshold(threshold_text)
    if comp is None or pd.isna(current) or val is None:
        return "—", ""
    ok = (current > val) if ">" in comp else (current < val)
    return ("✅", "ok") if ok else ("⚠️", "warn")


# --------------------------- Build table ---------------------------
rows: List[Dict[str, object]] = []
histories: List[List[float]] = []

for ind in INDICATORS:
    unit = UNITS.get(ind, "")
    cur = float("nan")
    prev = float("nan")
    src = "—"
    hist: List[float] = []

    # WB direct
    if ind in WB_US:
        c, p, s, h = wb_last_two(WB_US[ind], "USA")
        if not pd.isna(c):
            cur, prev, src, hist = c, p, s, h

    # Shares
    if ("GDP share" in ind or "Economic output share" in ind) and pd.isna(cur):
        series, ssrc = wb_share_series("NY.GDP.MKTP.CD")
        if not series.empty:
            cur = to_float(series.iloc[-1]["share"])
            prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float(
                "nan"
            )
            unit = "% of world"
            src = ssrc
            hist = series["share"].tail(24).astype(float).tolist()

    if "Trade dominance" in ind and pd.isna(cur):
        series, ssrc = wb_share_series("NE.EXP.GNFS.CD")
        if not series.empty:
            cur = to_float(series.iloc[-1]["share"])
            prev = to_float(series.iloc[-2]["share"]) if len(series) > 1 else float(
                "nan"
            )
            unit = "% of world exports"
            src = ssrc
            hist = series["share"].tail(24).astype(float).tolist()

    # Special mirrors
    if ind.startswith("Education (PISA"):
        path_pisa = os.path.join(DATA_DIR, "pisa_math_usa.csv")
        c, p, s, h = mirror_latest_csv(
            path_pisa, "pisa_math_mean_usa", "year", numeric_time=True
        )
        if not pd.isna(c):
            cur, prev, src, hist = c, p, "OECD PISA — " + s, h

    if ind.startswith("Power index (CINC"):
        path_cinc = os.path.join(DATA_DIR, "cinc_usa.csv")
        c, p, s, h = mirror_latest_csv(
            path_cinc, "cinc_usa", "year", numeric_time=True
        )
        if not pd.isna(c):
            cur, prev, src, hist = c, p, "CINC — " + s, h

    if ind.startswith("Military losses (UCDP"):
        path_ucdp = os.path.join(DATA_DIR, "ucdp_battle_deaths_global.csv")
        c, p, s, h = mirror_latest_csv(
            path_ucdp, "ucdp_battle_deaths_global", "year", numeric_time=True
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

    # FRED fallback
    if pd.isna(cur) and ind in FRED_MAP:
        mode = "yoy" if ind in FRED_MODE else "level"
        try:
            c, p = fred_last_two(FRED_MAP[ind], mode)
            if not pd.isna(c):
                cur = c
                prev = p
                src = "Mirror: FRED"
                hist = fred_history(FRED_MAP[ind], mode, n=24)
        except Exception as e:
            src = f"FRED error: {e}"

    delta = (
        cur - prev if (not pd.isna(cur) and not pd.isna(prev)) else float("nan")
    )
    signal_icon, _signal_cls = evaluate_signal(cur, THRESHOLDS.get(ind, "—"))
    if "Pinned seed" in src:
        _seed_badge = "Pinned seed"  # retained for potential future UI tags

    rows.append(
        {
            "Indicator": ind,
            "Current": None if pd.isna(cur) else round(cur, 2),
            "Previous": None if pd.isna(prev) else round(prev, 2),
            "Delta": None if pd.isna(delta) else round(delta, 2),
            "Unit": unit,
            "Threshold": THRESHOLDS.get(ind, "—"),
            "Signal": f"{signal_icon}",
            "Source": src,
        }
    )
    histories.append(hist)

df = pd.DataFrame(rows)
df["History"] = histories

# --------------------------- Toolbar ---------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    q = st.text_input(
        "Quick search",
        "",
        placeholder="Filter indicators, e.g., 'inflation' or 'trade'",
    ).strip().lower()
with c2:
    show_only_supported = st.checkbox("Only rows with Signal", value=False)
with c3:
    csv_bytes = df.drop(columns=["History"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="econ_mirror_table.csv",
        mime="text/csv",
        use_container_width=True,
    )

vdf = df.copy()
if q:
    vdf = vdf[
        vdf["Indicator"].str.lower().str.contains(q)
        | vdf["Source"].str.lower().str.contains(q)
    ]
if show_only_supported:
    vdf = vdf[vdf["Signal"].isin(["✅", "⚠️"])]

col_config = {
    "Indicator": st.column_config.TextColumn("Indicator"),
    "Current": st.column_config.NumberColumn("Current"),
    "Previous": st.column_config.NumberColumn("Previous"),
    "Delta": st.column_config.NumberColumn("Δ"),
    "Unit": st.column_config.TextColumn("Unit"),
    "Threshold": st.column_config.TextColumn("Threshold"),
    "Signal": st.column_config.TextColumn("Signal"),
    "Source": st.column_config.TextColumn("Source"),
    "History": st.column_config.LineChartColumn("History", width="medium"),
}

st.dataframe(vdf, use_container_width=True, hide_index=True, column_config=col_config)
