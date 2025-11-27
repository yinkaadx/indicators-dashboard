import os
import io
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from fredapi import Fred


def auto_create(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content.strip() + "\n")


auto_create(
    "data/imf_cofer_usd_share.csv",
    """
date,usd_share
2023-12-31,58.4
2024-03-31,58.1
2024-06-30,57.8
2024-09-30,57.2
2025-06-30,56.3
""",
)

auto_create(
    "data/pe_sp500.csv",
    """
date,pe
2025-11-26,30.57
""",
)

auto_create(
    "data/margin_finra.csv",
    """
date,debit_bil
2025-10-31,1180
""",
)

auto_create(
    "data/spx_percent_above_200dma.csv",
    """
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
""",
)


st.set_page_config(
    page_title="ECON MIRROR — Immortal Edition",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)


FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
FMP_API_KEY = st.secrets.get("FMP_API_KEY", "")
ALPHAVANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "")

fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "EconMirror/1.0",
        "Accept": "application/json,text/html,*/*;q=0.8",
    }
)


def to_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def load_csv(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def mirror_latest_csv(path: str, value_col: str, time_col: str):
    df = load_csv(path)
    if df.empty or value_col not in df.columns or time_col not in df.columns:
        return float("nan"), float("nan"), "Mirror empty", []
    df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d", errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
    if df.empty:
        return float("nan"), float("nan"), "Mirror empty", []
    cur = float(df.iloc[-1][value_col])
    prev = float(df.iloc[-2][value_col]) if len(df) > 1 else float("nan")
    hist = df[value_col].tail(24).astype(float).tolist()
    return cur, prev, "Mirror", hist


@st.cache_data(ttl=1800, show_spinner=False)
def fred_series(series_id: str):
    if not fred:
        return None
    try:
        s = fred.get_series(series_id)
        if s is None or len(s) == 0:
            return None
        return s
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fmp_historical(symbol: str, days: int = 800) -> pd.DataFrame | None:
    if not FMP_API_KEY:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries={days}&apikey={FMP_API_KEY}"
        r = SESSION.get(url, timeout=30)
        data = r.json()
        hist = data.get("historical", [])
        if not hist:
            return None
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fmp_quote(symbol: str) -> dict | None:
    if not FMP_API_KEY:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
        r = SESSION.get(url, timeout=30)
        data = r.json()
        if not data:
            return None
        return data[0]
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_spx_history() -> pd.DataFrame | None:
    return fmp_historical("%5EGSPC")


@st.cache_data(ttl=1800, show_spinner=False)
def get_vix_history() -> pd.DataFrame | None:
    return fmp_historical("%5EVIX")


def spx_last():
    df = get_spx_history()
    if df is None or df.empty:
        return float("nan")
    return float(df["close"].iloc[-1])


def spx_ath():
    df = get_spx_history()
    if df is None or df.empty:
        return float("nan")
    return float(df["close"].max())


def spx_drawdown_pct():
    c = spx_last()
    a = spx_ath()
    if np.isnan(c) or np.isnan(a) or a == 0:
        return float("nan")
    return (c / a - 1.0) * 100.0


def spx_ytd_return():
    df = get_spx_history()
    if df is None or df.empty:
        return float("nan")
    year = datetime.utcnow().year
    dfy = df[df["date"].dt.year == year]
    if dfy.empty:
        return float("nan")
    first = float(dfy["close"].iloc[0])
    last = float(dfy["close"].iloc[-1])
    if first == 0:
        return float("nan")
    return (last / first - 1.0) * 100.0


def vix_last():
    df = get_vix_history()
    if df is None or df.empty:
        return float("nan")
    return float(df["close"].iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def gold_usd():
    q = fmp_quote("XAUUSD")
    if not q:
        cur, _, _, _ = mirror_latest_csv("data/gold_spot_usd.csv", "price", "date")
        return float(cur)
    return float(q.get("price", float("nan")))


@st.cache_data(ttl=1800, show_spinner=False)
def oil_wti():
    q = fmp_quote("WTIUSD")
    if not q:
        cur, _, _, _ = mirror_latest_csv("data/crude_oil_price.csv", "price", "date")
        return float(cur)
    return float(q.get("price", float("nan")))


@st.cache_data(ttl=1800, show_spinner=False)
def btc_usd():
    q = fmp_quote("BTCUSD")
    if not q:
        cur, _, _, _ = mirror_latest_csv("data/btc_usd.csv", "price", "date")
        return float(cur)
    return float(q.get("price", float("nan")))


@st.cache_data(ttl=1800, show_spinner=False)
def real_assets_index():
    g = gold_usd()
    o = oil_wti()
    b = btc_usd()
    vals = [g, o, b]
    if any(np.isnan(v) for v in vals):
        return float("nan")
    norm = (g / 2000.0) + (o / 90.0) + (b / 60000.0)
    return norm


@st.cache_data(ttl=1800, show_spinner=False)
def hy_spread():
    s = fred_series("BAMLH0A0HYM2")
    if s is None or len(s) == 0:
        cur, _, _, _ = mirror_latest_csv("data/hy_spread.csv", "spread_bps", "date")
        return float(cur)
    return float(s.iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def real_fed_rate():
    s_ff = fred_series("FEDFUNDS")
    s_cpi = fred_series("CPIAUCSL")
    if s_ff is None or s_cpi is None or len(s_ff) == 0 or len(s_cpi) < 13:
        cur, _, _, _ = mirror_latest_csv("data/real_fed_rate.csv", "real_rate", "date")
        return float(cur)
    ff = float(s_ff.iloc[-1])
    cpi_yoy = float(s_cpi.pct_change(12).iloc[-1] * 100.0)
    return ff - cpi_yoy


@st.cache_data(ttl=1800, show_spinner=False)
def put_call_ratio():
    s = fred_series("PC_RATIO")
    if s is None or len(s) == 0:
        cur, _, _, _ = mirror_latest_csv("data/cboe_total_pc.csv", "total_pc", "date")
        return float(cur)
    return float(s.iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def aaii_bulls():
    s = fred_series("AAIIBULL")
    if s is None or len(s) == 0:
        cur, _, _, _ = mirror_latest_csv("data/aaii_bulls.csv", "bulls_pct", "date")
        return float(cur)
    return float(s.iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def spx_breadth_pct():
    cur, _, _, _ = mirror_latest_csv(
        "data/spx_percent_above_200dma.csv", "value", "date"
    )
    return float(cur)


@st.cache_data(ttl=1800, show_spinner=False)
def cofer_usd_share_latest():
    cur, prev, _, hist = mirror_latest_csv(
        "data/imf_cofer_usd_share.csv", "usd_share", "date"
    )
    return float(cur), float(prev), hist


@st.cache_data(ttl=1800, show_spinner=False)
def margin_debt_bil():
    cur, _, _, _ = mirror_latest_csv(
        "data/margin_finra.csv", "debit_bil", "date"
    )
    return float(cur)


@st.cache_data(ttl=1800, show_spinner=False)
def spx_pe_latest():
    cur, _, _, _ = mirror_latest_csv("data/pe_sp500.csv", "pe", "date")
    return float(cur)


@st.cache_data(ttl=1800, show_spinner=False)
def fed_balance_yoy():
    s = fred_series("WALCL")
    if s is None or len(s) < 53:
        cur, _, _, _ = mirror_latest_csv(
            "data/fed_balance_yoy.csv", "yoy_pct", "date"
        )
        return float(cur)
    latest = float(s.iloc[-1])
    prev = float(s.iloc[-53])
    if prev == 0:
        return float("nan")
    return (latest / prev - 1.0) * 100.0


@st.cache_data(ttl=1800, show_spinner=False)
def sofr_spread_bps():
    s_sofr = fred_series("SOFR")
    s_ff = fred_series("FEDFUNDS")
    if s_sofr is None or s_ff is None or len(s_sofr) == 0 or len(s_ff) == 0:
        cur, _, _, _ = mirror_latest_csv(
            "data/sofr_spread.csv", "spread_bp", "date"
        )
        return float(cur)
    sofr = float(s_sofr.iloc[-1])
    ff = float(s_ff.iloc[-1])
    return (sofr - ff) * 100.0


def kill_signal_rows():
    vix_val = vix_last()
    dd = spx_drawdown_pct()
    ytd = spx_ytd_return()
    margin_bil = margin_debt_bil()
    hy = hy_spread()
    real_ff = real_fed_rate()
    pc = put_call_ratio()
    aaii_val = aaii_bulls()
    breadth = spx_breadth_pct()
    fed_yoy = fed_balance_yoy()
    sofr_bp = sofr_spread_bps()

    rows = []

    k1 = int(vix_val < 20.0) if not np.isnan(vix_val) else 0
    rows.append(
        {
            "Signal": "VIX < 20",
            "Value": f"{vix_val:.2f}",
            "KILL": "KILL" if k1 else "",
        }
    )

    k2 = int(not np.isnan(dd) and dd > -8.0)
    rows.append(
        {
            "Signal": "SPX within -8% of ATH",
            "Value": f"{dd:.2f}%",
            "KILL": "KILL" if k2 else "",
        }
    )

    k3 = int(not np.isnan(ytd) and ytd > 10.0)
    rows.append(
        {
            "Signal": "SPX YTD > 10%",
            "Value": f"{ytd:.2f}%",
            "KILL": "KILL" if k3 else "",
        }
    )

    k4 = int(not np.isnan(margin_bil) and margin_bil >= 1000.0)
    rows.append(
        {
            "Signal": "Margin Debt ≥ 1000B",
            "Value": f"{margin_bil:.1f} B",
            "KILL": "KILL" if k4 else "",
        }
    )

    k5 = int(not np.isnan(hy) and hy < 4.0)
    rows.append(
        {
            "Signal": "HY Spread < 400 bps",
            "Value": f"{hy:.2f}",
            "KILL": "KILL" if k5 else "",
        }
    )

    k6 = int(not np.isnan(real_ff) and real_ff >= 1.5)
    rows.append(
        {
            "Signal": "Real Fed Funds ≥ 1.5%",
            "Value": f"{real_ff:.2f}%",
            "KILL": "KILL" if k6 else "",
        }
    )

    k7 = int(not np.isnan(pc) and pc < 0.7)
    rows.append(
        {
            "Signal": "Put/Call < 0.70",
            "Value": f"{pc:.3f}",
            "KILL": "KILL" if k7 else "",
        }
    )

    k8 = int(not np.isnan(aaii_val) and aaii_val > 45.0)
    rows.append(
        {
            "Signal": "AAII Bulls > 45%",
            "Value": f"{aaii_val:.1f}%",
            "KILL": "KILL" if k8 else "",
        }
    )

    k9 = int(not np.isnan(breadth) and breadth < 25.0)
    rows.append(
        {
            "Signal": "% SPX above 200d < 25%",
            "Value": f"{breadth:.1f}%",
            "KILL": "KILL" if k9 else "",
        }
    )

    k10 = int(
        (not np.isnan(fed_yoy) and fed_yoy <= -5.0)
        or (not np.isnan(sofr_bp) and sofr_bp >= 50.0)
    )
    rows.append(
        {
            "Signal": "Fed BS YoY ≤ -5% OR SOFR spread ≥ 50 bp",
            "Value": f"{fed_yoy:.2f}% / {sofr_bp:.1f} bp",
            "KILL": "KILL" if k10 else "",
        }
    )

    kill_count = sum(1 for r in rows if r["KILL"] == "KILL")
    return kill_count, rows


def long_term_rows(reset_flag: bool):
    total_debt = fred_series("GFDEGDQ188S")
    debt_flag = 0
    debt_val = float("nan")
    if total_debt is not None and len(total_debt) > 0:
        debt_val = float(total_debt.iloc[-1])
        debt_flag = int(debt_val >= 110.0)

    gold_val = gold_usd()
    gold_flag = int(not np.isnan(gold_val) and gold_val >= 2500.0)

    usd_share, usd_prev, _ = cofer_usd_share_latest()
    usd_flag = int(not np.isnan(usd_share) and usd_share <= 57.0)

    real_assets_val = real_assets_index()
    real_assets_flag = int(not np.isnan(real_assets_val) and real_assets_val >= 3.0)

    gini_s = fred_series("SIPOVGINIUSA")
    gini_val = float("nan")
    gini_flag = 0
    if gini_s is not None and len(gini_s) > 0:
        gini_val = float(gini_s.iloc[-1])
        gini_flag = int(gini_val >= 0.50)

    wage_s = fred_series("LABSHPUSA156NRUG")
    wage_val = float("nan")
    wage_flag = 0
    if wage_s is not None and len(wage_s) > 0:
        wage_val = float(wage_s.iloc[-1])
        wage_flag = int(wage_val < 50.0)

    prod_s = fred_series("OPHNFB")
    prod_val = float("nan")
    prod_flag = 0
    if prod_s is not None and len(prod_s) > 0:
        prod_val = float(prod_s.iloc[-1])
        prod_flag = int(prod_val <= 0.0)

    gpr_s = fred_series("GPR")
    gpr_val = float("nan")
    gpr_flag = 0
    if gpr_s is not None and len(gpr_s) > 0:
        gpr_val = float(gpr_s.iloc[-1])
        gpr_flag = int(gpr_val >= 300.0)

    pe_val = spx_pe_latest()
    pe_flag = int(not np.isnan(pe_val) and pe_val >= 30.0)

    hy_val = hy_spread()
    hy_flag = int(not np.isnan(hy_val) and hy_val <= 4.0)

    cb_gold_flag = gold_flag
    cb_gold_desc = f"Gold spot ${gold_val:,.0f}" if not np.isnan(gold_val) else "No data"

    reset_flag_int = int(bool(reset_flag))

    rows = [
        {
            "Signal": "Total Debt / GDP high",
            "Value": f"{debt_val:.1f}" if not np.isnan(debt_val) else "No data",
            "Dark red?": "🔴" if debt_flag else "🟡",
        },
        {
            "Signal": "Gold breaking ATH",
            "Value": f"{gold_val:,.0f}" if not np.isnan(gold_val) else "No data",
            "Dark red?": "🔴" if gold_flag else "🟡",
        },
        {
            "Signal": "USD reserve share falling",
            "Value": f"{usd_share:.1f}%" if not np.isnan(usd_share) else "No data",
            "Dark red?": "🔴" if usd_flag else "🟡",
        },
        {
            "Signal": "Real assets basket elevated",
            "Value": f"{real_assets_val:.2f}" if not np.isnan(real_assets_val) else "No data",
            "Dark red?": "🔴" if real_assets_flag else "🟡",
        },
        {
            "Signal": "Gini high",
            "Value": f"{gini_val:.3f}" if not np.isnan(gini_val) else "No data",
            "Dark red?": "🔴" if gini_flag else "🟡",
        },
        {
            "Signal": "Wage share low",
            "Value": f"{wage_val:.1f}%" if not np.isnan(wage_val) else "No data",
            "Dark red?": "🔴" if wage_flag else "🟡",
        },
        {
            "Signal": "Productivity stagnation",
            "Value": f"{prod_val:.2f}" if not np.isnan(prod_val) else "No data",
            "Dark red?": "🔴" if prod_flag else "🟡",
        },
        {
            "Signal": "Geopolitical risk high",
            "Value": f"{gpr_val:.1f}" if not np.isnan(gpr_val) else "No data",
            "Dark red?": "🔴" if gpr_flag else "🟡",
        },
        {
            "Signal": "S&P P/E extreme",
            "Value": f"{pe_val:.2f}" if not np.isnan(pe_val) else "No data",
            "Dark red?": "🔴" if pe_flag else "🟡",
        },
        {
            "Signal": "HY spreads very tight",
            "Value": f"{hy_val:.2f}" if not np.isnan(hy_val) else "No data",
            "Dark red?": "🔴" if hy_flag else "🟡",
        },
        {
            "Signal": "Official reset / CB gold regime",
            "Value": cb_gold_desc + (" + RESET" if reset_flag else ""),
            "Dark red?": "🔴" if (cb_gold_flag or reset_flag_int) else "🟡",
        },
    ]

    flags = [
        debt_flag,
        gold_flag,
        usd_flag,
        real_assets_flag,
        gini_flag,
        wage_flag,
        prod_flag,
        gpr_flag,
        pe_flag,
        hy_flag,
        int(cb_gold_flag or reset_flag_int),
    ]
    dark_count = sum(flags)

    no_return_flags = [
        int(cb_gold_flag),
        int(usd_flag),
        reset_flag_int,
    ]
    no_return_count = sum(no_return_flags)

    return dark_count, no_return_count, rows


def get_reset_flag() -> bool:
    return bool(st.session_state.get("official_reset_flag", False))


def set_reset_flag(value: bool) -> None:
    st.session_state["official_reset_flag"] = bool(value)


def regime_summary(kill_count: int, near_ath: bool, dark_count: int, no_return_count: int) -> str:
    if kill_count >= 7 and near_ath:
        return "FINAL TOP → 7+ short-term kill signals while SPX is within −8% of ATH: sell 80–90% of equities over days/weeks."
    if dark_count >= 8 and no_return_count >= 2:
        return "POINT OF NO RETURN → 8+ long-term dark-red + 2+ no-return triggers: tilt 80–100% into hard assets for 5–15 years."
    if kill_count >= 4 and near_ath:
        return "LATE-STAGE MELT-UP → multiple kill signals on while price is near ATH."
    if dark_count >= 5:
        return "LATE SUPER-CYCLE → many structural lights flashing, but reset not yet locked in."
    return "MID-CYCLE → mixed signals, no extreme cluster yet."


def render_banner(kill_count: int, dark_count: int, no_return_count: int):
    spx = spx_last()
    dd = spx_drawdown_pct()
    ath = spx_ath()
    near_ath = False
    if not np.isnan(spx) and not np.isnan(ath) and ath > 0:
        near_ath = spx >= ath * 0.92

    summary = regime_summary(kill_count, near_ath, dark_count, no_return_count)

    top_html = f"""
<div style="background:#111111;border-radius:14px;padding:18px 22px;border:1px solid #333333;margin-bottom:18px;">
  <div style="font-size:20px;font-weight:700;color:#ffffff;margin-bottom:4px;">
    CURRENT REGIME:
  </div>
  <div style="font-size:16px;color:#dddddd;margin-bottom:4px;">
    <span style="color:#ff6666;font-weight:700;">{kill_count}/10</span> short-term kill •
    <span style="color:#ff9966;font-weight:700;">{dark_count}/11</span> long-term dark-red •
    <span style="color:#ffcc66;font-weight:700;">{no_return_count}/3</span> no-return
  </div>
  <div style="font-size:14px;color:#bbbbbb;margin-bottom:6px;">
    SPX: {spx:,.0f} &nbsp;|&nbsp; Drawdown: {dd:.2f}% &nbsp;|&nbsp; ATH: {ath:,.0f}
  </div>
  <div style="font-size:14px;color:#ffffff;">
    {summary}
  </div>
</div>
"""
    st.markdown(top_html, unsafe_allow_html=True)

    if kill_count >= 7 and near_ath:
        st.markdown(
            """
<div style="background:#290000;border:2px solid #ff4444;border-radius:12px;padding:16px;margin-bottom:12px;color:#ffcccc;font-weight:700;text-align:center;">
FINAL TOP ZONE: 7+ short-term kill signals while SPX is near ATH → I plan to sell 80–90% of equity risk this week.
</div>
""",
            unsafe_allow_html=True,
        )

    if dark_count >= 8 and no_return_count >= 2:
        st.markdown(
            """
<div style="background:#261300;border:2px solid #ff9900;border-radius:12px;padding:16px;margin-bottom:12px;color:#ffe6cc;font-weight:700;text-align:center;">
POINT OF NO RETURN: 8+ long-term dark-red + 2+ no-return triggers → I tilt 80–100% of long-horizon capital into hard assets for 5–15 years.
</div>
""",
            unsafe_allow_html=True,
        )


def render_core_tab():
    spx = spx_last()
    dd = spx_drawdown_pct()
    ytd = spx_ytd_return()
    vix_val = vix_last()
    gold_val = gold_usd()
    oil_val = oil_wti()
    btc_val = btc_usd()
    pe_val = spx_pe_latest()
    margin_val = margin_debt_bil()
    breadth_val = spx_breadth_pct()
    hy_val = hy_spread()
    real_ff = real_fed_rate()
    usd_share_val, usd_prev_val, _ = cofer_usd_share_latest()
    real_assets_val = real_assets_index()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("S&P 500", f"{spx:,.0f}")
        st.metric("Drawdown from ATH", f"{dd:.2f}%")
        st.metric("YTD Return", f"{ytd:.2f}%")

    with c2:
        st.metric("VIX Index", f"{vix_val:.2f}")
        st.metric("HY Spread (approx)", f"{hy_val:.2f}")
        st.metric("Real Fed Funds", f"{real_ff:.2f}%")

    with c3:
        st.metric("Gold Spot (USD/oz)", f"{gold_val:,.0f}")
        st.metric("Oil WTI", f"{oil_val:.2f}")
        st.metric("BTCUSD", f"{btc_val:,.0f}")

    with c4:
        st.metric("S&P 500 P/E", f"{pe_val:.2f}")
        st.metric("Margin Debt (B)", f"{margin_val:.1f}")
        st.metric(
            "USD Reserve Share",
            f"{usd_share_val:.1f}%" if not np.isnan(usd_share_val) else "No data",
        )

    st.markdown("---")

    tbl_rows = [
        ["SPX", f"{spx:,.0f}", "Index level", "FMP ^GSPC"],
        ["SPX Drawdown %", f"{dd:.2f}%", "From ATH", "FMP ^GSPC"],
        ["SPX YTD %", f"{ytd:.2f}%", "This calendar year", "FMP ^GSPC"],
        ["VIX", f"{vix_val:.2f}", "Volatility", "FMP ^VIX"],
        ["HY Spread", f"{hy_val:.2f}", "Approx OAS", "FRED BAMLH0A0HYM2"],
        ["Real Fed Funds", f"{real_ff:.2f}%", "FEDFUNDS − CPI YoY", "FRED"],
        ["Gold (USD/oz)", f"{gold_val:,.0f}", "Spot", "FMP XAUUSD"],
        ["Oil WTI", f"{oil_val:.2f}", "Spot", "FMP WTIUSD"],
        ["BTCUSD", f"{btc_val:,.0f}", "Spot", "FMP BTCUSD"],
        ["SPX P/E", f"{pe_val:.2f}", "Trailing PE", "Mirror pe_sp500.csv"],
        ["Margin Debt", f"{margin_val:.1f} B", "FINRA debit bal", "Mirror margin_finra.csv"],
        [
            "USD Reserve Share",
            f"{usd_share_val:.1f}%",
            "IMF COFER",
            "Mirror imf_cofer_usd_share.csv",
        ],
        [
            "% SPX above 200d",
            f"{breadth_val:.1f}%",
            "Breadth",
            "Mirror spx_percent_above_200dma.csv",
        ],
        [
            "Real Assets Index",
            f"{real_assets_val:.2f}" if not np.isnan(real_assets_val) else "No data",
            "Gold/Oil/BTC basket",
            "FMP + mirror",
        ],
    ]

    core_df = pd.DataFrame(
        tbl_rows, columns=["Indicator", "Current", "Notes", "Source"]
    )
    st.dataframe(core_df, use_container_width=True, hide_index=True)


def render_short_term_tab():
    kill_count, rows = kill_signal_rows()
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown(f"**Kill combo active:** {kill_count}/10")


def render_long_term_tab():
    current_flag = get_reset_flag()
    new_flag = st.checkbox(
        "Mark OFFICIAL RESET / SYSTEMIC CURRENCY RESET as ON",
        value=current_flag,
        help="Manual override when there is an explicit law/treaty/announcement of a currency or debt system reset.",
    )
    if new_flag != current_flag:
        set_reset_flag(new_flag)
    dark_count, no_return_count, rows = long_term_rows(get_reset_flag())
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown(
        f"**Dark red signals:** {dark_count}/11 &nbsp;&nbsp; | &nbsp;&nbsp; **No-return triggers:** {no_return_count}/3",
        unsafe_allow_html=True,
    )


st.markdown(
    "<h1 style='text-align:center;font-size:40px;'>ECON MIRROR — Immortal Edition</h1>",
    unsafe_allow_html=True,
)

reset_flag = get_reset_flag()
kill_count, kill_rows = kill_signal_rows()
dark_count, no_return_count, _rows_long = long_term_rows(reset_flag)

render_banner(kill_count, dark_count, no_return_count)

tab_core, tab_short, tab_long = st.tabs(
    ["Core Mirror", "Short-Term Kill Combo", "Long-Term Super-Cycle"]
)

with tab_core:
    render_core_tab()

with tab_short:
    render_short_term_tab()

with tab_long:
    render_long_term_tab()

st.caption(
    "Econ Mirror — short-term kill combo + long-term super-cycle map, wired to official data and mirrored CSVs."
)
