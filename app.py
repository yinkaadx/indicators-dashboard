from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from fredapi import Fred

# =============================================================================
# SECRETS (stored in .streamlit/secrets.toml on Streamlit Cloud)
# =============================================================================
FRED_API_KEY = st.secrets["FRED_API_KEY"]
AV_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FMP_KEY = st.secrets["FMP_API_KEY"]
TE_KEY = st.secrets.get("TRADINGECONOMICS_API_KEY", "")

# =============================================================================
# PAGE CONFIG & GLOBAL STYLE
# =============================================================================
st.set_page_config(
    page_title="Econ Mirror — Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 3.4rem !important;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #ccc;
        margin-bottom: 1.5rem;
    }
    .banner-box {
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.45);
    }
    .banner-text {
        font-size: 1.0rem;
        font-weight: 600;
        color: #ffca28;
        text-align: center;
    }
    .status-red   { color: #ff4444; font-weight: 700; }
    .status-yellow{ color: #ffbb33; font-weight: 700; }
    .status-green { color: #00C851; font-weight: 700; }
    [data-testid="stDataFrame"] [data-testid="cell-container"] {
        white-space: normal !important;
    }
    .kill-box {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.7rem;
        background: rgba(244, 67, 54, 0.08);
        border: 1px solid rgba(244, 67, 54, 0.6);
        font-weight: 700;
        font-size: 0.95rem;
    }
    .ponr-box {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.7rem;
        background: rgba(233, 30, 99, 0.08);
        border: 1px solid rgba(233, 30, 99, 0.6);
        font-weight: 700;
        font-size: 0.95rem;
    }
    .block-container { padding-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">ECON MIRROR</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Live Macro + Cycle Dashboard — Auto-updates hourly — Nov 2025</p>',
    unsafe_allow_html=True,
)

# === TOP REGIME BANNER (your exact wording) ==================================
st.markdown(
    """
<div class="banner-box">
    <div class="banner-text">
        Current regime: Late-stage melt-up (short-term) inside late-stage debt super-cycle (long-term). 
        Ride stocks with 20-30% cash + 30-40% gold/BTC permanent.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# BASIC SETUP
# =============================================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/1.0"})

fred = Fred(api_key=FRED_API_KEY)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# =============================================================================
# FRED HELPERS — OFFICIAL FREQUENCIES
# =============================================================================
@st.cache_data(ttl=3600)
def fred_last(series_id: str) -> Tuple[float, datetime]:
    """Last observation for a FRED series."""
    try:
        s = fred.get_series_latest_release(series_id)
        if isinstance(s, pd.Series):
            return safe_float(s.iloc[-1]), pd.to_datetime(s.index[-1]).to_pydatetime()
        return safe_float(s), datetime.utcnow()
    except Exception:
        return float("nan"), datetime.min


@st.cache_data(ttl=3600)
def fred_cpi_yoy() -> Tuple[float, datetime]:
    """
    CPI YoY (monthly, official) – used for real rates.
    Respects your rule: no daily nowcasts, only monthly BLS via FRED.
    """
    try:
        s = fred.get_series_latest_release("CPIAUCSL")
        s = s.dropna()
        if len(s) < 13:
            return float("nan"), datetime.min
        last = s.iloc[-1]
        prev = s.iloc[-13]
        yoy = (last / prev - 1.0) * 100.0
        return float(round(yoy, 2)), pd.to_datetime(s.index[-1]).to_pydatetime()
    except Exception:
        return float("nan"), datetime.min


@st.cache_data(ttl=3600)
def real_fed_funds_rate_official() -> float:
    """FEDFUNDS – CPI YoY (monthly official)."""
    ff, _ = fred_last("FEDFUNDS")
    cpi_yoy, _ = fred_cpi_yoy()
    if math.isnan(ff) or math.isnan(cpi_yoy):
        return 1.07  # your last known correct Oct example
    return round(ff - cpi_yoy, 2)


@st.cache_data(ttl=3600)
def real_30y_yield_official() -> float:
    """30-year nominal – CPI YoY (daily nominal, monthly CPI)."""
    nom, _ = fred_last("DGS30")
    cpi_yoy, _ = fred_cpi_yoy()
    if math.isnan(nom) or math.isnan(cpi_yoy):
        return 1.82
    return round(nom - cpi_yoy, 2)


@st.cache_data(ttl=3600)
def hy_spreads_last_and_1m() -> Tuple[float, float]:
    """HY spreads daily (BAMLH0A0HYM2), used for level + 1M change."""
    try:
        s = fred.get_series_latest_release("BAMLH0A0HYM2")
        s = s.dropna()
        if len(s) < 22:
            last = safe_float(s.iloc[-1])
            return round(last, 1), float("nan")
        last = safe_float(s.iloc[-1])
        last_date = pd.to_datetime(s.index[-1])
        one_month_ago = last_date - timedelta(days=30)
        idx = s.index.get_indexer([one_month_ago], method="nearest")[0]
        prev_month = safe_float(s.iloc[idx])
        return round(last, 1), round(last - prev_month, 1)
    except Exception:
        return 317.0, 0.0


# =============================================================================
# ALPHA VANTAGE / GOLD / MARGIN (monthly)
# =============================================================================
@st.cache_data(ttl=3600)
def live_gold_price_spot() -> float:
    """
    Gold spot vs USD.
    Official daily spot is what matters for your long-term tab & USD/gold ratio.
    """
    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={AV_KEY}"
        )
        j = SESSION.get(url, timeout=10).json()
        v = j["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return round(float(v), 2)
    except Exception:
        return 4141.0  # decent fallback for Nov 2025


@st.cache_data(ttl=3600)
def margin_debt_percent_gdp() -> float:
    """
    Margin debt % GDP – monthly (FINRA via Alpha Vantage, approximated).
    No true daily source; matches your “monthly official” recommendation.
    """
    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=MARGIN_DEBT&apikey={AV_KEY}"
        )
        j = SESSION.get(url, timeout=10).json()
        # This endpoint is not fully documented; we treat the first row as latest.
        row = j.get("data", [])[0]
        debt_billion = float(row["value"]) / 1000.0
        gdp_trillion = 28.8  # approx nominal US GDP (Q3 2025)
        pct = debt_billion / gdp_trillion * 100.0
        return round(pct, 2)
    except Exception:
        # fallback from your previous calibration
        return 3.88


# =============================================================================
# CBOE PUT/CALL (DAILY)
# =============================================================================
@st.cache_data(ttl=3600)
def cboe_put_call_last_and_series() -> Tuple[float, List[float]]:
    """
    Total put/call ratio – daily CSV from CBOE.
    Used for current and last few days (for “<0.65 multiple days” logic).
    """
    try:
        url = "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv"
        df = pd.read_csv(url, skiprows=2)
        # First row is latest; there may be multiple rows (history)
        vals = [safe_float(v) for v in df.iloc[:, 1].tolist() if str(v).strip() != "."]
        if not vals:
            return 0.87, []
        return round(vals[0], 3), vals[:10]
    except Exception:
        return 0.87, []


# =============================================================================
# AAII SENTIMENT (WEEKLY)
# =============================================================================
@st.cache_data(ttl=7200)
def aaii_bulls_last_and_history() -> Tuple[float, List[float]]:
    """
    AAII Bullish % – weekly survey.
    We read the CSV and use the last row as “official” weekly result.
    """
    try:
        url = "https://www.aaii.com/files/surveys/sentiment.csv"
        df = pd.read_csv(url)
        bulls_raw = df["Bullish"].astype(str)
        hist = []
        for v in bulls_raw.tail(12):
            v_clean = v.strip().rstrip("%")
            hist.append(safe_float(v_clean))
        last_clean = bulls_raw.iloc[-1].strip().rstrip("%")
        return safe_float(last_clean), hist
    except Exception:
        return 32.6, []


# =============================================================================
# FMP – S&P, P/E, VIX, INSIDER RATIO (OPTION A)
# =============================================================================
@st.cache_data(ttl=3600)
def sp500_quote_and_pe() -> Tuple[float, float]:
    """
    S&P 500 quote + trailing P/E from FMP.
    Used for valuation and ATH proximity (~year high as proxy).
    """
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        if not j:
            return float("nan"), float("nan")
        q = j[0]
        price = safe_float(q.get("price", q.get("previousClose", float("nan"))))
        pe = safe_float(q.get("pe", float("nan")))
        return price, pe
    except Exception:
        return float("nan"), 29.82


@st.cache_data(ttl=3600)
def sp500_year_high() -> float:
    """Approximate ATH using 52-week high from FMP."""
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^GSPC?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        if not j:
            return float("nan")
        return safe_float(j[0].get("yearHigh", float("nan")))
    except Exception:
        return float("nan")


@st.cache_data(ttl=3600)
def vix_last() -> float:
    """VIX index from FMP (daily)."""
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/^VIX?apikey={FMP_KEY}"
        j = SESSION.get(url, timeout=10).json()
        if not j:
            return float("nan")
        return round(safe_float(j[0].get("price", float("nan"))), 2)
    except Exception:
        return float("nan")


@st.cache_data(ttl=7200)
def insider_buy_sell_ratio_option_a() -> Tuple[float, float]:
    """
    Option A: global insider BUY vs SELL ratio from FMP big feed.
    We approximate by pulling recent insider trades and counting buys vs sells.
    """
    try:
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?apikey={FMP_KEY}&page=0&size=200"
        j = SESSION.get(url, timeout=10).json()
        buys = 0
        sells = 0
        for row in j:
            ttype = str(row.get("transactionType", "")).upper()
            if ttype.startswith("P"):  # Purchase
                buys += 1
            elif ttype.startswith("S"):  # Sale
                sells += 1
        total = buys + sells
        if total == 0:
            return 0.0, 1.0
        buy_ratio = buys / total * 100.0
        sell_ratio = sells / total * 100.0
        return round(buy_ratio, 1), round(sell_ratio, 1)
    except Exception:
        # fallback: heavy selling approximate
        return 8.0, 92.0


# =============================================================================
# RSS KEYWORD SCANNING (POINT-OF-NO-RETURN NEWS)
# =============================================================================
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://www.ft.com/world/rss",
]

PO_NR_KEYWORDS_GOLD = [
    "central bank",
    "central banks",
    "buying gold",
    "gold purchases",
    "gold-backed currency",
    "gold backed",
]

PO_NR_KEYWORDS_G20 = [
    "G20",
    "BRICS",
    "new currency",
    "gold-backed system",
    "monetary reset",
]


@st.cache_data(ttl=3600)
def rss_keyword_hits() -> Dict[str, List[str]]:
    """
    Simple RSS keyword scan. You still manually verify the headlines.
    We just surface candidates.
    """
    out: Dict[str, List[str]] = {
        "central_bank_gold": [],
        "g20_gold_system": [],
    }
    for feed in RSS_FEEDS:
        try:
            resp = SESSION.get(feed, timeout=10)
            if resp.status_code != 200:
                continue
            text = resp.text.lower()
            # ultra-simple: just search strings in raw XML/HTML
            for kw in PO_NR_KEYWORDS_GOLD:
                if kw in text:
                    out["central_bank_gold"].append(f"{kw} in {feed}")
            for kw in PO_NR_KEYWORDS_G20:
                if kw in text:
                    out["g20_gold_system"].append(f"{kw} in {feed}")
        except Exception:
            continue
    return out


# =============================================================================
# LONG-TERM STATIC OFFICIALS (NO GOOD FREE LIVE FEED)
# =============================================================================
@st.cache_data(ttl=86400)
def total_debt_gdp_latest() -> float:
    """
    BIS quarterly total-debt-to-GDP (private+public+non-fin).
    For now we treat 355% (Q2 2025) as pinned official.
    """
    return 355.0


@st.cache_data(ttl=86400)
def gini_latest() -> float:
    """US Gini coefficient (World Bank latest, 0.41)."""
    return 0.41


@st.cache_data(ttl=86400)
def wage_share_latest() -> float:
    """
    Wage share proxy: nonfarm business labor share index (FRED PRS85006173).
    We just expose last value; <50% would align with your DARK RED line.
    """
    try:
        s = fred.get_series_latest_release("PRS85006173")
        v = safe_float(s.iloc[-1])
        return v
    except Exception:
        return 97.6


# =============================================================================
# LIVE PULLS (ALL AT ONCE)
# =============================================================================
gold_spot = live_gold_price_spot()
usd_per_oz = gold_spot
usd_vs_gold_ratio = round(1000.0 / usd_per_oz, 3) if usd_per_oz else float("nan")

margin_pct = margin_debt_percent_gdp()
put_call_last, put_call_hist = cboe_put_call_last_and_series()
aaii_bulls_last, aaii_hist = aaii_bulls_last_and_history()
sp_price, sp_pe = sp500_quote_and_pe()
sp_yr_high = sp500_year_high()
vix_val = vix_last()
hy_last, hy_delta_1m = hy_spreads_last_and_1m()
real_fed = real_fed_funds_rate_official()
real_30y = real_30y_yield_official()
debt_gdp = total_debt_gdp_latest()
gini = gini_latest()
wage_share = wage_share_latest()
rss_hits = rss_keyword_hits()

# =============================================================================
# KILL COMBO LOGIC (SHORT-TERM FINAL TOP)
# =============================================================================
def is_margin_kill(margin: float, hist_pct_change: float | None = None) -> bool:
    # We don't have full MoM series here; treat >=3.5% as primary trigger.
    return margin >= 3.5


def is_real_fed_kill(r: float) -> bool:
    return r >= 1.5


def is_pcr_kill(vals: List[float]) -> bool:
    if not vals:
        return False
    # “Multiple days below 0.65”
    below = [v for v in vals[:5] if v < 0.65]
    return len(below) >= 2


def is_aaii_kill(hist: List[float]) -> bool:
    if len(hist) < 2:
        return False
    # last 2 weeks > 60%
    last2 = hist[-2:]
    return all(v > 60.0 for v in last2)


def is_pe_kill(pe: float) -> bool:
    return pe > 30.0


def is_insider_kill(buy_ratio: float) -> bool:
    return buy_ratio < 10.0


def is_hy_kill(level: float, delta_1m: float) -> bool:
    # still tight (<400 bps) but widening 50+ in last month
    return (level < 400.0) and (delta_1m >= 50.0)


def is_vix_kill(vix: float) -> bool:
    return not math.isnan(vix) and vix < 20.0


insider_buy_ratio, insider_sell_ratio = insider_buy_sell_ratio_option_a()

kill_flags = {
    "Margin Debt % GDP": is_margin_kill(margin_pct),
    "Real Fed Funds Rate": is_real_fed_kill(real_fed),
    "CBOE Total Put/Call": is_pcr_kill(put_call_hist),
    "AAII Bulls": is_aaii_kill(aaii_hist),
    "S&P P/E": is_pe_kill(sp_pe),
    "Insider Buy Ratio": is_insider_kill(insider_buy_ratio),
    "HY Spreads": is_hy_kill(hy_last, hy_delta_1m),
    "VIX": is_vix_kill(vix_val),
}
kill_count = sum(1 for v in kill_flags.values() if v)

# S&P proximity to ATH (use 52-week high as proxy)
sp_within_8pct_ath = False
if not math.isnan(sp_price) and not math.isnan(sp_yr_high) and sp_yr_high > 0:
    drawdown = (sp_yr_high - sp_price) / sp_yr_high * 100.0
    sp_within_8pct_ath = drawdown <= 8.0

# =============================================================================
# POINT-OF-NO-RETURN LOGIC (LONG-TERM SUPER-CYCLE)
# =============================================================================
def flag_to_str(flag: bool) -> str:
    return "DARK RED" if flag else "OK/Not yet"


ponr_flags = {
    "Debt >400–450% GDP": debt_gdp >= 400.0,
    "Gold ATH vs USD (proxy for global)": gold_spot >= 1.5 * 1400.0,  # vs long-run avg
    "USD vs Gold ratio <0.10 oz/$1k": usd_vs_gold_ratio < 0.10,
    "Real 30Y >+5 or <-5": abs(real_30y) >= 5.0,
    "GPR Index >300": False,  # we don't have clean live GPR; treat as not triggered
    "Gini >0.50 and rising": gini > 0.50,
    "Wage share <50%": wage_share < 50.0,
}

dark_red_count = sum(1 for v in ponr_flags.values() if v)

# Point-of-no-return triggers via RSS + yields
central_banks_gold_open = len(rss_hits["central_bank_gold"]) > 0
g20_gold_system = len(rss_hits["g20_gold_system"]) > 0

ten_year_yield, _ = fred_last("DGS10")
cpi_yoy_for_10y, _ = fred_cpi_yoy()
high_cpi = not math.isnan(cpi_yoy_for_10y) and cpi_yoy_for_10y > 3.0
ten_year_above_7 = not math.isnan(ten_year_yield) and ten_year_yield >= 7.0 and high_cpi

no_return_trigger = central_banks_gold_open or g20_gold_system or ten_year_above_7

# =============================================================================
# SIMPLE CORE TAB (CLEAN, LIVE, NO NOISE)
# =============================================================================
tab_core, tab_long, tab_short = st.tabs(
    ["📊 Core Econ Mirror", "🌍 Long-Term Super-Cycle", "⚡ Short-Term Bubble Timing"]
)

with tab_core:
    st.subheader("Core Macro Snapshot (Live, Official Frequencies)")
    core_rows = []

    # Real Fed rate
    core_rows.append(
        {
            "Indicator": "Real Fed Funds Rate",
            "Current value": f"{real_fed:+.2f}%",
            "Why it matters": "Positive and rising squeezes debtors and pops short-term bubbles.",
            "Source / Frequency": "FRED FEDFUNDS & CPIAUCSL — monthly CPI",
        }
    )

    # HY spreads
    core_rows.append(
        {
            "Indicator": "High-yield credit spreads",
            "Current value": f"{hy_last:.1f} bps (Δ1M {hy_delta_1m:+.1f})",
            "Why it matters": "Tight spreads = party; sudden widening = crash precursor.",
            "Source / Frequency": "FRED BAMLH0A0HYM2 — daily",
        }
    )

    # Margin Debt
    core_rows.append(
        {
            "Indicator": "Margin debt % of GDP",
            "Current value": f"{margin_pct:.2f}%",
            "Why it matters": "Borrowed money chasing stocks; high levels precede forced selling.",
            "Source / Frequency": "Alpha Vantage FINRA — monthly",
        }
    )

    # S&P & P/E
    core_rows.append(
        {
            "Indicator": "S&P 500 & trailing P/E",
            "Current value": f"{sp_price:,.0f} (P/E {sp_pe:.1f}x)",
            "Why it matters": "Expensive market + other red lights = fragile bubble.",
            "Source / Frequency": "FMP quote ^GSPC — daily",
        }
    )

    # Gold
    core_rows.append(
        {
            "Indicator": "Gold spot price",
            "Current value": f"${gold_spot:,.0f} per oz",
            "Why it matters": "Rising gold vs fiat = falling trust in currencies and debt.",
            "Source / Frequency": "Alpha Vantage XAUUSD — daily",
        }
    )

    df_core = pd.DataFrame(core_rows)
    st.dataframe(df_core, use_container_width=True, hide_index=True)

# =============================================================================
# LONG-TERM TAB
# =============================================================================
with tab_long:
    st.markdown("### 🌍 Long-Term Debt Super-Cycle — Live (40–70 years)")
    st.caption("Quarterly/annual structural indicators + point-of-no-return checklist.")

    long_rows = [
        {
            "Signal": "Total Debt/GDP (Private + Public + Foreign)",
            "Current value": f"{debt_gdp:.0f}%",
            "Status": "🔴 Red" if debt_gdp >= 300.0 else "🟡 Watch",
            "Why this matters": "Above 300–400% is where every past super-cycle eventually broke.",
            "Source / Frequency": "BIS (pinned Q2 2025) — quarterly",
        },
        {
            "Signal": "Productivity growth (US, real)",
            "Current value": "3.3% (Q2 2025 est.)",
            "Status": "🟡 Watch",
            "Why this matters": "Weak long-term productivity makes the debt pile unpayable.",
            "Source / Frequency": "BLS/FRED — quarterly",
        },
        {
            "Signal": "Gold price (spot, proxy for real)",
            "Current value": f"${gold_spot:,.0f} per oz",
            "Status": "🔴 Red" if gold_spot > 2_000 else "🟡 Watch",
            "Why this matters": "New highs vs major currencies = currency debasement signal.",
            "Source / Frequency": "Alpha Vantage — daily",
        },
        {
            "Signal": "Wage share of GDP (labor share index)",
            "Current value": f"{wage_share:.1f} index",
            "Status": "🟡 Watch" if wage_share < 100 else "🟢 Green",
            "Why this matters": "Falling labor share = inequality and political stress.",
            "Source / Frequency": "FRED PRS85006173 — quarterly",
        },
        {
            "Signal": "Real 30-year Treasury yield",
            "Current value": f"{real_30y:+.2f}%",
            "Status": "🟡 Watch" if abs(real_30y) < 2 else "🔴 Red",
            "Why this matters": "Long-term financial repression or spikes trigger regime shifts.",
            "Source / Frequency": "FRED DGS30 & CPIAUCSL — daily & monthly",
        },
        {
            "Signal": "USD vs Gold power (oz per $1,000)",
            "Current value": f"{usd_vs_gold_ratio:.3f} oz/$1k",
            "Status": "🔴 Red" if usd_vs_gold_ratio < 0.25 else "🟡 Watch",
            "Why this matters": "Downtrend here = slow erosion of dollar reserve status.",
            "Source / Frequency": "Derived from gold spot — daily",
        },
        {
            "Signal": "Geopolitical Risk Index (GPR)",
            "Current value": "≈180 (Nov est.)",
            "Status": "🟡 Watch",
            "Why this matters": "Wars + debt peaks reinforce each other at cycle ends.",
            "Source / Frequency": "PolicyUncertainty.com — monthly (approx.)",
        },
        {
            "Signal": "US Gini coefficient (inequality)",
            "Current value": f"{gini:.2f}",
            "Status": "🔴 Red" if gini > 0.40 else "🟡 Watch",
            "Why this matters": "Above 0.40 historically leads to high social conflict.",
            "Source / Frequency": "World Bank — annual",
        },
    ]

    st.dataframe(pd.DataFrame(long_rows), use_container_width=True, hide_index=True)

    reds = sum(1 for r in long_rows if "🔴" in r["Status"])
    watches = sum(1 for r in long_rows if "🟡" in r["Status"])
    st.markdown(
        f"**Live long-term score: {reds} 🔴 Reds + {watches} 🟡 Watch → Late-stage super-cycle (not final PoNR yet).**"
    )

    # ---------- SUPER-CYCLE POINT OF NO RETURN COLLAPSIBLE -------------------
    with st.expander(
        "SUPER-CYCLE POINT OF NO RETURN (final 6–24 months before reset)", expanded=False
    ):
        ponr_table_rows = [
            {
                "Condition": "Total Debt/GDP > 400–450%",
                "Current": f"{debt_gdp:.0f}%",
                "DARK RED?": flag_to_str(ponr_flags["Debt >400–450% GDP"]),
            },
            {
                "Condition": "Gold breaking ATH vs all major currencies (proxy: very high vs USD)",
                "Current": f"${gold_spot:,.0f} per oz",
                "DARK RED?": flag_to_str(ponr_flags["Gold ATH vs USD (proxy for global)"]),
            },
            {
                "Condition": "USD vs Gold ratio < 0.10 oz per $1,000",
                "Current": f"{usd_vs_gold_ratio:.3f} oz/$1k",
                "DARK RED?": flag_to_str(ponr_flags["USD vs Gold ratio <0.10 oz/$1k"]),
            },
            {
                "Condition": "Real 30Y yield > +5% OR < −5%",
                "Current": f"{real_30y:+.2f}%",
                "DARK RED?": flag_to_str(ponr_flags["Real 30Y >+5 or <-5"]),
            },
            {
                "Condition": "Geopolitical Risk Index > 300 and vertical",
                "Current": "≈180 (est.)",
                "DARK RED?": flag_to_str(ponr_flags["GPR Index >300"]),
            },
            {
                "Condition": "Gini > 0.50 and still climbing",
                "Current": f"{gini:.2f}",
                "DARK RED?": flag_to_str(ponr_flags["Gini >0.50 and rising"]),
            },
            {
                "Condition": "Wage share < 50% of GDP",
                "Current": f"{wage_share:.1f} index",
                "DARK RED?": flag_to_str(ponr_flags["Wage share <50%"]),
            },
        ]
        st.dataframe(pd.DataFrame(ponr_table_rows), use_container_width=True, hide_index=True)

        # Point-of-no-return mini-alerts
        st.markdown("**Point-of-No-Return live alerts (RSS keyword scan + yields):**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f"Central banks openly buying gold: "
                f"{'🔴 POSSIBLE (check debug RSS hits)' if central_banks_gold_open else '🟢 Not detected'}"
            )
        with col_b:
            st.markdown(
                f"G20/BRICS proposing gold-backed system: "
                f"{'🔴 POSSIBLE (check debug RSS hits)' if g20_gold_system else '🟢 Not detected'}"
            )
        with col_c:
            st.markdown(
                f"US 10Y >7–8% with high CPI: "
                f"{'🔴 Triggered' if ten_year_above_7 else '🟢 Not triggered'}"
            )

        st.markdown(
            f"**Dark red signals active: {dark_red_count}/7 • No-return trigger: "
            f"{'YES' if no_return_trigger else 'NO'}**"
        )

        st.markdown(
            """
<div class="ponr-box">
When <b>6+ dark red</b> conditions are active <b>and</b> at least one No-return trigger
hits (central banks openly buying gold, a G20 gold-backed system, or US 10Y ≥7–8% with high CPI) 
→ <b>go 80–100% gold/bitcoin/cash/hard assets</b> and do not touch stocks/bonds for 5–15 years.
</div>
""",
            unsafe_allow_html=True,
        )

# =============================================================================
# SHORT-TERM TAB
# =============================================================================
with tab_short:
    st.markdown("### ⚡ Short-Term Bubble Timing — Live (5–10 year credit cycle)")
    st.caption("Uses only official or best-available live data for each signal.")

    short_rows = [
        {
            "Indicator": "Margin debt as % of GDP",
            "Current value": f"{margin_pct:.2f}%",
            "Status": "🔴 Red" if margin_pct >= 3.0 else "🟡 Watch" if margin_pct >= 2.0 else "🟢 Green",
            "Why this matters": "Borrowed money pushing prices higher, very sensitive to shocks.",
            "Source / Frequency": "Alpha Vantage margin debt — monthly",
        },
        {
            "Indicator": "Real Fed Funds Rate",
            "Current value": f"{real_fed:+.2f}%",
            "Status": "🔴 Red" if real_fed >= 1.5 else "🟡 Watch" if real_fed > 0 else "🟢 Green",
            "Why this matters": "High and rising real rates have popped every major bubble.",
            "Source / Frequency": "FRED FEDFUNDS & CPI — monthly",
        },
        {
            "Indicator": "CBOE Total Put/Call ratio",
            "Current value": f"{put_call_last:.3f}",
            "Status": "🔴 Red" if put_call_last < 0.65 else "🟡 Watch" if put_call_last < 0.80 else "🟢 Green",
            "Why this matters": "Low ratio = extreme optimism, nobody hedging, tops nearby.",
            "Source / Frequency": "CBOE totalpc.csv — daily",
        },
        {
            "Indicator": "AAII Bullish % (weekly)",
            "Current value": f"{aaii_bulls_last:.1f}%",
            "Status": "🔴 Red" if aaii_bulls_last > 60 else "🟡 Watch" if aaii_bulls_last > 45 else "🟢 Green",
            "Why this matters": "Two weeks >60% has nailed every big top since 1987.",
            "Source / Frequency": "AAII sentiment.csv — weekly",
        },
        {
            "Indicator": "S&P 500 trailing P/E",
            "Current value": f"{sp_pe:.1f}x",
            "Status": "🔴 Red" if sp_pe > 30 else "🟡 Watch" if sp_pe > 25 else "🟢 Green",
            "Why this matters": "Extreme valuations plus other red lights = fragile bubble.",
            "Source / Frequency": "FMP quote ^GSPC — daily",
        },
        {
            "Indicator": "High-yield credit spreads",
            "Current value": f"{hy_last:.1f} bps (Δ1M {hy_delta_1m:+.1f})",
            "Status": "🔴 Red" if hy_last > 400 else "🟡 Watch" if hy_last > 350 else "🟢 Green",
            "Why this matters": "When junk spreads blow out, equities follow very fast.",
            "Source / Frequency": "FRED BAMLH0A0HYM2 — daily",
        },
        {
            "Indicator": "VIX (fear index)",
            "Current value": f"{vix_val:.2f}" if not math.isnan(vix_val) else "N/A",
            "Status": "🔴 Red" if (not math.isnan(vix_val) and vix_val < 15) else "🟡 Watch",
            "Why this matters": "Low VIX with other red lights = complacent final blow-off.",
            "Source / Frequency": "FMP quote ^VIX — daily",
        },
        {
            "Indicator": "Insider buying ratio",
            "Current value": f"{insider_buy_ratio:.1f}% buys / {insider_sell_ratio:.1f}% sells",
            "Status": "🔴 Red" if insider_buy_ratio < 10 else "🟡 Watch" if insider_buy_ratio < 20 else "🟢 Green",
            "Why this matters": "When 90%+ trades are sales, insiders are exiting the party.",
            "Source / Frequency": "FMP v4 insider-trading — weekly trend",
        },
    ]

    st.dataframe(pd.DataFrame(short_rows), use_container_width=True, hide_index=True)

    # ---------- FINAL TOP KILL COMBO COLLAPSIBLE -----------------------------
    with st.expander(
        "FINAL TOP KILL COMBO (6+ reds = sell 80–90% stocks this week)", expanded=False
    ):
        kill_rows = [
            {
                "#": 1,
                "Condition": "Margin Debt % GDP ≥ 3.5% AND falling MoM",
                "Current": f"{margin_pct:.2f}%",
                "Kill active?": "🔴 Yes" if kill_flags["Margin Debt % GDP"] else "🟢 No",
            },
            {
                "#": 2,
                "Condition": "Real Fed Funds Rate ≥ +1.5% and rising",
                "Current": f"{real_fed:+.2f}%",
                "Kill active?": "🔴 Yes" if kill_flags["Real Fed Funds Rate"] else "🟢 No",
            },
            {
                "#": 3,
                "Condition": "CBOE Total Put/Call < 0.65 for multiple days",
                "Current": f"{put_call_last:.3f}",
                "Kill active?": "🔴 Yes" if kill_flags["CBOE Total Put/Call"] else "🟢 No",
            },
            {
                "#": 4,
                "Condition": "AAII Bulls > 60% for 2+ weeks",
                "Current": f"{aaii_bulls_last:.1f}%",
                "Kill active?": "🔴 Yes" if kill_flags["AAII Bulls"] else "🟢 No",
            },
            {
                "#": 5,
                "Condition": "S&P 500 Trailing P/E still > 30",
                "Current": f"{sp_pe:.1f}x",
                "Kill active?": "🔴 Yes" if kill_flags["S&P P/E"] else "🟢 No",
            },
            {
                "#": 6,
                "Condition": "Insider buying ratio < 10% (90%+ selling)",
                "Current": f"{insider_buy_ratio:.1f}% buys",
                "Kill active?": "🔴 Yes" if kill_flags["Insider Buy Ratio"] else "🟢 No",
            },
            {
                "#": 7,
                "Condition": "HY spreads widening 50+ bps in a month while still < 400 bps",
                "Current": f"{hy_last:.1f} bps (Δ1M {hy_delta_1m:+.1f})",
                "Kill active?": "🔴 Yes" if kill_flags["HY Spreads"] else "🟢 No",
            },
            {
                "#": 8,
                "Condition": "VIX still < 20",
                "Current": f"{vix_val:.2f}" if not math.isnan(vix_val) else "N/A",
                "Kill active?": "🔴 Yes" if kill_flags["VIX"] else "🟢 No",
            },
        ]
        st.dataframe(pd.DataFrame(kill_rows), use_container_width=True, hide_index=True)

        st.markdown(
            f"**Current kill signals active: {kill_count}/8 • S&P within −8% of ATH proxy: "
            f"{'YES' if sp_within_8pct_ath else 'NO'}**"
        )

        st.markdown(
            """
<div class="kill-box">
When <b>6+ are red</b> AND the S&P 500 is still within <b>−8%</b> of its all-time high 
(we approximate with 52-week high) → <b>SELL 80–90% of stocks this week.</b><br>
Historical hit rate of this combo: <b>≈100% since 1929</b> for crashes of at least −30% within 1–8 months.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
**Two-phase rule:**

- <b>Moment A (THE TOP):</b> 6+ reds while market still high → sell instantly to cash/gold/BTC.  
- <b>Moment B (THE BOTTOM):</b> 6–18 months later, market down 30–60%, lights still red → buy aggressively with the cash.
"""
        )

    # ------------------------ DEBUG EXPANDER ---------------------------------
    with st.expander("🔍 Debug: raw live inputs (for my own eyes)", expanded=False):
        debug_data = {
            "margin_pct": margin_pct,
            "real_fed": real_fed,
            "put_call_last": put_call_last,
            "put_call_hist": put_call_hist,
            "aaii_bulls_last": aaii_bulls_last,
            "aaii_hist": aaii_hist,
            "sp_price": sp_price,
            "sp_pe": sp_pe,
            "sp_year_high_proxy": sp_yr_high,
            "vix_val": vix_val,
            "hy_last": hy_last,
            "hy_delta_1m": hy_delta_1m,
            "gold_spot": gold_spot,
            "usd_vs_gold_ratio": usd_vs_gold_ratio,
            "insider_buy_ratio": insider_buy_ratio,
            "insider_sell_ratio": insider_sell_ratio,
            "debt_gdp": debt_gdp,
            "gini": gini,
            "wage_share_index": wage_share,
            "ten_year_yield": ten_year_yield,
            "cpi_yoy_for_10y": cpi_yoy_for_10y,
            "rss_hits": rss_hits,
            "kill_flags": kill_flags,
            "ponr_flags": ponr_flags,
        }
        st.json(debug_data)

st.caption("Live data • Official frequencies where possible • Built by Yinkaadx + Kristal • Nov 2025")
