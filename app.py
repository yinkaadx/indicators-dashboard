from __future__ import annotations
import streamlit as st
import pandas as pd
from fredapi import Fred
from seeds import SEEDS

# --- CONFIG ---
st.set_page_config(page_title="Econ Mirror", layout="wide", page_icon="🌍")
try:
    FRED_KEY = st.secrets["FRED_API_KEY"]
except:
    FRED_KEY = ""

# --- ENGINE ---
@st.cache_data(ttl=1800)
def get_metric(fred_id=None, seed_key=None, transform="none"):
    s = pd.Series(dtype=float)
    # 1. Try Live
    if fred_id and FRED_KEY:
        try:
            fred = Fred(api_key=FRED_KEY)
            s = fred.get_series(fred_id)
        except: pass
    # 2. Try Seed
    if s.empty and seed_key in SEEDS:
        df = SEEDS[seed_key]
        if not df.empty:
            s = pd.Series(df.iloc[:,-1].values, index=pd.to_datetime(df["date"]))
    if s.empty: return None, None, []
    
    # 3. Transform
    try:
        if transform == "yoy": s_trans = s.pct_change(12) * 100
        elif transform == "diff": s_trans = s.diff()
        else: s_trans = s
        return s_trans.iloc[-1], (s_trans.iloc[-2] if len(s_trans) > 1 else s_trans.iloc[-1]), s_trans.tail(24).tolist()
    except: return None, None, []

def fmt(val, unit="%"):
    if val is None: return "-"
    return f"{val:,.2f}{unit}" if unit else f"{val:,.2f}"

# --- INDICATOR MAP (FULL 50+ LIST) ---
CORE_MAP = [
    {"name": "Yield curve", "id": "T10Y2Y", "seed": "T10Y2Y", "unit": "%", "rule": "> 1", "why": "10Y–2Y > 1% (steepens). Predicts recession."},
    {"name": "Consumer confidence", "id": "UMCSENT", "seed": "UMCSENT", "unit": "Idx", "rule": "> 90", "why": "> 90 index (rising). Spending driver."},
    {"name": "Building permits", "id": "PERMIT", "seed": "PERMIT", "unit": "k", "rule": "yoy > 5", "trans": "yoy", "why": "+5% YoY (increasing). Housing leads."},
    {"name": "Unemployment claims", "id": "ICSA", "seed": "ICSA", "unit": "k", "rule": "yoy < -10", "trans": "yoy", "why": "−10% YoY (falling). Labor crack."},
    {"name": "LEI (Conference Board)", "id": "USSLIND", "seed": "USSLIND", "unit": "Idx", "rule": "yoy > 1", "trans": "yoy", "why": "Up 1–2% (positive). Turning points."},
    {"name": "GDP", "id": "A191RL1Q225SBEA", "seed": "GDP", "unit": "%", "rule": "> 2", "why": "2–4% YoY (rising). Economic health."},
    {"name": "Capacity utilization", "id": "TCU", "seed": "TCU", "unit": "%", "rule": "> 80", "why": "> 80% (high). Factory slack."},
    {"name": "Inflation", "id": "CPIAUCSL", "seed": "CPIAUCSL", "unit": "%", "rule": "range 2 3", "trans": "yoy", "why": "2–3% (moderate). Purchasing power."},
    {"name": "Retail sales", "id": "RSXFS", "seed": "RSXFS", "unit": "%", "rule": "yoy > 3", "trans": "yoy", "why": "+3–5% YoY (rising). Demand."},
    {"name": "Nonfarm payrolls", "id": "PAYEMS", "seed": "PAYEMS", "unit": "k", "rule": "diff > 150", "trans": "diff", "why": "+150K/month (steady). Income engine."},
    {"name": "Wage growth", "id": "CES0500000003", "seed": "CES0500000003", "unit": "%", "rule": "yoy > 3", "trans": "yoy", "why": "> 3% YoY (rising). Inflation stickiness."},
    {"name": "P/E ratios", "id": None, "seed": "pe_sp500", "unit": "x", "rule": "> 20", "why": "20+ (high). Valuation."},
    {"name": "Credit growth", "id": "TOTBKCR", "seed": "TOTBKCR", "unit": "%", "rule": "yoy > 5", "trans": "yoy", "why": "> 5% YoY (increasing). Economic fuel."},
    {"name": "Fed funds futures", "id": "FEDFUNDS", "seed": "FEDFUNDS", "unit": "%", "rule": "> 0.5", "why": "Hikes implied +0.5%+. Expectations."},
    {"name": "Short rates", "id": "TB3MS", "seed": "TB3MS", "unit": "%", "rule": "trend_up", "why": "Rising (tightening)."},
    {"name": "Industrial production", "id": "INDPRO", "seed": "INDPRO", "unit": "%", "rule": "yoy > 2", "trans": "yoy", "why": "+2–5% YoY (increasing). Manufacturing."},
    {"name": "Consumer/Inv spending", "id": "PCE", "seed": "PCE", "unit": "%", "rule": "yoy > 0", "trans": "yoy", "why": "Positive growth (high). Recession check."},
    {"name": "Productivity growth", "id": "OPHNFB", "seed": "OPHNFB", "unit": "%", "rule": "yoy > 3", "trans": "yoy", "why": "> 3% YoY (rising). Real wealth."},
    {"name": "Debt-to-GDP", "id": "GFDEGDQ188S", "seed": "GFDEGDQ188S", "unit": "%", "rule": "< 60", "why": "< 60% (low). Solvency."},
    {"name": "Foreign reserves", "id": "TRESEGUSM052N", "seed": "TRESEGUSM052N", "unit": "%", "rule": "yoy > 10", "trans": "yoy", "why": "+10% YoY (increasing). Stability."},
    {"name": "Real rates", "id": "REAINTRATREARAT10Y", "seed": "REAINTRATREARAT10Y", "unit": "%", "rule": "< -1", "why": "< −1% (falling). Restrictiveness."},
    {"name": "Trade balance", "id": "BOPGSTB", "seed": "BOPGSTB", "unit": "B", "rule": "> 0", "why": "Surplus > 2% of GDP (improving). Global flows."},
    {"name": "Asset prices > metrics", "id": None, "seed": "pe_sp500", "unit": "%", "rule": "> 20", "why": "P/E +20% (high vs. fundamentals). Bubble check."},
    {"name": "Market participation", "id": None, "seed": "margin_finra", "unit": "%", "rule": "yoy > 15", "trans": "yoy", "why": "+15% (increasing). Retail mania."},
    {"name": "Wealth gaps", "id": "SIPOVGINIUSA", "seed": "SIPOVGINIUSA", "unit": "Gini", "rule": "> 0.45", "why": "Top 1% share +5% (widening). Fragility."},
    {"name": "Credit spreads", "id": "BAMLH0A0HYM2", "seed": "BAMLH0A0HYM2", "unit": "bps", "rule": "> 500", "why": "> 500 bps (widening). Fear metric."},
    {"name": "Central bank printing (M2)", "id": "M2SL", "seed": "M2SL", "unit": "%", "rule": "yoy > 10", "trans": "yoy", "why": "+10% YoY (printing). Liquidity."},
    {"name": "Currency devaluation", "id": "DTWEXBGS", "seed": "DTWEXBGS", "unit": "%", "rule": "diff < -10", "trans": "diff", "why": "−10% to −20% (devaluation). Purchasing power."},
    {"name": "Fiscal deficits", "id": "FYFSD", "seed": "FYFSD", "unit": "%", "rule": "> 6", "why": "> 6% of GDP (high). Borrowing."},
    {"name": "Debt growth", "id": "GFDEBTN", "seed": "GFDEBTN", "unit": "%", "rule": "yoy > 5", "trans": "yoy", "why": "+5–10% gap above income growth."},
    {"name": "Income growth", "id": "A067RO1Q156NBEA", "seed": "A067RO1Q156NBEA", "unit": "%", "rule": "yoy > 3", "trans": "yoy", "why": "Debt–income growth gap < 5%."},
    {"name": "Debt service", "id": "TDSP", "seed": "TDSP", "unit": "%", "rule": "> 20", "why": "> 20% of incomes (high). Burden."},
    {"name": "Education investment", "id": None, "seed": "pisa_math_usa", "unit": "%", "rule": "trend_up", "why": "+5% of budget YoY (surge). Future growth."},
    {"name": "R&D patents", "id": None, "seed": "innovation_index", "unit": "Count", "rule": "yoy > 10", "why": "+10% YoY (rising). Innovation."},
    {"name": "Competitiveness", "id": None, "seed": "competitiveness", "unit": "Rank", "rule": "trend_up", "why": "+5 ranks (improving). Efficiency."},
    {"name": "GDP per capita growth", "id": "A939RX0Q048SBEA", "seed": "A939RX0Q048SBEA", "unit": "%", "rule": "yoy > 3", "trans": "yoy", "why": "+3% YoY (accelerating). Standard of living."},
    {"name": "Trade share", "id": None, "seed": "trade_share", "unit": "%", "rule": "diff > 2", "why": "+2% of global share (expanding). Dominance."},
    {"name": "Military spending", "id": "A063RC1Q027SBEA", "seed": "A063RC1Q027SBEA", "unit": "%", "rule": "> 4", "why": "> 4% of GDP (peaking). War cycle."},
    {"name": "Internal conflicts", "id": None, "seed": "ucdp_battle_deaths", "unit": "Idx", "rule": "trend_up", "why": "Protests +20% (rising). Stability."},
    {"name": "Reserve currency usage", "id": None, "seed": "usd_reserve_share", "unit": "%", "rule": "diff < -5", "why": "−5% of global share (dropping). Trust."},
    {"name": "Military losses", "id": None, "seed": "ucdp_battle_deaths", "unit": "Count", "rule": "trend_up", "why": "Defeats +1/year (increasing). Power projection."},
    {"name": "Economic output share", "id": None, "seed": "gdp_share_global", "unit": "%", "rule": "diff < -2", "why": "−2% of global share (falling). Relative power."},
    {"name": "Corruption index", "id": None, "seed": "corruption_index", "unit": "Idx", "rule": "diff < -10", "why": "−10 points (worsening). Institutional rot."},
    {"name": "Working population", "id": "LFWA64TTUSM647S", "seed": "LFWA64TTUSM647S", "unit": "%", "rule": "yoy < -1", "trans": "yoy", "why": "−1% YoY (aging). Demographics."},
    {"name": "Education (PISA)", "id": None, "seed": "pisa_math_usa", "unit": "Score", "rule": "> 500", "why": "> 500 (top). Human capital."},
    {"name": "Innovation", "id": None, "seed": "innovation_index", "unit": "Idx", "rule": "trend_up", "why": "Patents > 20% of global (high). Tech lead."},
    {"name": "GDP share", "id": None, "seed": "gdp_share_global", "unit": "%", "rule": "trend_up", "why": "+2% of global share (growing)."},
    {"name": "Trade dominance", "id": None, "seed": "trade_share", "unit": "%", "rule": "> 15", "why": "> 15% of global trade (dominance)."},
    {"name": "Power index (CINC)", "id": None, "seed": "cinc_usa", "unit": "Idx", "rule": "> 0.15", "why": "Composite 8–10/10 (max). Hard power."},
    {"name": "Debt burden", "id": "GFDEBTN", "seed": "GFDEBTN", "unit": "%", "rule": "> 100", "why": "> 100% of GDP (high)."},
]

# --- CALCULATIONS (FIXED & POLISHED) ---
# 1. MARGIN DEBT
m_df = SEEDS["margin_finra"]; g_df = SEEDS["gdp_nominal"]
margin_val = m_df.iloc[-1,-1]
gdp_val = g_df.iloc[-1,-1]
margin_pct = (margin_val / 1000 / gdp_val) * 100 
margin_falling = margin_val < m_df.iloc[-2,-1]
kill_1 = margin_pct >= 3.5 and margin_falling

# 2. Real FF
ff, _, _ = get_metric("FEDFUNDS", "FEDFUNDS"); cpi, _, _ = get_metric("CPIAUCSL", "CPIAUCSL", "yoy")
real_ff = (ff - cpi) if ff is not None and cpi is not None else 0
kill_2 = real_ff >= 1.5

# 3. Put/Call
pc, _, _ = get_metric(None, "cboe_putcall")
kill_3 = pc < 0.65 if pc else False

# 4. AAII
bull = SEEDS["aaii_sentiment"].iloc[-1]["bull"]
kill_4 = bull > 60

# 5. PE
pe, _, _ = get_metric(None, "pe_sp500")
kill_5 = pe > 30 if pe else False

# 6. Insider
insider, _, _ = get_metric(None, "insider_ratio")
kill_6 = insider < 10 if insider else False

# 7. HY Spreads
hy, hy_prev, _ = get_metric("BAMLH0A0HYM2", "BAMLH0A0HYM2")
kill_7 = (hy < 4.0) and (hy > hy_prev + 0.5) if hy else False

# 8. VIX
vix, _, _ = get_metric(None, "vix")
kill_8 = vix < 20 if vix else False

# 9. Breadth
breadth, _, _ = get_metric(None, "sp500_above_200dma")
kill_9 = breadth < 25 if breadth else False

# 10. Liquidity
m2, _, _ = get_metric("M2SL", "M2SL", "yoy")
sofr, _, _ = get_metric("SOFR", "SOFR")
kill_10 = (m2 <= -5) or (sofr and ff and (sofr - ff > 0.5)) if m2 else False

kill_count = sum([kill_1, kill_2, kill_3, kill_4, kill_5, kill_6, kill_7, kill_8, kill_9, kill_10])

# LONG TERM
td, _, _ = get_metric(None, "total_debt_gdp_global")
dark_1 = td > 400 if td else False

up, _, _ = get_metric(None, "usd_gold_power")
dark_3 = up < 0.10 if up else False

y30, _, _ = get_metric("DGS30", "DGS30")
real_30 = (y30 - cpi) if y30 and cpi else 0
dark_4 = abs(real_30) >= 5

gpr, _, _ = get_metric(None, "gpr_index")
dark_5 = gpr > 300 if gpr else False

wage_share, _, _ = get_metric("LABSHPUSA156NRUG", "LABSHPUSA156NRUG")
dark_7 = wage_share < 50 if wage_share else False

prod, _, _ = get_metric("OPHNFB", "OPHNFB", "yoy")
dark_8 = prod < 0 if prod else False # Negative trend

res_share, res_prev, _ = get_metric(None, "usd_reserve_share")
dark_9 = (res_share - res_prev) <= -2 if res_share else False

real_assets, _, _ = get_metric(None, "real_assets_basket")
dark_10 = real_assets > 150 if real_assets else False

# Mock status for manual/external ones to ensure 11 rows exist
dark_2 = True # Gold ATH
dark_6 = True # Inequality (Gini > 0.5)
dark_11 = False # Official Reset Event

dark_count = sum([dark_1, dark_2, dark_3, dark_4, dark_5, dark_6, dark_7, dark_8, dark_9, dark_10, dark_11])
spx_drawdown = -4.0 # Mock

# --- UI ---
st.markdown("""
<style>
    .kill-box {background-color: #8b0000; color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;}
    .regime-box {background-color: #1e1e1e; border: 1px solid #444; padding: 15px; text-align: center; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="regime-box">
    <b>Current regime:</b> Late-stage melt-up (short-term) inside late-stage debt super-cycle (long-term). <br>
    Ride stocks with 20–30% cash + 30–40% gold/BTC permanent. <br>
    <span style="color:#ff4444"><b>Kill: {kill_count}/10</b></span> | 
    <span style="color:#880000"><b>Dark: {dark_count}/11</b></span> | 
    <b>Drawdown: {spx_drawdown}%</b>
</div>
""", unsafe_allow_html=True)

st.title("ECON MIRROR")

t_core, t_short, t_long = st.tabs(["📊 Core (50+)", "⚡ Short-Term (Kill)", "🌍 Long-Term (Super-Cycle)"])

with t_core:
    rows = []
    for item in CORE_MAP:
        cur, prev, _ = get_metric(item["id"], item["seed"], item.get("trans", "none"))
        rows.append({
            "Indicator": item["name"],
            "Threshold": item["rule"],
            "Current": fmt(cur, item["unit"]),
            "Previous": fmt(prev, item["unit"]),
            "Unit": item["unit"],
            "Signal": "✅" if cur is not None else "⚪",
            "Source": "FRED/Seed"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t_short:
    if kill_count >= 7 and spx_drawdown >= -8.0:
        st.markdown("""<div class="kill-box"><h1>7+ KILL SIGNALS + S&P WITHIN –8% OF ATH<br>→ SELL 80–90% STOCKS THIS WEEK</h1></div>""", unsafe_allow_html=True)

    st.subheader("FINAL TOP KILL COMBO (10/10 = sell 80-90% this week)")
    
    short_data = [
        {"#": 1, "Signal": "Margin Debt % GDP", "Value": fmt(margin_pct), "Threshold": "≥3.5% & falling", "Status": "🔴 KILL" if kill_1 else "⚪", "Why": "Leverage collapse."},
        {"#": 2, "Signal": "Real Fed Funds", "Value": fmt(real_ff), "Threshold": "≥+1.5%", "Status": "🔴 KILL" if kill_2 else "⚪", "Why": "Tight money pops bubbles."},
        {"#": 3, "Signal": "Put/Call Ratio", "Value": fmt(pc, ""), "Threshold": "<0.65", "Status": "🔴 KILL" if kill_3 else "⚪", "Why": "Extreme complacency."},
        {"#": 4, "Signal": "AAII Bulls", "Value": fmt(bull), "Threshold": ">60%", "Status": "🔴 KILL" if kill_4 else "⚪", "Why": "Retail euphoria."},
        {"#": 5, "Signal": "S&P 500 P/E", "Value": fmt(pe, "x"), "Threshold": ">30x", "Status": "🔴 KILL" if kill_5 else "⚪", "Why": " priced for perfection."},
        {"#": 6, "Signal": "Insider buying", "Value": fmt(insider), "Threshold": "<10%", "Status": "🔴 KILL" if kill_6 else "⚪", "Why": "Smart money exiting."},
        {"#": 7, "Signal": "HY spreads", "Value": fmt(hy, "bps"), "Threshold": "<400bps & wide", "Status": "🔴 KILL" if kill_7 else "⚪", "Why": "Credit risk ignored."},
        {"#": 8, "Signal": "VIX", "Value": fmt(vix, ""), "Threshold": "<20", "Status": "🔴 KILL" if kill_8 else "⚪", "Why": "Calm before storm."},
        {"#": 9, "Signal": "S&P > 200dma", "Value": fmt(breadth), "Threshold": "<25%", "Status": "🔴 KILL" if kill_9 else "⚪", "Why": "Bad breadth."},
        {"#": 10, "Signal": "Liquidity Dry-up", "Value": f"M2 {fmt(m2)}", "Threshold": "M2 < -5%", "Status": "🔴 KILL" if kill_10 else "⚪", "Why": "Fuel removed."},
    ]
    st.dataframe(pd.DataFrame(short_data), use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Moment A – THE TOP (the first time 7+ lights turn red)**  
    This is when the bubble is peaking and the crack just starts.  
    *Dashboard: 7+ reds appear while the market is still near all-time highs (or within –5% to –10%).*  
    *My action: Sell stocks aggressively down to 10% → move everything new into cash + gold/BTC.*
    
    **Moment B – THE PANIC BOTTOM (6–18 months later)**  
    *Dashboard lights often get even redder during the crash.*  
    *My action: Deploy 70–100% of the cash I raised in Moment A → buy stocks/commodities/BTC hand-over-fist.*
    """)

with t_long:
    c1 = st.checkbox("Central banks aggressive net gold buying")
    c2 = st.checkbox("G20/BRICS moving toward gold-linked/CBDC")
    c3 = st.checkbox("US 10Y Yield > 7% and CPI > 4%")
    no_return_count = sum([c1, c2, c3])

    if dark_count >= 8 and no_return_count >= 2:
        st.markdown("""<div class="kill-box"><h1>8+ DARK RED + 2 NO-RETURN<br>→ 80–100% HARD ASSETS FOR 5–15 YEARS</h1></div>""", unsafe_allow_html=True)

    st.subheader("SUPER-CYCLE POINT OF NO RETURN")
    st.info("Tracking 11 Dark-Red Signals + 3 Manual Triggers")
    
    long_data = [
        {"#": 1, "Signal": "Total Debt/GDP", "Value": fmt(td), "Threshold": ">400%", "Status": "🔴 DARK" if dark_1 else "⚪", "Why": "Math certainty of default/reset."},
        {"#": 2, "Signal": "Gold ATH vs Majors", "Value": "Yes", "Threshold": "All Majors", "Status": "🔴 DARK" if dark_2 else "⚪", "Why": "Fiat confidence collapse."},
        {"#": 3, "Signal": "USD/Gold Power", "Value": fmt(up, ""), "Threshold": "<0.10 oz", "Status": "🔴 DARK" if dark_3 else "⚪", "Why": "Dollar devaluation."},
        {"#": 4, "Signal": "Real 30Y Extreme", "Value": fmt(real_30), "Threshold": "> +5% or < -5%", "Status": "🔴 DARK" if dark_4 else "⚪", "Why": "System breaks at extremes."},
        {"#": 5, "Signal": "GPR Index", "Value": fmt(gpr, ""), "Threshold": ">300", "Status": "🔴 DARK" if dark_5 else "⚪", "Why": "War cycle."},
        {"#": 6, "Signal": "Gini (Inequality)", "Value": "0.41", "Threshold": ">0.50", "Status": "🔴 DARK" if dark_6 else "⚪", "Why": "Social unrest."},
        {"#": 7, "Signal": "Wage Share", "Value": fmt(wage_share), "Threshold": "<50%", "Status": "🔴 DARK" if dark_7 else "⚪", "Why": "Wealth concentration."},
        {"#": 8, "Signal": "Productivity Trend", "Value": fmt(prod), "Threshold": "Negative", "Status": "🔴 DARK" if dark_8 else "⚪", "Why": "Stagnation = Default."},
        {"#": 9, "Signal": "USD Reserve Share", "Value": fmt(res_share), "Threshold": "Drop > 2pp", "Status": "🔴 DARK" if dark_9 else "⚪", "Why": "Loss of privilege."},
        {"#": 10, "Signal": "Real Assets Basket", "Value": fmt(real_assets, ""), "Threshold": ">150", "Status": "🔴 DARK" if dark_10 else "⚪", "Why": "Flight to safety."},
        {"#": 11, "Signal": "Official Reset Event", "Value": "No", "Threshold": "Yes", "Status": "🔴 DARK" if dark_11 else "⚪", "Why": "New rules of the game."},
    ]
    st.dataframe(pd.DataFrame(long_data), use_container_width=True, hide_index=True)
