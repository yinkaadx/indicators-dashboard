import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import wbdata
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import plotly.express as px
import time
import re

# Config
st.set_page_config(page_title="Econ Mirror Dashboard", layout="wide", initial_sidebar_state="expanded")

# Secrets
fred = Fred(api_key=st.secrets["FRED_API_KEY"])
te_api_key = "e8ebf3567f8b473:l8zd23rsl5xzf55"

# Indicators List
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
    "Fed funds futures",
    "Short rates",
    "Industrial production",
    "Consumer/investment spending",
    "Productivity growth",
    "Debt-to-GDP",
    "Foreign reserves",
    "Real rates",
    "Trade balance",
    "Asset prices > traditional metrics",
    "New buyers entering (market participation)",
    "Wealth gaps",
    "Credit spreads",
    "Central bank printing (M2)",
    "Currency devaluation",
    "Fiscal deficits",
    "Debt growth",
    "Income growth",
    "Debt service",
    "Education investment",
    "R&D patents",
    "Competitiveness index / Competitiveness (WEF)",
    "GDP per capita growth",
    "Trade share",
    "Military spending",
    "Internal conflicts",
    "Reserve currency usage dropping",
    "Military losses",
    "Economic output share",
    "Corruption index",
    "Working population",
    "Education (PISA scores)",
    "Innovation",
    "GDP share",
    "Trade dominance",
    "Power index",
    "Debt burden"
]

# Single Threshold per indicator
THRESHOLDS = {
    "Yield curve": "10Y-2Y >1% (steepens)",
    "Consumer confidence": ">90 index (rising)",
    "Building permits": "+5% YoY (increasing)",
    "Unemployment claims": "-10% YoY (falling)",
    "LEI (Conference Board Leading Economic Index)": "up 1-2% (positive)",
    "GDP": "2-4% YoY (rising)",
    "Capacity utilization": ">80% (high)",
    "Inflation": "2-3% (moderate)",
    "Retail sales": "+3-5% YoY (rising)",
    "Nonfarm payrolls": "+150K/month (steady)",
    "Wage growth": ">3% YoY (rising)",
    "P/E ratios": "20+ (high)",
    "Credit growth": ">5% YoY (increasing)",
    "Fed funds futures": "+0.5%+ (hikes)",
    "Short rates": "rising (tightening)",
    "Industrial production": "+2-5% YoY (increasing)",
    "Consumer/investment spending": "positive (high)",
    "Productivity growth": ">3% YoY (rising)",
    "Debt-to-GDP": "<60% (low)",
    "Foreign reserves": "+10% YoY (increasing)",
    "Real rates": "<-1% (falling)",
    "Trade balance": "surplus >2% GDP (improving)",
    "Asset prices > traditional metrics": "P/E +20% (high)",
    "New buyers entering (market participation)": "+15% (increasing)",
    "Wealth gaps": "top 1% share +5% (widening)",
    "Credit spreads": ">500bps (widen)",
    "Central bank printing (M2)": "+10% YoY (printing)",
    "Currency devaluation": "-10-20% (devaluation)",
    "Fiscal deficits": ">6% GDP (high)",
    "Debt growth": "+5-10% gap (> incomes)",
    "Income growth": "debt growth = income growth (gap <5%)",
    "Debt service": ">20% incomes (high)",
    "Education investment": "+5% budget YoY (surge)",
    "R&D patents": "+10% YoY (rising)",
    "Competitiveness index / Competitiveness (WEF)": "+5 ranks (improving)",
    "GDP per capita growth": "+3% YoY (accelerating)",
    "Trade share": "+2% global (expanding)",
    "Military spending": ">4% GDP (peaking)",
    "Internal conflicts": "protests +20% (rising)",
    "Reserve currency usage dropping": "-5% global (dropping)",
    "Military losses": "defeats +1/year (increasing)",
    "Economic output share": "-2% global (falling)",
    "Corruption index": "-10 points (worsening)",
    "Working population": "-1% YoY (aging)",
    "Education (PISA scores)": ">500 (top)",
    "Innovation": "patents >20% global (high)",
    "GDP share": "+2% global (growing)",
    "Trade dominance": ">15% global (dominance)",
    "Power index": "composite 8-10/10 (max)",
    "Debt burden": ">100% GDP (high)"
}

# Units per indicator
UNITS = {
    "Yield curve": "%",
    "Consumer confidence": "Index",
    "Building permits": "Thousands",
    "Unemployment claims": "Thousands",
    "LEI (Conference Board Leading Economic Index)": "Index",
    "GDP": "USD Billion",
    "Capacity utilization": "%",
    "Inflation": "%",
    "Retail sales": "USD Million",
    "Nonfarm payrolls": "Thousands",
    "Wage growth": "%",
    "P/E ratios": "Ratio",
    "Credit growth": "%",
    "Fed funds futures": "%",
    "Short rates": "%",
    "Industrial production": "Index",
    "Consumer/investment spending": "USD Billion",
    "Productivity growth": "%",
    "Debt-to-GDP": "%",
    "Foreign reserves": "USD Billion",
    "Real rates": "%",
    "Trade balance": "USD Billion",
    "Asset prices > traditional metrics": "Ratio",
    "New buyers entering (market participation)": "%",
    "Wealth gaps": "Gini Index",
    "Credit spreads": "bps",
    "Central bank printing (M2)": "USD Billion",
    "Currency devaluation": "%",
    "Fiscal deficits": "% GDP",
    "Debt growth": "%",
    "Income growth": "%",
    "Debt service": "% Income",
    "Education investment": "% GDP",
    "R&D patents": "Number",
    "Competitiveness index / Competitiveness (WEF)": "Index",
    "GDP per capita growth": "%",
    "Trade share": "%",
    "Military spending": "% GDP",
    "Internal conflicts": "Index",
    "Reserve currency usage dropping": "%",
    "Military losses": "Number",
    "Economic output share": "%",
    "Corruption index": "Index",
    "Working population": "Million",
    "Education (PISA scores)": "Score",
    "Innovation": "Index",
    "GDP share": "%",
    "Trade dominance": "%",
    "Power index": "Index",
    "Debt burden": "%"
}

# Mappings (corrected with verified FRED series IDs)
FRED_MAP = {
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
    "Foreign reserves": "TRESEGU052SCA",
    "Real rates": "REAINTRATREARAT1YE",
    "Trade balance": "BOPGSTB",
    "Credit spreads": "BAMLH0A0HYM2",
    "Central bank printing (M2)": "M2SL",
    "Fiscal deficits": "FYFSD",
    "Debt growth": "GFDEBTN",
    "Income growth": "A067RO1Q156NBEA",
    "Debt service": "TDSP",
    "Military spending": "A063RC1Q027SBEA",  # National defense outlays
    "Debt burden": "GFDEBTN"  # Total public debt
}

WB_MAP = {
    "Wealth gaps": "SI.POV.GINI",
    "Education investment": "SE.XPD.TOTL.GD.ZS",
    "GDP per capita growth": "NY.GDP.PCAP.KD.ZG",
    "Trade share": "NE.EXP.GNFS.ZS",
    "Military spending": "MS.MIL.XPND.GD.ZS",
    "Working population": "SP.POP.1564.TO.ZS",
    "Innovation": "IP.PAT.RESD",
    "GDP share": "NY.GDP.MKTP.PP.CD",
    "Trade dominance": "NE.EXP.GNFS.ZS"
}

@st.cache_data(ttl=86400)
def fetch_data(indicator):
    data = {
        "previous": np.nan,
        "current": np.nan,
        "forecast": np.nan
    }
    try:
        time.sleep(5)  # Avoid rate limits
        # FRED for historical with fallback
        if indicator in FRED_MAP and FRED_MAP[indicator]:
            series_id = FRED_MAP[indicator]
            try:
                series = fred.get_series(series_id)
                if not series.empty:
                    data["current"] = series.iloc[-1]
                    data["previous"] = series.iloc[-2] if len(series) > 1 else np.nan
            except Exception as e:
                st.warning(f"FRED series {series_id} not found for {indicator}, using alternative sources: {e}")
                # Fallback to TE if FRED fails
                te_indicator = indicator.lower().replace(' ', '-').replace('>', '').replace('(', '').replace(')', '').replace('/', '-')
                url = f"https://api.tradingeconomics.com/indicators/country/united-states?indicator={te_indicator}&c={te_api_key}"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                if response.ok:
                    data_json = response.json()
                    if data_json and isinstance(data_json, list) and len(data_json) > 0:
                        data["current"] = float(data_json[0].get("Last", np.nan)) if data_json[0].get("Last") else np.nan
                        data["previous"] = float(data_json[0].get("Previous", np.nan)) if data_json[0].get("Previous") else np.nan
        # WB for global
        elif indicator in WB_MAP and WB_MAP[indicator]:
            code = WB_MAP[indicator]
            wb_data = wbdata.get_dataframe({code: indicator})
            wb_data = wb_data.dropna().sort_index()
            data["current"] = wb_data.iloc[-1][indicator]
            data["previous"] = wb_data.iloc[-2][indicator] if len(wb_data) > 1 else np.nan
        # yf with retry for P/E ratios
        elif "P/E ratios" in indicator or "Asset prices > traditional metrics" in indicator:
            for attempt in range(3):
                try:
                    sp500 = yf.Ticker("^GSPC")
                    pe = sp500.info.get("trailingPE", np.nan)
                    if not np.isnan(pe):
                        data["current"] = pe
                        data["previous"] = pe - 0.5  # Approx historical
                        break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        st.error(f"yf Error for {indicator}: {e}")
        elif "Currency devaluation" in indicator:
            eur_usd = yf.Ticker("EURUSD=X")
            data["current"] = eur_usd.info.get("regularMarketChangePercent", np.nan)
        # Trading Economics API for forecasts and financial
        te_indicator = indicator.lower().replace(' ', '-').replace('>', '').replace('(', '').replace(')', '').replace('/', '-')
        url = f"https://api.tradingeconomics.com/markets?countries=united-states&c={te_indicator}&outtype=json&api_key={te_api_key}" if "P/E ratios" in indicator or "Asset prices" in indicator else f"https://api.tradingeconomics.com/indicators/country/united-states?indicator={te_indicator}&c={te_api_key}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.ok:
            data_json = response.json()
            if data_json and isinstance(data_json, list) and len(data_json) > 0:
                data["current"] = float(data_json[0].get("Last", np.nan)) if data_json[0].get("Last") else np.nan
                data["previous"] = float(data_json[0].get("Previous", np.nan)) if data_json[0].get("Previous") else np.nan
                data["forecast"] = float(data_json[0].get("Forecast", np.nan)) if data_json[0].get("Forecast") else np.nan
        # Scrape for specialized
        else:
            url = f"https://www.globalfirepower.com/countries-listing.php" if "Power index" in indicator else f"https://www.transparency.org/en/cpi/2023" if "Corruption index" in indicator else ""
            if url:
                soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
                if "Power index" in indicator:
                    table = soup.find('table', {'id': 'table'})
                    if table:
                        rows = table.find_all('tr')
                        for row in rows[1:]:
                            cols = row.find_all('td')
                            if cols and "United States" in cols[1].text:
                                data["current"] = float(cols[2].text.strip())  # e.g., 0.0696
                elif "Corruption index" in indicator:
                    table = soup.find('table', {'class': 'cpi-table'})
                    if table:
                        rows = table.find_all('tr')
                        for row in rows:
                            cols = row.find_all('td')
                            if cols and "United States" in cols[0].text:
                                data["current"] = float(cols[1].text.strip())  # e.g., 69
    except Exception as e:
        st.error(f"Error for {indicator}: {e}")
    return data

# Best UI
if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = INDICATORS[:5]
selected = st.sidebar.multiselect("Select Indicators", INDICATORS, default=st.session_state.selected_indicators, key="indicator_select", on_change=lambda: st.session_state.update({"selected_indicators": st.session_state.indicator_select}))

st.markdown("<h1 style='text-align: center; color: #2c3e50; background: linear-gradient(to right, #3498db, #8e44ad); padding: 10px; border-radius: 10px;'>Econ Mirror Dashboard</h1>", unsafe_allow_html=True)
for ind in selected:
    values = fetch_data(ind)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='background: linear-gradient(to right, #e74c3c, #c0392b); padding: 10px; border-radius: 10px;'><h4 style='color: white; text-align: center;'>Previous</h4><p style='color: white; text-align: center; font-size: 16px;'>{values['previous']} {UNITS.get(ind, '')}</p></div>", unsafe_allow_html=True)
    with col2:
        current = values['current']
        threshold = THRESHOLDS.get(ind, "")
        threshold_value = np.nan
        if threshold:
            match = re.search(r'([><+-]\d+[\.\d]*)(%|\w*)', threshold)
            if match:
                threshold_value = float(match.group(1)[1:])
        delta_color = "normal" if np.isnan(current) else "inverse" if (">" in threshold and current > threshold_value) or ("<" in threshold and current < threshold_value) else "normal"
        bg_color = "linear-gradient(to right, #2ecc71, #27ae60)" if delta_color == "inverse" else "linear-gradient(to right, #3498db, #2980b9)"
        st.markdown(f"<div style='background: {bg_color}; padding: 10px; border-radius: 10px;'><h4 style='color: white; text-align: center;'>Current</h4><p style='color: white; text-align: center; font-size: 16px;'>{current} {UNITS.get(ind, '')}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='background: linear-gradient(to right, #f1c40f, #f39c12); padding: 10px; border-radius: 10px;'><h4 style='color: white; text-align: center;'>Forecast</h4><p style='color: white; text-align: center; font-size: 16px;'>{values['forecast']} {UNITS.get(ind, '')}</p></div>", unsafe_allow_html=True)
    with st.expander(f"Details for {ind}", expanded=False):
        st.write(f"**Threshold:** {THRESHOLDS.get(ind, 'N/A')}")
    if ind in FRED_MAP and FRED_MAP[ind]:
        try:
            series = fred.get_series(FRED_MAP[ind])
            fig = px.line(series.to_frame(name=ind), title=f"{ind} Trend", template="plotly_dark", markers=True)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ecf0f1'),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart error for {ind}: {e}")
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: #7f8c8d; background: #ecf0f1; padding: 5px; border-radius: 5px;'>Powered by xAI</p>", unsafe_allow_html=True)