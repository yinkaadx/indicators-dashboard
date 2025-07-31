import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import wbdata
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import plotly.express as px

# Config
st.set_page_config(page_title="Econ Mirror Dashboard", layout="wide")

# Secrets
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

# Indicators List
INDICATORS = [
    "10-2 Year Treasury Yield Spread",
    "Consumer Confidence Index",
    "Building Permits Issued",
    "Initial Jobless Claims",
    "Leading Economic Index (LEI)",
    "Gross Domestic Product (GDP)",
    "Capacity Utilization Rate",
    "Consumer Price Index (CPI)",
    "Retail Sales",
    "Nonfarm Payroll Employment",
    "Average Hourly Earnings Growth",
    "Price-to-Earnings Ratio (P/E)",
    "Bank Credit Growth",
    "Federal Funds Futures",
    "Short-Term Interest Rates",
    "Industrial Production Index",
    "Personal Consumption Expenditures (PCE) / Gross Private Domestic Investment",
    "Labor Productivity Growth",
    "Government Debt to GDP Ratio",
    "Foreign Exchange Reserves",
    "Real Interest Rates",
    "Balance of Trade",
    "Asset Prices Relative to Fundamentals",
    "Retail Investor Participation Rate",
    "Gini Coefficient (Wealth Inequality)",
    "Credit Spreads (e.g., High-Yield Bond Spread)",
    "Money Supply (M2)",
    "Currency Depreciation Rate",
    "Government Budget Deficit",
    "Public Debt Growth Rate",
    "Personal Income Growth Rate",
    "Debt Service Coverage Ratio",
    "Education Expenditure as % of GDP",
    "Patent Applications Filed",
    "Global Competitiveness Index (GCI)",
    "GDP Per Capita Growth Rate",
    "Share of World Trade",
    "Military Expenditure as % of GDP",
    "Internal Conflict Index",
    "Share of Currency in Global Reserves",
    "Military Casualties",
    "Share of Global GDP",
    "Corruption Perceptions Index (CPI)",
    "Working-Age Population (Ages 15-64)",
    "PISA Scores",
    "Global Innovation Index",
    "Share of World GDP",
    "Export Market Share",
    "Global Firepower Index",
    "Debt Burden Ratio"
]

# Mappings
FRED_MAP = {
    "10-2 Year Treasury Yield Spread": "T10Y2Y",
    "Consumer Confidence Index": "UMCSENT",
    "Building Permits Issued": "PERMIT",
    "Initial Jobless Claims": "ICSA",
    "Leading Economic Index (LEI)": "USSLIND",
    "Gross Domestic Product (GDP)": "GDP",
    "Capacity Utilization Rate": "TCU",
    "Consumer Price Index (CPI)": "CPIAUCSL",
    "Retail Sales": "RSXFS",
    "Nonfarm Payroll Employment": "PAYEMS",
    "Average Hourly Earnings Growth": "AHETPI",
    "Bank Credit Growth": "TOTBKCR",
    "Short-Term Interest Rates": "TB3MS",
    "Industrial Production Index": "INDPRO",
    "Personal Consumption Expenditures (PCE) / Gross Private Domestic Investment": "PCE",
    "Labor Productivity Growth": "OPHNFB",
    "Government Debt to GDP Ratio": "GFDEGDQ188S",
    "Foreign Exchange Reserves": "TRESEGU052SCA",
    "Real Interest Rates": "REAINTRATREARAT1YE",
    "Balance of Trade": "BOPGSTB",
    "Money Supply (M2)": "M2SL",
    "Government Budget Deficit": "FYFSD",
    "Public Debt Growth Rate": "GFDEBTN",
    "Personal Income Growth Rate": "A067RO1Q156NBEA",
    "Credit Spreads (e.g., High-Yield Bond Spread)": "BAMLH0A0HYM2",
    "Debt Burden Ratio": "TDSP"
}

WB_MAP = {
    "Gini Coefficient (Wealth Inequality)": "SI.POV.GINI",
    "Education Expenditure as % of GDP": "SE.XPD.TOTL.GD.ZS",
    "GDP Per Capita Growth Rate": "NY.GDP.PCAP.KD.ZG",
    "Military Expenditure as % of GDP": "MS.MIL.XPND.GD.ZS",
    "Working-Age Population (Ages 15-64)": "SP.POP.1564.TO.ZS"
}

@st.cache_data(ttl=86400)
def fetch_data(indicator):
    data = pd.DataFrame()
    try:
        if indicator in FRED_MAP and FRED_MAP[indicator]:
            series_id = FRED_MAP[indicator]
            data = fred.get_series(series_id).to_frame(name=indicator)
        elif indicator in WB_MAP and WB_MAP[indicator]:
            code = WB_MAP[indicator]
            data = wbdata.get_dataframe({code: indicator})
            data = data.dropna().rename(columns={code: indicator})
        elif "Price-to-Earnings Ratio (P/E)" in indicator:
            sp500 = yf.Ticker("^GSPC")
            data = pd.DataFrame({"P/E": [sp500.info.get("trailingPE", np.nan)]}, index=[pd.Timestamp.now()])
        elif "Asset Prices Relative to Fundamentals" in indicator:
            sp500 = yf.Ticker("^GSPC")
            pe = sp500.info.get("trailingPE", np.nan)
            historical_avg = 15
            data = pd.DataFrame({"Relative": [pe / historical_avg]}, index=[pd.Timestamp.now()])
        elif "Retail Investor Participation Rate" in indicator:
            data = pd.DataFrame({"Rate": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Currency Depreciation Rate" in indicator:
            eur_usd = yf.Ticker("EURUSD=X")
            rate = eur_usd.info.get("regularMarketPrice", np.nan) - eur_usd.info.get("previousClose", np.nan)
            data = pd.DataFrame({"Depreciation": [rate]}, index=[pd.Timestamp.now()])
        elif "Debt Service Coverage Ratio" in indicator:
            data = pd.DataFrame({"DSCR": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Patent Applications Filed" in indicator:
            url = "https://api.uspto.gov/patents/search?query=patent_date>2020"
            response = requests.get(url)
            if response.ok:
                data = pd.DataFrame({"Patents": [len(response.json())]}, index=[pd.Timestamp.now()])
        elif "Global Competitiveness Index (GCI)" in indicator:
            url = "https://www.weforum.org/reports/global-competitiveness-report-2020"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            data = pd.DataFrame({"GCI": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Share of World Trade" in indicator:
            world_exports = wbdata.get_dataframe({"NE.EXP.GNFS.CD": "Exports"})["Exports"].sum()
            data = pd.DataFrame({"Share": [world_exports / world_exports]}, index=[pd.Timestamp.now()])
        elif "Internal Conflict Index" in indicator:
            data = pd.DataFrame({"Index": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Share of Currency in Global Reserves" in indicator:
            url = "https://data.imf.org/regular.aspx?key=41175"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            data = pd.DataFrame({"Share": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Military Casualties" in indicator:
            data = pd.DataFrame({"Casualties": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Share of Global GDP" in indicator:
            world_gdp = wbdata.get_dataframe({"NY.GDP.MKTP.CD": "GDP"})["GDP"].sum()
            data = pd.DataFrame({"Share": [world_gdp / world_gdp]}, index=[pd.Timestamp.now()])
        elif "Corruption Perceptions Index (CPI)" in indicator:
            url = "https://www.transparency.org/en/cpi/2023"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            data = pd.DataFrame({"CPI": [np.nan]}, index=[pd.Timestamp.now()])
        elif "PISA Scores" in indicator:
            url = "https://www.oecd.org/pisa/data/"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            data = pd.DataFrame({"PISA": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Global Innovation Index" in indicator:
            url = "https://www.wipo.int/global-innovation-index/en/2024/"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            data = pd.DataFrame({"GII": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Export Market Share" in indicator:
            data = pd.DataFrame({"Share": [np.nan]}, index=[pd.Timestamp.now()])
        elif "Global Firepower Index" in indicator:
            url = "https://www.globalfirepower.com/countries-listing.php"
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            data = pd.DataFrame({"GFP": [np.nan]}, index=[pd.Timestamp.now()])
    except Exception as e:
        st.error(f"Error for {indicator}: {e}")
    return data if not data.empty else pd.DataFrame()

# UI
st.title("Econ Mirror Dashboard")
selected = st.multiselect("Select Indicators", INDICATORS, default=INDICATORS[:5])

for ind in selected:
    data = fetch_data(ind)
    if not data.empty:
        st.subheader(ind)
        fig = px.line(data, title=ind)
        st.plotly_chart(fig)
        st.dataframe(data.tail(10))
