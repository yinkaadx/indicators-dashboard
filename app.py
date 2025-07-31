import streamlit as st
import pandas as pd
import time
from fredapi import Fred

st.set_page_config(page_title="Econ Mirror Dashboard", layout="wide")
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

INDICATORS = ["Gross Domestic Product (GDP)", "Consumer Price Index (CPI)"]
FRED_MAP = {"Gross Domestic Product (GDP)": "GDP", "Consumer Price Index (CPI)": "CPIAUCSL"}

@st.cache_data(ttl=86400)
def fetch_data(indicator):
    time.sleep(1)
    data = pd.DataFrame()
    try:
        if indicator in FRED_MAP:
            data = fred.get_series(FRED_MAP[indicator]).to_frame(name=indicator)
    except Exception as e:
        st.error(f"Error for {indicator}: {e}")
    return data

st.title("Econ Mirror Dashboard")
selected = st.multiselect("Select Indicators", INDICATORS, default=["Gross Domestic Product (GDP)"])

for ind in selected:
    data = fetch_data(ind)
    if not data.empty:
        st.subheader(ind)
        st.line_chart(data)
        st.dataframe(data.tail(10))
