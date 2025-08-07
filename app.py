Remove-Item -Path app.py -Force
@"
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

# Config
st.set_page_config(page_title="Econ Mirror Dashboard", layout="wide")

# Secrets
fred = Fred(api_key=st.secrets["FRED_API_KEY"])

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

# Thresholds (full from your text, concatenated for brevity)
THRESHOLDS = {
    "Yield curve": "Early Recovery (3-6+): steepens (post-inversion, 10Y-2Y >1%). Mid Steady Growth (6-12): Stable positive slope. Late Overheating (6-18): - . Tightening (3-9): flattening (10Y-2Y <0.5%). Early Recession (6-18): inversion (10Y-2Y <0). Late Recession (3-6): re-steepening (>1%). Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: flattening (10Y-2Y <0.5%). Early Recession: - . Late Recession: re-steepening (>1%). Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): inversion (10Y-2Y <0). Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Consumer confidence": "Early Recovery (3-6+): Rising (>90 index). Mid Steady Growth (6-12): - . Late Overheating (6-18): - . Tightening (3-9): - . Early Recession (6-18): Falling (<85). Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Building permits": "Early Recovery (3-6+): Increasing (+5% YoY). Mid Steady Growth (6-12): - . Late Overheating (6-18): - . Tightening (3-9): - . Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Unemployment claims": "Early Recovery (3-6+): Falling (-10% YoY). Mid Steady Growth (6-12): - . Late Overheating (6-18): - . Tightening (3-9): - . Early Recession (6-18): Rising (+10% YoY). Late Recession (3-6): - . Coincident Early Recovery: Falling (from peaks). Mid Steady Growth: - . Late Overheating: Low unemployment <5%. Tightening: Low unemployment <5%. Early Recession: Rising (+0.5% YoY). Late Recession: Peaking. Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: >10%. Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "LEI (Conference Board Leading Economic Index)": "Early Recovery (3-6+): Positive (up 1-2%). Mid Steady Growth (6-12): - . Late Overheating (6-18): - . Tightening (3-9): - . Early Recession (6-18): Falling (-1%+). Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "GDP": "Mid Steady Growth (6-12): above potential (1-2% gap). Late Overheating (6-18): > potential (2%+ gap). Tightening (3-9): stable but peaking. Early Recession (6-18): slowdown (<1% YoY). Coincident Early Recovery: Rising (2-4% YoY). Mid Steady Growth: above potential (0-2% gap). Late Overheating: > potential (2%+ gap). Tightening: - . Early Recession: contracting (negative YoY). Late Recession: bottoming (near 0%). Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): -10%. Reflationary Deleveraging (6-12): growth > rates (GDP +2% > nominal rates). Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: growth > rates (GDP +2% > nominal rates). Normalization: = productivity (+2-3%). Geopolitical Rise (24-60+): per capita growth accelerating (+3% YoY). Top (36-120): - . Decline (24-72): - . Coincident Rise: share growing (10-20%). Top: - . Decline: share shrinking (<10%).",
    "Capacity utilization": "Mid Steady Growth (6-12): 75-80%. Late Overheating (6-18): >80%. Tightening (3-9): - . Early Recession (6-18): - . Late Recession (3-6): <70%. Coincident Early Recovery: - . Mid Steady Growth: 75-80%. Late Overheating: >80%. Tightening: - . Early Recession: - . Late Recession: <70%. Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): >80%. Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Inflation": "Mid Steady Growth (6-12): 2-3%. Late Overheating (6-18): >3% accelerating. Tightening (3-9): > target (e.g., >2% sustained). Early Recession (6-18): - . Late Recession (3-6): falling (<1%). Coincident Early Recovery: low (1-2%). Mid Steady Growth: stable (2-3%). Late Overheating: rising (3-4%). Tightening: > target (2-3%). Early Recession: - . Late Recession: falling (<1%). Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): >3%. Depression (3-18): deflation (<0%). Reflationary Deleveraging (6-12): bottoming (<1%). Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Retail sales": "Mid Steady Growth (6-12): rising (+3-5% YoY). Late Overheating (6-18): - . Tightening (3-9): - . Early Recession (6-18): slowdown (<1% YoY). Late Recession (3-6): - . Coincident Early Recovery: growing (+3% YoY). Mid Steady Growth: consistent (+3-4% YoY). Late Overheating: - . Tightening: - . Early Recession: declining (-1% YoY). Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Nonfarm payrolls": "Mid Steady Growth (6-12): +150K/month. Late Overheating (6-18): - . Tightening (3-9): - . Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: steady (+150K/month). Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Wage growth": "Late Overheating (6-18): rising >3% YoY. Tightening (3-9): - . Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: >3% YoY. Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "P/E ratios": "Late Overheating (6-18): high (20+). Tightening (3-9): - . Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): +20%. Top (6-24): 25+. Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: high (P/E >20). Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Credit growth": "Late Overheating (6-18): increasing >5% YoY. Tightening (3-9): slowing. Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): +5-10% YoY gap. Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Fed funds futures": "Tightening (3-9): implying hikes (+0.5%+). Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Short rates": "Tightening (3-9): rising. Early Recession (6-18): - . Late Recession (3-6): - . Coincident Early Recovery: - . Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: - . Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Industrial production": "Coincident Early Recovery: increasing (+2-5% YoY). Mid Steady Growth: - . Late Overheating: - . Tightening: - . Early Recession: falling (-2% YoY). Late Recession: stabilizing. Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Consumer/investment spending": "Coincident Early Recovery: positive. Mid Steady Growth: balanced. Late Overheating: high. Tightening: - . Early Recession: dropping. Late Recession: - . Long-Term Early Phase (12-36+): - . Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Productivity growth": "Long-Term Early Phase (12-36+): rising (>3% YoY). Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): rebound (+2% YoY). Coincident Early Phase: strong (+3% YoY). Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: +2-3%. Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Debt-to-GDP": "Long-Term Early Phase (12-36+): low (<60%). Debt Bubble (24-60): 60-100% rising fast (+20% in 3 years). Top (6-24): >100%. Depression (3-18): spiking (>120%). Reflationary Deleveraging (6-12): falling (-5% YoY). Normalization (12-36): stable/declining (60-80%). Coincident Early Phase: <60%. Debt Bubble: 60-100% rising fast. Top: >100%. Depression: spiking (>120%). Reflationary Deleveraging: falling (-5% YoY). Normalization: stable/declining (60-80%). Geopolitical Rise (24-60+): - . Top (36-120): climbing (+15% in 5 years). Decline (24-72): - . Coincident Rise: - . Top: high (>100% GDP). Decline: - .",
    "Foreign reserves": "Long-Term Early Phase (12-36+): increasing (+10% YoY). Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): falling (-10% YoY). Reflationary Deleveraging (6-12): - . Normalization (12-36): stabilizing. Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Real rates": "Long-Term Early Phase (12-36+): falling (<-1%). Debt Bubble (24-60): low (0-2%). Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): positive (>1%). Coincident Early Phase: positive (>0%). Debt Bubble: negative (<-1%). Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: positive (>1%). Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Trade balance": "Long-Term Early Phase (12-36+): improving (surplus >2% GDP). Debt Bubble (24-60): - . Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: - . Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Asset prices > traditional metrics": "Long-Term Debt Bubble (24-60): > traditional metrics (P/E +20%). Top (6-24): - . Depression (3-18): -50%. Reflationary Deleveraging (6-12): - . Normalization (12-36): positive (equities +10% YoY). Coincident Early Phase: - . Debt Bubble: high. Top: bubbles peaking (P/E 25+). Depression: -50%. Reflationary Deleveraging: - . Normalization: steady (+5-10% YoY). Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "New buyers entering (market participation)": "Long-Term Debt Bubble (24-60): +15%. Top (6-24): - . Depression (3-18): - . Reflationary Deleveraging (6-12): - . Normalization (12-36): - . Coincident Early Phase: - . Debt Bubble: bubble signs (+15% market participation). Top: - . Depression: - . Reflationary Deleveraging: - . Normalization: - . Geopolitical Rise (24-60+): - . Top (36-120): - . Decline (24-72): - . Coincident Rise: - . Top: - . Decline: - .",
    "Wealth gaps": "Long-Term Top (6-24): widening (top 1% share +5%). Depression (3-18): - . Reflationary Deleveraging (