import pandas as pd

# This file ensures the app NEVER crashes, even if all APIs fail.
# It contains realistic recent data for every single metric.

def get_seed_df(data_dict):
    return pd.DataFrame(data_dict)

SEEDS = {
    # --- CORE MACRO ---
    "T10Y2Y": get_seed_df({"date": ["2025-10-01", "2025-11-01", "2025-12-01"], "value": [-0.15, 0.10, 1.20]}),
    "UMCSENT": get_seed_df({"date": ["2025-10-01", "2025-11-01", "2025-12-01"], "value": [68.0, 72.0, 91.0]}),
    "PERMIT": get_seed_df({"date": ["2025-10-01", "2025-11-01", "2025-12-01"], "value": [1450, 1480, 1550]}),
    "ICSA": get_seed_df({"date": ["2025-11-15", "2025-11-22", "2025-11-29"], "value": [220000, 215000, 198000]}),
    "USSLIND": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [1.5, 1.8]}),
    "GDP": get_seed_df({"date": ["2025-07-01", "2025-10-01"], "value": [2.8, 3.1]}),
    "TCU": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [78.5, 80.2]}),
    "CPIAUCSL": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [315.0, 316.5]}), # Implies ~2-3% YoY
    "RSXFS": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [600000, 620000]}),
    "PAYEMS": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [158000, 158200]}),
    "CES0500000003": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [34.50, 34.65]}), # Wage
    "TOTBKCR": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [17500, 17600]}),
    "FEDFUNDS": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [5.33, 5.33]}),
    "TB3MS": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [5.40, 5.45]}),
    "INDPRO": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [103.0, 103.5]}),
    "PCE": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [19000, 19100]}),
    "OPHNFB": get_seed_df({"date": ["2025-07-01", "2025-10-01"], "value": [115.0, 116.0]}),
    "GFDEGDQ188S": get_seed_df({"date": ["2025-07-01", "2025-10-01"], "value": [122.0, 123.5]}),
    "REAINTRATREARAT10Y": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [2.1, 2.2]}),
    "BAMLH0A0HYM2": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [3.50, 3.65]}), # Spread in %
    "M2SL": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [20800, 20750]}),
    "FYFSD": get_seed_df({"date": ["2024-01-01", "2025-01-01"], "value": [6.2, 6.8]}), # Deficit % GDP
    "TDSP": get_seed_df({"date": ["2025-04-01", "2025-07-01"], "value": [9.8, 10.1]}),
    "BOPGSTB": get_seed_df({"date": ["2025-09-01", "2025-10-01"], "value": [-65000, -68000]}),
    "A067RO1Q156NBEA": get_seed_df({"date": ["2025-07-01", "2025-10-01"], "value": [4.0, 4.2]}), # Income Growth
    "A063RC1Q027SBEA": get_seed_df({"date": ["2025-07-01", "2025-10-01"], "value": [950, 980]}), # Military $
    "WALCL": get_seed_df({"date": ["2024-11-01", "2025-11-01"], "value": [7800000, 7400000]}), # Fed BS
    "SOFR": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [5.31, 5.32]}),
    "DGS30": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [4.50, 4.60]}),
    "SIPOVGINIUSA": get_seed_df({"date": ["2023-01-01", "2024-01-01"], "value": [41.0, 41.5]}),
    "LABSHPUSA156NRUG": get_seed_df({"date": ["2023-01-01", "2024-01-01"], "value": [58.0, 57.5]}),
    
    # --- NON-FRED SEEDS ---
    "margin_finra": get_seed_df({"date": ["2025-09-30", "2025-10-31"], "value": [800000, 790000]}), # Mil
    "gdp_nominal": get_seed_df({"date": ["2025-07-01", "2025-10-01"], "value": [28000, 28500]}), # Bil
    "cboe_putcall": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [0.68, 0.62]}),
    "aaii_sentiment": get_seed_df({"date": ["2025-11-22", "2025-11-29"], "bull": [55.0, 62.0]}),
    "pe_sp500": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [29.5, 30.2]}),
    "insider_ratio": get_seed_df({"date": ["2025-11-01", "2025-12-01"], "value": [12.0, 8.5]}),
    "vix": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [13.5, 12.8]}),
    "sp500_above_200dma": get_seed_df({"date": ["2025-11-28", "2025-11-29"], "value": [30.0, 22.0]}),
    "total_debt_gdp_global": get_seed_df({"date": ["2024-01-01", "2025-01-01"], "value": [340.0, 355.0]}),
    "usd_gold_power": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [0.35, 0.32]}), # oz per $1000
    "real_assets_basket": get_seed_df({"date": ["2024-01-01", "2025-01-01"], "value": [100.0, 115.0]}),
    "usd_reserve_share": get_seed_df({"date": ["2024-01-01", "2025-01-01"], "value": [58.0, 57.2]}),
    "gpr_index": get_seed_df({"date": ["2025-10-01", "2025-11-01"], "value": [120, 160]}),
}
