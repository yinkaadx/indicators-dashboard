#!/usr/bin/env python
import os
import sys
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "update_mirrors.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EconMirror/Updater"})


def safe_write_csv(path: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        logging.warning("Not writing %s: empty dataframe", path)
        return
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    logging.info("Wrote %s (%d rows)", path, len(df))


def get_fred() -> Optional["Fred"]:
    if not FRED_API_KEY:
        logging.warning("FRED_API_KEY not set; FRED-based updates will be skipped.")
        return None
    if Fred is None:
        logging.warning("fredapi not installed; FRED-based updates will be skipped.")
        return None
    return Fred(api_key=FRED_API_KEY)


def update_margin_finra() -> None:
    """
    margin_finra.csv
    Columns: date,debit_bil
    Strategy:
      - Try FINRA margin statistics page and read the first numeric table.
      - Take the LAST row as latest observation.
    Note: HTML structure may change; if parsing fails, we keep the old CSV.
    """
    url = "https://www.finra.org/rules-guidance/key-topics/margin-statistics"
    path = os.path.join(DATA_DIR, "margin_finra.csv")
    logging.info("Updating FINRA margin debt from %s", url)
    try:
        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            raise ValueError("No tables found on FINRA margin page")
        df_raw = tables[0]
        df_raw = df_raw.dropna(how="all")
        if df_raw.empty:
            raise ValueError("FINRA table empty after dropna")

        # Heuristic: assume first column is date, last numeric column is debit balances.
        date_col = df_raw.columns[0]
        numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            # Try to coerce all except first column to numeric
            for col in df_raw.columns[1:]:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Could not find numeric debit column in FINRA table")

        debit_col = numeric_cols[-1]
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.dropna(subset=[date_col, debit_col]).sort_values(date_col)
        latest = df_raw.iloc[[-1]][[date_col, debit_col]].copy()
        latest.columns = ["date", "debit_bil"]
        latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
        # FINRA usually reports in millions; if values look too big, rescale.
        if latest["debit_bil"].iloc[0] > 1e5:
            latest["debit_bil"] = latest["debit_bil"] / 1000.0
        safe_write_csv(path, latest)
    except Exception as e:
        logging.error("Failed to update margin_finra.csv: %s", e)


def update_us_gdp_nominal() -> None:
    """
    us_gdp_nominal.csv
    Columns: date,gdp_trillions
    Uses FRED GDP series as proxy for nominal GDP (billions → trillions).
    """
    path = os.path.join(DATA_DIR, "us_gdp_nominal.csv")
    fred = get_fred()
    if fred is None:
        logging.warning("Skipping us_gdp_nominal.csv (no FRED).")
        return
    try:
        logging.info("Updating US GDP from FRED series GDP.")
        s = fred.get_series("GDP")
        df = s.to_frame(name="gdp_billions")
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        df = df.reset_index().rename(columns={"index": "date"})
        df["gdp_trillions"] = df["gdp_billions"] / 1000.0
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        safe_write_csv(path, df[["date", "gdp_trillions"]].tail(8))
    except Exception as e:
        logging.error("Failed to update us_gdp_nominal.csv: %s", e)


def update_cboe_putcall() -> None:
    """
    cboe_putcall.csv
    Columns: date,pc_ratio
    Uses CBOE official delayed CSV.
    """
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/totalpc.csv"
    path = os.path.join(DATA_DIR, "cboe_putcall.csv")
    logging.info("Updating CBOE put/call from %s", url)
    try:
        df = pd.read_csv(url, skiprows=2)
        if df.empty:
            raise ValueError("CBOE CSV empty")
        # Heuristic: first row, second column is total put/call.
        ratio = float(df.iloc[0, 1])
        today = datetime.utcnow().strftime("%Y-%m-%d")
        out = pd.DataFrame([{"date": today, "pc_ratio": ratio}])
        safe_write_csv(path, out)
    except Exception as e:
        logging.error("Failed to update cboe_putcall.csv: %s", e)


def update_aaii_sentiment() -> None:
    """
    aaii_sentiment.csv
    Columns: date,bull,bear,neutral
    Uses AAII's official sentiment CSV.
    """
    url = "https://www.aaii.com/files/surveys/sentiment.csv"
    path = os.path.join(DATA_DIR, "aaii_sentiment.csv")
    logging.info("Updating AAII sentiment from %s", url)
    try:
        df = pd.read_csv(url)
        # AAII usually uses 'Survey Date' and 'Bullish', 'Bearish', 'Neutral'
        date_col = "Survey Date" if "Survey Date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        for col in ["Bullish", "Bearish", "Neutral"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.rstrip("%")
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df_out = df[[date_col, "Bullish", "Bearish", "Neutral"]].copy()
        df_out.columns = ["date", "bull", "bear", "neutral"]
        df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
        safe_write_csv(path, df_out.tail(52))
    except Exception as e:
        logging.error("Failed to update aaii_sentiment.csv: %s", e)


def update_insider_ratio() -> None:
    """
    insider_ratio.csv
    Columns: date,buy_ratio_pct
    NOTE: OpenInsider HTML layout changes often and may block bots.
    This function is a template; if parsing fails, you should update the CSV manually.
    """
    url = "http://openinsider.com/"
    path = os.path.join(DATA_DIR, "insider_ratio.csv")
    logging.info("Attempting to update insider_ratio.csv from %s", url)
    try:
        import bs4  # type: ignore

        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
        soup = bs4.BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if table is None:
            raise ValueError("No table found on OpenInsider front page")
        df = pd.read_html(str(table))[0]
        # Heuristic: look for column containing 'Buy/Sell Ratio' or '% Buy'
        target_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if "buy" in col_str and ("ratio" in col_str or "%" in col_str):
                target_col = col
                break
        if target_col is None:
            raise ValueError("Could not find buy-ratio column in OpenInsider table")
        # Assume last row is latest summary
        val = pd.to_numeric(df[target_col].iloc[-1], errors="coerce")
        if pd.isna(val):
            raise ValueError("Parsed insider ratio is NaN")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        out = pd.DataFrame([{"date": today, "buy_ratio_pct": float(val)}])
        safe_write_csv(path, out)
    except Exception as e:
        logging.error(
            "Failed to auto-update insider_ratio.csv: %s. "
            "You may need to update this file manually.", e
        )


def update_total_debt_gdp() -> None:
    """
    total_debt_gdp.csv
    Columns: date,total_debt_gdp_pct
    This is highly composite (public + private + foreign). There is no single free API.
    Here we only ensure the file exists; adjust manually with your own research.
    """
    path = os.path.join(DATA_DIR, "total_debt_gdp.csv")
    if os.path.exists(path):
        logging.info("total_debt_gdp.csv already exists; leaving as-is.")
        return
    logging.warning(
        "total_debt_gdp.csv does not exist. Creating a placeholder; "
        "please update it manually with your own numbers."
    )
    today = datetime.utcnow().strftime("%Y-%m-%d")
    df = pd.DataFrame([{"date": today, "total_debt_gdp_pct": 355.0}])
    safe_write_csv(path, df)


def update_gpr_us() -> None:
    """
    gpr_us.csv
    Columns: date,gpr
    PolicyUncertainty provides GPR data files, but URLs and formats can change.
    This function is a template; if it fails, download their latest GPR US data
    manually and save as data/gpr_us.csv with date,gpr columns.
    """
    # This URL may need adjustment if PolicyUncertainty updates their site.
    url = "https://www.policyuncertainty.com/media/GPR_US.csv"
    path = os.path.join(DATA_DIR, "gpr_us.csv")
    logging.info("Attempting to update GPR US from %s", url)
    try:
        df = pd.read_csv(url)
        # Heuristic: first column date, last numeric column is GPR
        date_col = df.columns[0]
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric GPR column found")
        value_col = numeric_cols[-1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
        df_out = df[[date_col, value_col]].copy()
        df_out.columns = ["date", "gpr"]
        df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
        safe_write_csv(path, df_out.tail(120))
    except Exception as e:
        logging.error(
            "Failed to auto-update gpr_us.csv: %s. "
            "You may need to download and save this file manually.", e
        )


def update_real_assets_basket() -> None:
    """
    real_assets_basket.csv
    Columns: date,index
    Template for a composite (gold + oil + BTC + farmland proxy).
    Because reliable free history for all components is tricky, this function
    only ensures the file exists. You can replace it with your own logic.
    """
    path = os.path.join(DATA_DIR, "real_assets_basket.csv")
    if os.path.exists(path):
        logging.info("real_assets_basket.csv already exists; leaving as-is.")
        return
    logging.warning(
        "real_assets_basket.csv does not exist. Creating a flat placeholder index; "
        "replace with your own real-assets composite when ready."
    )
    today = datetime.utcnow().strftime("%Y-%m-%d")
    df = pd.DataFrame([{"date": today, "index": 100.0}])
    safe_write_csv(path, df)


def main() -> None:
    logging.info("=== Econ Mirror mirror updater starting ===")
    update_margin_finra()
    update_us_gdp_nominal()
    update_cboe_putcall()
    update_aaii_sentiment()
    update_insider_ratio()
    update_total_debt_gdp()
    update_gpr_us()
    update_real_assets_basket()
    logging.info("=== Econ Mirror mirror updater finished ===")


if __name__ == "__main__":
    main()
