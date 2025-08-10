import os, io, time, requests, pandas as pd, wbdata
from datetime import datetime

OUT_WB   = "data/wb"
OUT_FRED = "data/fred"
OUT_ROOT = "data"
os.makedirs(OUT_WB, exist_ok=True)
os.makedirs(OUT_FRED, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

SES = requests.Session()
SES.headers.update({"User-Agent": "EconMirror/MirrorBot/1.3"})

def get_json(url, tries=3, timeout=45):
    for i in range(tries):
        try:
            r = SES.get(url, timeout=timeout); r.raise_for_status(); return r.json()
        except Exception:
            if i==tries-1: raise
            time.sleep(2)

def get_text(url, tries=3, timeout=45):
    for i in range(tries):
        try:
            r = SES.get(url, timeout=timeout); r.raise_for_status(); return r.text
        except Exception:
            if i==tries-1: raise
            time.sleep(2)

def safe_to_csv(df, path):
    if df is None or df.empty: return False
    df.to_csv(path, index=False, encoding="utf-8"); return True

def write_seed(path, flag_path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    with open(flag_path, "w", encoding="utf-8") as f: f.write("seed")
    print(f"[SEED] {path} rows={len(df)}")

# -------------------- World Bank mirrors --------------------
WB_CODES = [
    "SI.POV.GINI","SE.XPD.TOTL.GD.ZS","IP.PAT.RESD","NY.GDP.PCAP.KD.ZG","NE.TRD.GNFS.ZS",
    "MS.MIL.XPND.GD.ZS","SP.POP.1564.TO.ZS","GB.XPD.RSDV.GD.ZS","CC.EST","PV.EST","LP.LPI.OVRL.XQ",
    "NY.GDP.MKTP.CD","NE.EXP.GNFS.CD"
]
WB_COUNTRIES = ["USA","WLD"]

def save_wb(country, code):
    try:
        df = wbdata.get_dataframe({code:"val"}, country=country).dropna()
        if df.empty: return
        df.index = pd.to_datetime(df.index); df = df.sort_index()
        out = df.reset_index().rename(columns={"index":"date"}); out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        path = os.path.join(OUT_WB, f"{country}_{code}.csv")
        safe_to_csv(out, path); print(f"[WB] {path} rows={len(out)}")
    except Exception as e:
        print(f"[WB] {country} {code} error: {e}")

def fetch_all_wb():
    for code in WB_CODES:
        for c in WB_COUNTRIES: save_wb(c, code)

# -------------------- FRED mirrors (public CSV) --------------------
FRED_SERIES = [
    "M2SL","CPIAUCSL","PAYEMS","RSXFS","TCU","FEDFUNDS","TB3MS","INDPRO","PCE",
    "AHETPI","TOTBKCR","DTWEXBGS","BOPGSTB","GFDEBTN","TDSP","ICSA","PERMIT","USSLIND","OPHNFB"
]

def save_fred_csv(series_id):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        txt = get_text(url, tries=3, timeout=60)
        path = os.path.join(OUT_FRED, f"{series_id}.csv")
        with open(path, "w", encoding="utf-8") as f: f.write(txt)
        print(f"[FRED] {path} rows={max(0, len(txt.splitlines())-1)}"); time.sleep(0.35)
    except Exception as e:
        print(f"[FRED] {series_id} error: {e}")

def fetch_all_fred():
    for sid in FRED_SERIES: save_fred_csv(sid)

# -------------------- OECD PISA (Math mean, USA) --------------------
def fetch_pisa_math_usa():
    path = os.path.join(OUT_ROOT,"pisa_math_usa.csv")
    flag = path + ".SEED"
    try:
        js = get_json("https://stats.oecd.org/sdmx-json/data/PISA_2022/MATH.MEAN.USA.A.T?_format=json", tries=2, timeout=60)
        ser = js.get("dataSets",[{}])[0].get("series",{})
        if ser:
            k = next(iter(ser)); ob = ser[k].get("observations",{})
            years = sorted([int(y) for y in ob.keys()]); vals  = [float(ob[str(y)][0]) for y in years]
            df = pd.DataFrame({"year": years, "pisa_math_mean_usa": vals})
            safe_to_csv(df, path); print(f"[OECD] {path} rows={len(df)}"); 
            if os.path.exists(flag): os.remove(flag)
            return
    except Exception as e:
        print(f"[OECD] 2022 error: {e}")
    try:
        js = get_json("https://stats.oecd.org/sdmx-json/data/PISA_2018/MATH.MEAN.USA.A.T?_format=json", tries=2, timeout=60)
        ser = js.get("dataSets",[{}])[0].get("series",{})
        if ser:
            k = next(iter(ser)); ob = ser[k].get("observations",{})
            years = sorted([int(y) for y in ob.keys()]); vals  = [float(ob[str(y)][0]) for y in years]
            df = pd.DataFrame({"year": years, "pisa_math_mean_usa": vals})
            safe_to_csv(df, path); print(f"[OECD] {path} rows={len(df)} (2018)"); 
            if os.path.exists(flag): os.remove(flag)
            return
    except Exception as e:
        print(f"[OECD] 2018 error: {e}")
    # Seed (clearly labeled by .SEED)
    seed = pd.DataFrame({
        "year":[2003,2006,2009,2012,2015,2018,2022],
        "pisa_math_mean_usa":[483,474,487,481,470,478,465]
    })
    write_seed(path, flag, seed)

# -------------------- CINC (USA) --------------------
def fetch_cinc_usa():
    path = os.path.join(OUT_ROOT,"cinc_usa.csv"); flag = path + ".SEED"
    urls = [
        "https://raw.githubusercontent.com/prio-data/nmc/main/output/cinc.csv",
        "https://raw.githubusercontent.com/correlatesofwar/CorrelatesofWar_Project/master/NMC/NMC_60_abridged.csv",
        "https://raw.githubusercontent.com/correlatesofwar/CorrelatesofWar_Project/master/NMC/NMC_60.csv"
    ]
    for u in urls:
        try:
            txt = get_text(u, tries=2, timeout=60)
            df  = pd.read_csv(io.StringIO(txt))
            lower = {c.lower(): c for c in df.columns}
            abb = lower.get("stateabb") or lower.get("ccodealp") or next((c for lc,c in lower.items() if "abb" in lc), None)
            year = lower.get("year") or next((c for lc,c in lower.items() if lc.endswith("year")), None)
            cinc = lower.get("cinc") or next((c for lc,c in lower.items() if "cinc" in lc), None)
            if not (abb and year and cinc): continue
            sdf = df[df[abb].astype(str).str.upper()=="USA"][[year, cinc]].rename(columns={year:"year", cinc:"cinc_usa"}).dropna()
            sdf["year"] = pd.to_numeric(sdf["year"], errors="coerce"); sdf = sdf.dropna().sort_values("year")
            if safe_to_csv(sdf, path):
                print(f"[CINC] {path} rows={len(sdf)} from {u}")
                if os.path.exists(flag): os.remove(flag)
                return
        except Exception:
            continue
    seed = pd.DataFrame({"year":[2000,2010,2020,2023],"cinc_usa":[0.14,0.12,0.10,0.10]})
    write_seed(path, flag, seed)

# -------------------- UCDP (battle-related deaths, World) --------------------
def fetch_ucdp_battle_deaths_global():
    path = os.path.join(OUT_ROOT,"ucdp_battle_deaths_global.csv"); flag = path + ".SEED"
    try:
        url = "https://raw.githubusercontent.com/owid/owid-data/master/datasets/Conflict%20and%20battle-related%20deaths/Conflict%20and%20battle-related%20deaths.csv"
        txt = get_text(url, tries=2, timeout=60)
        df  = pd.read_csv(io.StringIO(txt))
        lower = {c.lower(): c for c in df.columns}
        ent = lower.get("entity") or lower.get("country") or "Entity"
        yr  = lower.get("year") or "Year"
        cand = [c for c in df.columns if "battle" in c.lower() and "death" in c.lower()]
        if not cand: raise ValueError("deaths column not found")
        vcol = cand[0]
        sdf = df[df[ent].astype(str)=="World"][[yr, vcol]].rename(columns={yr:"year", vcol:"ucdp_battle_deaths_global"}).dropna()
        sdf["year"] = pd.to_numeric(sdf["year"], errors="coerce"); sdf = sdf.dropna().sort_values("year")
        if safe_to_csv(sdf, path):
            print(f"[UCDP] {path} rows={len(sdf)}")
            if os.path.exists(flag): os.remove(flag)
            return
    except Exception as e:
        print(f"[UCDP] error: {e}")
    seed = pd.DataFrame({"year":[2010,2014,2016,2020,2022],"ucdp_battle_deaths_global":[56000,100000,120000,90000,110000]})
    write_seed(path, flag, seed)

# -------------------- IMF COFER (USD share of allocated reserves) --------------------
def fetch_imf_cofer_usd_share():
    path = os.path.join(OUT_ROOT,"imf_cofer_usd_share.csv"); flag = path + ".SEED"
    try:
        js = get_json("https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/COFER/D.USD.A", tries=3, timeout=60)
        series = js.get("CompactData",{}).get("DataSet",{}).get("Series")
        obs = None
        if isinstance(series, dict): obs = series.get("Obs", [])
        elif isinstance(series, list) and series: obs = series[0].get("Obs", [])
        rows=[]
        for o in (obs or []):
            t = o.get("@TIME_PERIOD") or o.get("TIME_PERIOD"); v = o.get("@OBS_VALUE") or o.get("OBS_VALUE")
            if t and v: rows.append((t, float(v)))
        if rows:
            df = pd.DataFrame(rows, columns=["date","usd_share"])
            safe_to_csv(df, path); print(f"[IMF] {path} rows={len(df)}")
            if os.path.exists(flag): os.remove(flag)
            return
    except Exception as e:
        print(f"[IMF] error: {e}")
    seed = pd.DataFrame({"date":["2023-Q4","2024-Q1","2024-Q2","2024-Q3","2024-Q4"],"usd_share":[59.4,59.8,59.2,58.9,59.1]})
    write_seed(path, flag, seed)

# -------------------- S&P500 Trailing P/E --------------------
def fetch_sp500_pe():
    path = os.path.join(OUT_ROOT,"pe_sp500.csv"); flag = path + ".SEED"
    try:
        txt = get_text("https://www.multpl.com/s-p-500-pe-ratio/table/by-month?format=csv", tries=2, timeout=60)
        first = txt.splitlines()[0].strip().lower()
        if not (first.startswith("date") or first.startswith("month")): raise ValueError("MULTPL returned non-CSV")
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip().lower() for c in df.columns]
        dcol = [c for c in df.columns if "date" in c or "month" in c][0]
        vcol = [c for c in df.columns if "value" in c or c=="pe"][0]
        out = df.rename(columns={dcol:"date", vcol:"pe"})[["date","pe"]].dropna()
        out["date"] = pd.to_datetime(out["date"], errors="coerce"); out = out.dropna().sort_values("date")
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        safe_to_csv(out, path); print(f"[PE] {path} rows={len(out)} (MULTPL)")
        if os.path.exists(flag): os.remove(flag)
        return
    except Exception as e:
        print(f"[PE] MULTPL error: {e}")
    try:
        xls_url = "https://www.econ.yale.edu/~shiller/data/ie_data.xls"
        content = SES.get(xls_url, timeout=60).content
        xdf = pd.read_excel(io.BytesIO(content), sheet_name=0, skiprows=7)
        colmap = {c.lower(): c for c in xdf.columns}
        datec = next((colmap[k] for k in colmap if "date" in k or ("year" in k and "month" in k)), None) or "Date"
        price = next((colmap[k] for k in colmap if k.strip()=="p" or "price" in k), None)
        earn  = next((colmap[k] for k in colmap if k.strip()=="e" or "earn" in k), None)
        df = xdf[[datec, price, earn]].dropna()
        df["date"] = pd.to_datetime(df[datec], errors="coerce"); df = df.dropna()
        df["pe"] = df[price] / df[earn]
        out = df[["date","pe"]].sort_values("date"); out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        safe_to_csv(out, path); print(f"[PE] {path} rows={len(out)} (Shiller)")
        if os.path.exists(flag): os.remove(flag)
        return
    except Exception as e:
        print(f"[PE] Shiller error: {e}")
    seed = pd.DataFrame({"date":["2024-12-31","2025-03-31","2025-06-30"],"pe":[24.0,25.2,25.8]})
    write_seed(path, flag, seed)

if __name__ == "__main__":
    print("== WB mirrors =="); fetch_all_wb()
    print("== FRED mirrors =="); fetch_all_fred()
    print("== OECD PISA mirror =="); fetch_pisa_math_usa()
    print("== CINC mirror =="); fetch_cinc_usa()
    print("== UCDP mirror =="); fetch_ucdp_battle_deaths_global()
    print("== IMF COFER USD share =="); fetch_imf_cofer_usd_share()
    print("== S&P500 P/E =="); fetch_sp500_pe()
    print("Done.")
