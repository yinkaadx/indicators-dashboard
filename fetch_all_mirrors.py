import os, io, time, json, zipfile, requests, pandas as pd, wbdata

OUT_WB = "data/wb"
OUT_FRED = "data/fred"
OUT_ROOT = "data"
os.makedirs(OUT_WB, exist_ok=True)
os.makedirs(OUT_FRED, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

SES = requests.Session()
SES.headers.update({"User-Agent": "EconMirror/MirrorBot"})

# ---------------- World Bank ----------------
WB_CODES = [
    "SI.POV.GINI","SE.XPD.TOTL.GD.ZS","IP.PAT.RESD","NY.GDP.PCAP.KD.ZG","NE.TRD.GNFS.ZS",
    "MS.MIL.XPND.GD.ZS","SP.POP.1564.TO.ZS","GB.XPD.RSDV.GD.ZS","CC.EST","PV.EST","LP.LPI.OVRL.XQ",
    "NY.GDP.MKTP.CD","NE.EXP.GNFS.CD"
]
WB_COUNTRIES = ["USA","WLD"]

def save_wb(country, code):
    df = wbdata.get_dataframe({code:"val"}, country=country).dropna()
    if df.empty: return
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    out = df.reset_index().rename(columns={"index":"date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    path = os.path.join(OUT_WB, f"{country}_{code}.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"[WB] {path} rows={len(out)}")

def fetch_all_wb():
    for code in WB_CODES:
        for c in WB_COUNTRIES:
            try:
                save_wb(c, code)
            except Exception as e:
                print(f"[WB] {c} {code} error: {e}")

# ---------------- FRED (CSV endpoint) ----------------
FRED_SERIES = [
    "M2SL","CPIAUCSL","PAYEMS","RSXFS","TCU","FEDFUNDS","TB3MS","INDPRO","PCE",
    "AHETPI","TOTBKCR","DTWEXBGS","BOPGSTB","GFDEBTN","TDSP"
]

def save_fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = SES.get(url, timeout=30)
    r.raise_for_status()
    path = os.path.join(OUT_FRED, f"{series_id}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"[FRED] {path} rows={max(0, len(r.text.splitlines())-1)}")

def fetch_all_fred():
    for sid in FRED_SERIES:
        try:
            save_fred_csv(sid)
            time.sleep(0.5)
        except Exception as e:
            print(f"[FRED] {sid} error: {e}")

# ---------------- OECD PISA (Math mean, USA) ----------------
def fetch_pisa_math_usa():
    urls = [
        "https://stats.oecd.org/sdmx-json/data/PISA_2022/MATH.MEAN.USA.A.T?_format=json",
        "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/PISA%20scores%20(OWID%20repack)/PISA%20scores%20(OWID%20repack).csv"
    ]
    # Try OECD SDMX first
    try:
        js = SES.get(urls[0], timeout=30).json()
        ser = js.get("dataSets",[{}])[0].get("series",{})
        if ser:
            k = next(iter(ser)); ob = ser[k].get("observations",{})
            years = sorted(map(int, ob.keys()))
            vals = [ob[str(y)][0] for y in years]
            df = pd.DataFrame({"year": years, "pisa_math_mean_usa": vals})
            df.to_csv(os.path.join(OUT_ROOT, "pisa_math_usa.csv"), index=False, encoding="utf-8")
            print(f"[OECD] data/pisa_math_usa.csv rows={len(df)}")
            return
    except Exception as e:
        print(f"[OECD] primary error: {e}")
    # Fallback: OWID repack
    try:
        df = pd.read_csv(io.StringIO(SES.get(urls[1], timeout=30).text))
        df = df[(df["Entity"]=="United States") & (df["Indicator"].str.contains("Math", case=False, na=False))]
        df = df.rename(columns={"Year":"year","Value":"pisa_math_mean_usa"})[["year","pisa_math_mean_usa"]].dropna()
        df.to_csv(os.path.join(OUT_ROOT, "pisa_math_usa.csv"), index=False, encoding="utf-8")
        print(f"[OECD/OWID] data/pisa_math_usa.csv rows={len(df)}")
    except Exception as e:
        print(f"[OECD] fallback error: {e}")

# ---------------- CINC (USA) ----------------
def fetch_cinc_usa():
    urls = ["https://raw.githubusercontent.com/prio-data/nmc/main/output/cinc.csv"]
    for u in urls:
        try:
            txt = SES.get(u, timeout=45).text
            df = pd.read_csv(io.StringIO(txt))
            lower = {c.lower(): c for c in df.columns}
            abb = lower.get("stateabb") or lower.get("ccodealp") or "stateabb"
            year = lower.get("year") or "year"
            cinc = lower.get("cinc") or "cinc"
            if abb in df.columns and year in df.columns and cinc in df.columns:
                out = df[df[abb].astype(str).str.upper()=="USA"][[year, cinc]].rename(columns={year:"year", cinc:"cinc_usa"}).dropna()
                out.to_csv(os.path.join(OUT_ROOT,"cinc_usa.csv"), index=False, encoding="utf-8")
                print(f"[CINC] data/cinc_usa.csv rows={len(out)}")
                return
        except Exception as e:
            print(f"[CINC] error: {e}")

# ---------------- UCDP (battle-related deaths, World) ----------------
def fetch_ucdp_battle_deaths_global():
    try:
        url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Conflict%20and%20battle-related%20deaths/Conflict%20and%20battle-related%20deaths.csv"
        df = pd.read_csv(io.StringIO(SES.get(url, timeout=45).text))
        vcol = next(c for c in df.columns if "Battle" in c and "deaths" in c)
        out = df[df["Entity"]=="World"][["Year", vcol]].rename(columns={"Year":"year", vcol:"ucdp_battle_deaths_global"}).dropna()
        out.to_csv(os.path.join(OUT_ROOT,"ucdp_battle_deaths_global.csv"), index=False, encoding="utf-8")
        print(f"[UCDP] data/ucdp_battle_deaths_global.csv rows={len(out)}")
    except Exception as e:
        print(f"[UCDP] error: {e}")

if __name__ == "__main__":
    print("== WB mirrors =="); fetch_all_wb()
    print("== FRED mirrors =="); fetch_all_fred()
    print("== OECD PISA mirror =="); fetch_pisa_math_usa()
    print("== CINC mirror =="); fetch_cinc_usa()
    print("== UCDP mirror =="); fetch_ucdp_battle_deaths_global()
    print("Done.")
