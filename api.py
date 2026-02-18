"""
NAU Quantum v4.0 — FastAPI Backend (Production-Ready)
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
from nau_quantum_engine import NAUQuantumAlphaIndicator

app = FastAPI(title="NAU Quantum v4.0 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INTERVAL_MAP = {
    "1m":  {"yf": "1m",  "period": "7d",   "resample": None},
    "5m":  {"yf": "5m",  "period": "60d",  "resample": None},
    "15m": {"yf": "15m", "period": "60d",  "resample": None},
    "30m": {"yf": "30m", "period": "60d",  "resample": None},
    "1h":  {"yf": "1h",  "period": "730d", "resample": None},
    "4h":  {"yf": "1h",  "period": "730d", "resample": "4h"},
    "1d":  {"yf": "1d",  "period": "5y",   "resample": None},
    "1wk": {"yf": "1wk", "period": "10y",  "resample": None},
    "1mo": {"yf": "1mo", "period": "max",  "resample": None},
    "3mo": {"yf": "3mo", "period": "max",  "resample": None},
}

indicator = NAUQuantumAlphaIndicator()


def safe_download(sym, period, interval, prepost):
    """Robust yfinance download with MultiIndex handling."""
    try:
        raw = yf.download(sym, period=period, interval=interval,
                          prepost=prepost, auto_adjust=True, progress=False)
    except Exception as e:
        return None, str(e)

    if raw is None or raw.empty:
        return None, "Empty result"

    # Handle MultiIndex columns (yfinance >= 0.2.31)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Remove duplicate columns
    raw = raw.loc[:, ~raw.columns.duplicated(keep='first')]

    # Standardize column names
    rename = {}
    for c in raw.columns:
        cl = str(c).lower().strip()
        if cl == 'open': rename[c] = 'Open'
        elif cl == 'high': rename[c] = 'High'
        elif cl == 'low': rename[c] = 'Low'
        elif cl in ('close', 'adj close'): rename[c] = 'Close'
        elif cl == 'volume': rename[c] = 'Volume'
    raw = raw.rename(columns=rename)

    # Check required columns
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        return None, f"Missing columns: {missing}"

    df = raw[needed].copy()

    # Force numeric - handle Series that might be DataFrames
    for col in needed:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        df[col] = pd.to_numeric(s, errors='coerce')

    df['Volume'] = df['Volume'].fillna(0)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    return df, None


def download_and_compute(sym, interval, prepost=False):
    config = INTERVAL_MAP.get(interval)
    if not config:
        return {"error": f"Invalid interval: {interval}"}

    df, err = safe_download(sym, config["period"], config["yf"], prepost)
    if err:
        return {"error": f"Download failed for {sym}: {err}"}
    if df is None or df.empty:
        return {"error": f"No data for {sym}. Verify ticker exists on Yahoo Finance."}

    # Resample if needed (e.g. 1h -> 4h)
    if config["resample"]:
        df = df.resample(config["resample"]).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"
        }).dropna()

    if len(df) < 50:
        return {"error": f"Only {len(df)} bars for {sym} on {interval}. Need 50+."}

    # Compute indicator
    try:
        df = indicator.compute(df)
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": f"Indicator error: {str(e)} | {tb[-500:]}"}

    # Build JSON
    bars, signals = [], []
    ind_cols = ["NAU_Signal","NAU_Confidence","NAU_Regime","NAU_Kalman",
                "NAU_Kalman_Score","NAU_Wavelet_Score","NAU_HMM_Score",
                "NAU_Entropy_Score","NAU_Hurst_Score","NAU_Fractal_Score",
                "NAU_OB_Score","NAU_FVG_Score","NAU_Structure_Score",
                "NAU_Williams_Score","NAU_Attention_Score","NAU_RL_Score",
                "NAU_DeepRegime_Score","NAU_OrderFlow_Score",
                "NAU_MicroStructure_Score","NAU_MTF_Score"]

    for idx, row in df.iterrows():
        try:
            ts = int(idx.timestamp())
        except:
            ts = int(pd.Timestamp(idx).timestamp())

        bar = {
            "time": ts,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(float(row["Volume"])),
        }
        for col in ind_cols:
            if col in df.columns:
                v = row.get(col)
                if v is not None and pd.notna(v):
                    bar[col] = round(float(v), 4)
        bars.append(bar)

        if row.get("NAU_Long", False):
            signals.append({"time":ts,"type":"LONG","price":bar["close"],
                           "confidence":round(float(row["NAU_Confidence"])*100,1),
                           "signal_score":round(float(row["NAU_Signal"]),1)})
        if row.get("NAU_Short", False):
            signals.append({"time":ts,"type":"SHORT","price":bar["close"],
                           "confidence":round(float(row["NAU_Confidence"])*100,1),
                           "signal_score":round(float(row["NAU_Signal"]),1)})

    # SL/TP
    sl_tp = None
    if signals:
        last_sig = signals[-1]
        atr = np.mean([bars[i]["high"]-bars[i]["low"] for i in range(max(0,len(bars)-14),len(bars))])
        e = last_sig["price"]
        if last_sig["type"] == "LONG":
            sl_tp = {"type":"LONG","entry":e,"sl":round(e-1.5*atr,4),
                     "tp1":round(e+2*atr,4),"tp2":round(e+3*atr,4),"tp3":round(e+4.5*atr,4),"atr":round(atr,4)}
        else:
            sl_tp = {"type":"SHORT","entry":e,"sl":round(e+1.5*atr,4),
                     "tp1":round(e-2*atr,4),"tp2":round(e-3*atr,4),"tp3":round(e-4.5*atr,4),"atr":round(atr,4)}

    last = df.iloc[-1]
    summary = {
        "symbol":sym,"interval":interval,"bars_count":len(bars),
        "signal":round(float(last["NAU_Signal"]),1),
        "confidence":round(float(last["NAU_Confidence"])*100,1),
        "regime":int(last["NAU_Regime"]),
        "regime_name":{0:"BULL",1:"BEAR",2:"RANGE"}.get(int(last["NAU_Regime"]),"RANGE"),
        "total_long":int(df["NAU_Long"].sum()),
        "total_short":int(df["NAU_Short"].sum()),
        "last_price":round(float(last["Close"]),4),
    }
    return {"bars":bars,"signals":signals,"sl_tp":sl_tp,"summary":summary}


@app.get("/api/chart")
def get_chart(symbol:str=Query("PLTR"), interval:str=Query("1d"), prepost:bool=Query(False)):
    try:
        return download_and_compute(symbol.upper().strip(), interval, prepost)
    except Exception as e:
        return {"error":f"Server error: {str(e)}"}

@app.get("/api/scan")
def scan_stocks(interval:str=Query("1d"), min_confidence:float=Query(65)):
    watchlist = {
        "Technology":["AAPL","MSFT","NVDA","GOOGL","META","TSLA","PLTR","AMD","CRM","NFLX","MU","INTC","AVGO","QCOM"],
        "Finance":["JPM","BAC","GS","V","MA","COIN","PYPL"],
        "Healthcare":["JNJ","UNH","LLY","ABBV","PFE"],
        "Energy":["XOM","CVX","OXY"],
        "Consumer":["WMT","COST","DIS","MCD","KO","HD","NKE"],
        "ETFs":["SPY","QQQ","IWM","SOXX","ARKK","XLF","XLE"],
        "Crypto":["BTC-USD","ETH-USD","SOL-USD"],
    }
    results = []
    for sector, symbols in watchlist.items():
        for sym in symbols:
            try:
                data = download_and_compute(sym, interval, False)
                if "error" in data: continue
                s = data["summary"]
                if s["confidence"] >= min_confidence and abs(s["signal"]) > 15:
                    results.append({"symbol":sym,"sector":sector,"signal":s["signal"],
                        "confidence":s["confidence"],"regime":s["regime_name"],
                        "direction":"LONG" if s["signal"]>0 else "SHORT",
                        "price":s["last_price"],"sl_tp":data["sl_tp"],
                        "score":round(abs(s["signal"])*(s["confidence"]/100),1)})
            except: continue
    results.sort(key=lambda x:x["score"],reverse=True)
    return {"scan_results":results[:20],"total_scanned":sum(len(v) for v in watchlist.values())}

@app.get("/api/health")
def health():
    return {"status":"ok","version":"4.0","engine":"18-Factor AI/ML"}

@app.get("/")
def root():
    return {"message":"NAU Quantum v4.0 API","endpoints":["/api/health","/api/chart","/api/scan","/docs"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
