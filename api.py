"""
NAU Quantum v4.0 — FastAPI Backend
Serves indicator data, stock scanning, and SL/TP calculations.
Deploy free on Render.com
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
from nau_quantum_engine import NAUQuantumAlphaIndicator

app = FastAPI(title="NAU Quantum v4.0 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PERIOD_MAP = {
    "1m":"7d","2m":"60d","5m":"60d","15m":"60d","30m":"60d",
    "1h":"730d","4h":"730d","1d":"5y","1wk":"10y","1mo":"max","3mo":"max",
}

# ── Reusable indicator instance ──
indicator = NAUQuantumAlphaIndicator()

def download_and_compute(sym: str, interval: str, prepost: bool = False):
    """Download OHLCV, compute indicator, return JSON-ready dict."""
    period = PERIOD_MAP.get(interval, "60d")
    raw = yf.download(sym, period=period, interval=interval, prepost=prepost, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    cmap = {}
    for c in raw.columns:
        lc = str(c).lower()
        if lc == "open": cmap[c] = "Open"
        elif lc == "high": cmap[c] = "High"
        elif lc == "low": cmap[c] = "Low"
        elif lc in ("close","adj close"): cmap[c] = "Close"
        elif lc == "volume": cmap[c] = "Volume"
    raw = raw.rename(columns=cmap)
    needed = ["Open","High","Low","Close","Volume"]
    if any(c not in raw.columns for c in needed):
        return None
    df = raw[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col].squeeze(), errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    if len(df) < 50:
        return None

    df = indicator.compute(df)

    # Build response
    bars = []
    signals = []
    for idx, row in df.iterrows():
        ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(pd.Timestamp(idx).timestamp())
        bar = {
            "time": ts,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        }
        # Add indicator data
        for col in ["NAU_Signal","NAU_Confidence","NAU_Regime","NAU_Kalman",
                     "NAU_Kalman_Score","NAU_Wavelet_Score","NAU_HMM_Score",
                     "NAU_Entropy_Score","NAU_Hurst_Score","NAU_Fractal_Score",
                     "NAU_OB_Score","NAU_FVG_Score","NAU_Structure_Score",
                     "NAU_Williams_Score","NAU_Attention_Score","NAU_RL_Score",
                     "NAU_DeepRegime_Score","NAU_OrderFlow_Score",
                     "NAU_MicroStructure_Score","NAU_MTF_Score"]:
            if col in df.columns:
                bar[col] = round(float(row[col]), 4)
        bars.append(bar)

        # Collect signal markers
        if row.get("NAU_Long", False):
            signals.append({
                "time": ts, "type": "LONG",
                "price": round(float(row["Close"]), 4),
                "confidence": round(float(row["NAU_Confidence"]) * 100, 1),
                "signal_score": round(float(row["NAU_Signal"]), 1),
            })
        if row.get("NAU_Short", False):
            signals.append({
                "time": ts, "type": "SHORT",
                "price": round(float(row["Close"]), 4),
                "confidence": round(float(row["NAU_Confidence"]) * 100, 1),
                "signal_score": round(float(row["NAU_Signal"]), 1),
            })

    # Compute SL/TP for latest signal
    last_bar = bars[-1] if bars else None
    sl_tp = None
    if last_bar and signals:
        last_sig = signals[-1]
        atr_values = []
        for i in range(max(0, len(bars)-14), len(bars)):
            atr_values.append(bars[i]["high"] - bars[i]["low"])
        atr = np.mean(atr_values) if atr_values else 0
        entry = last_sig["price"]
        if last_sig["type"] == "LONG":
            sl_tp = {
                "type": "LONG", "entry": entry,
                "sl": round(entry - 1.5 * atr, 4),
                "tp1": round(entry + 2.0 * atr, 4),
                "tp2": round(entry + 3.0 * atr, 4),
                "tp3": round(entry + 4.5 * atr, 4),
                "risk_reward": "1:2 / 1:3",
                "atr": round(atr, 4),
            }
        else:
            sl_tp = {
                "type": "SHORT", "entry": entry,
                "sl": round(entry + 1.5 * atr, 4),
                "tp1": round(entry - 2.0 * atr, 4),
                "tp2": round(entry - 3.0 * atr, 4),
                "tp3": round(entry - 4.5 * atr, 4),
                "risk_reward": "1:2 / 1:3",
                "atr": round(atr, 4),
            }

    # Summary
    last = df.iloc[-1]
    summary = {
        "symbol": sym,
        "interval": interval,
        "bars_count": len(bars),
        "signal": round(float(last["NAU_Signal"]), 1),
        "confidence": round(float(last["NAU_Confidence"]) * 100, 1),
        "regime": int(last["NAU_Regime"]),
        "regime_name": {0:"BULL",1:"BEAR",2:"RANGE"}.get(int(last["NAU_Regime"]), "RANGE"),
        "total_long": int(df["NAU_Long"].sum()),
        "total_short": int(df["NAU_Short"].sum()),
        "last_price": round(float(last["Close"]), 4),
    }

    return {"bars": bars, "signals": signals, "sl_tp": sl_tp, "summary": summary}


@app.get("/api/chart")
def get_chart(
    symbol: str = Query("PLTR"),
    interval: str = Query("1d"),
    prepost: bool = Query(False),
):
    """Get full chart data with indicators for a symbol."""
    result = download_and_compute(symbol.upper(), interval, prepost)
    if result is None:
        return {"error": f"No data for {symbol} on {interval}"}
    return result


@app.get("/api/scan")
def scan_stocks(
    interval: str = Query("1d"),
    min_confidence: float = Query(70),
):
    """
    Scan top stocks across sectors for trading signals.
    Returns top 20 sorted by signal strength and confidence.
    """
    watchlist = {
        "Technology": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","PLTR","AMD","CRM","ADBE","ORCL","NFLX","INTC","UBER"],
        "Finance": ["JPM","BAC","GS","V","MA","PYPL","COIN","BRK-B","MS"],
        "Healthcare": ["JNJ","UNH","LLY","ABBV","MRK","PFE"],
        "Energy": ["XOM","CVX","OXY"],
        "Consumer": ["WMT","COST","HD","DIS","MCD","KO","NKE","SBUX"],
        "ETFs": ["SPY","QQQ","IWM","ARKK","SOXX","XLF","XLE"],
        "Crypto": ["BTC-USD","ETH-USD","SOL-USD"],
    }

    results = []
    for sector, symbols in watchlist.items():
        for sym in symbols:
            try:
                data = download_and_compute(sym, interval, False)
                if data is None:
                    continue
                s = data["summary"]
                sig_strength = abs(s["signal"])
                conf = s["confidence"]
                if conf >= min_confidence and sig_strength > 15:
                    results.append({
                        "symbol": sym,
                        "sector": sector,
                        "signal": s["signal"],
                        "confidence": conf,
                        "regime": s["regime_name"],
                        "direction": "LONG" if s["signal"] > 0 else "SHORT",
                        "price": s["last_price"],
                        "sl_tp": data["sl_tp"],
                        "score": round(sig_strength * (conf / 100), 1),
                    })
            except Exception:
                continue

    # Sort by composite score
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"scan_results": results[:20], "total_scanned": sum(len(v) for v in watchlist.values())}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "4.0", "engine": "18-Factor AI/ML"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
