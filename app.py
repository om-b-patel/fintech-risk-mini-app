import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# Helper to pick a 1D Close/Adj Close series from yfinance DataFrame (MultiIndex-safe)
import pandas as pd

def select_close_series(df: pd.DataFrame, symbol: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    # MultiIndex: ('Close','BTC-USD') or ('Adj Close','BTC-USD')
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        if "Adj Close" in lv0:
            sub = df["Adj Close"]
            s = sub if isinstance(sub, pd.Series) else (sub[symbol] if symbol in sub.columns else sub.iloc[:, 0])
        elif "Close" in lv0:
            sub = df["Close"]
            s = sub if isinstance(sub, pd.Series) else (sub[symbol] if symbol in sub.columns else sub.iloc[:, 0])
        else:
            return None
    else:
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            return None
        s = df[col]
    return pd.Series(pd.to_numeric(s, errors="coerce").dropna(), name=symbol)




def log_usage() -> None:
    conn = sqlite3.connect("metrics.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS usage_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT)")
    cur.execute("INSERT INTO usage_log (timestamp) VALUES (?)", (datetime.utcnow().isoformat(),))
    conn.commit()
    conn.close()


def download_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _to_1d_series(x) -> pd.Series:
    """Coerce Series/DataFrame/ndarray/list into a clean 1D float Series with index preserved if possible."""
    if isinstance(x, pd.DataFrame):
        # if single column, take it; else squeeze columns
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            s = x.squeeze("columns")
    elif isinstance(x, pd.Series):
        s = x
    else:
        # numpy array or list
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        s = pd.Series(arr)
    # ensure numeric
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna()


def compute_sma_signals(prices, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """Compute SMA crossover signals (robust to 2D inputs)."""
    prices = _to_1d_series(prices).astype(float)
    signals = pd.DataFrame(index=prices.index)
    signals["Short"] = prices.rolling(window=short_window, min_periods=short_window).mean()
    signals["Long"]  = prices.rolling(window=long_window,  min_periods=long_window).mean()
    signals["Signal"] = (signals["Short"] > signals["Long"]).astype(float).fillna(0.0)
    signals["Positions"] = signals["Signal"].diff().fillna(0.0)
    return signals


def compute_performance(returns: pd.Series, signals: pd.Series) -> dict:
    signals = signals.reindex(returns.index).fillna(0).astype(float)
    strat_returns = (returns.astype(float) * signals.shift(1).fillna(0)).dropna()
    if strat_returns.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0}
    daily_mean = float(strat_returns.mean())
    daily_std  = float(strat_returns.std())
    ann_return = float((1.0 + daily_mean) ** 252 - 1.0)
    ann_vol    = float(daily_std * np.sqrt(252))
    sharpe     = float(ann_return / ann_vol) if ann_vol != 0.0 else 0.0
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe}


def main() -> None:
    log_usage()
    st.title("Fintech Risk Mini-App")
    st.write("Analyze a simple moving average (SMA) crossover strategy.")

    ticker = st.text_input("Ticker symbol (e.g. BTC-USD, SPY, AAPL)", "BTC-USD")

    data = download_data(ticker)
    if data.empty:
        st.error("No price data returned. Try SPY, AAPL, BTC-USD, etc.")
        return

        prices = select_close_series(data, ticker)
    if prices is None or prices.empty:
        st.error("Price column not found or empty. Try another symbol.")
        return


    prices = data[price_col].dropna()

    colA, colB = st.columns(2)
    with colA:
        short_window = st.number_input("Short SMA window", min_value=3, max_value=200, value=20, step=1)
    with colB:
        long_window  = st.number_input("Long SMA window",  min_value=5, max_value=400, value=50, step=1)

    signals = compute_sma_signals(prices, short_window, long_window).reindex(prices.index)
    returns = _to_1d_series(prices).pct_change().fillna(0)

    perf = compute_performance(returns, signals["Signal"])

    st.subheader("Price & Moving Averages")
    st.line_chart(pd.DataFrame({"Price": _to_1d_series(prices), "Short MA": signals["Short"], "Long MA": signals["Long"]}))

    st.subheader("Strategy Signal (1=long, 0=flat)")
    st.line_chart(signals["Signal"])

    st.subheader("Backtest Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized Return", f"{perf['ann_return']*100:.2f}%")
    c2.metric("Annualized Volatility", f"{perf['ann_vol']*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{perf['sharpe']:.2f}")


if __name__ == "__main__":
    main()
