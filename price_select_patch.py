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
