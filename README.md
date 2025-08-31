# Fintech Risk Mini-App

This Streamlit app demonstrates a simple moving average (SMA) crossover strategy for any ticker symbol.  It fetches up to a year's worth of historical price data via [`yfinance`](https://pypi.org/project/yfinance/), calculates short and long moving averages, and produces basic performance metrics.

Every time you run the app it writes a timestamp to a local SQLite database (`metrics.db`) so you can verify usage.  Use this app as a lightweight risk analytics demo or as a starting point for your own trading research.

## Installation

1. Navigate into the `fintech_risk_app` directory.
2. Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Running the App

To start the app locally:

```bash
streamlit run app.py
```

Enter a ticker symbol (e.g. `BTC-USD`, `SPY`, `AAPL`) when prompted.  The app will fetch the latest price data, compute SMA crossover signals, and display charts along with annualized return, volatility and Sharpe ratio.

## Proving Usage

The app logs each session to a local SQLite database called `metrics.db`.  To see the usage log you can run:

```bash
python -c "import sqlite3; conn = sqlite3.connect('metrics.db'); print(conn.execute('SELECT * FROM usage_log').fetchall())"
```

Alternatively, open `metrics.db` with any SQLite viewer and inspect the `usage_log` table.  Each row contains an auto‑incremented ID and an ISO‑formatted timestamp representing when the app was launched.  Capture a screenshot of this table to demonstrate that the app has been executed.