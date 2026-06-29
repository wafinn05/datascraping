import yfinance as yf
import pandas as pd
import time
import json
import os
import sys
import libsql_client

# CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is ScrapingClone
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

sys.path.append(PROJECT_ROOT)
from src.database.connection import get_db_connection

DEFAULT_PERIOD = "1mo" # Ambil sebulan terakhir untuk scraping harian
REQUEST_DELAY = 1.0 

# UTILS
def load_tickers(region="indonesia"):
    path = os.path.join(CONFIG_DIR, "tickers.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get(region, [])

# CORE LOGIC
def fetch_and_store(ticker: str, period: str = DEFAULT_PERIOD):
    print(f"\n{ticker} | period={period}")
    time.sleep(REQUEST_DELAY)

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        print("No data returned")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
        
    df["ticker"] = ticker

    # Menyimpan ke Database Turso / SQLite (UPSERT)
    try:
        with get_db_connection() as client:
            records = df.to_dict('records')
            stmts = []
            for row in records:
                # Ensure no NaN values, convert to None for database
                for k, v in row.items():
                    if pd.isna(v): row[k] = None
                    
                query = """
                    INSERT INTO technical_prices (date, ticker, open, high, low, close, adj_close, volume)
                    VALUES (:date, :ticker, :open, :high, :low, :close, :adj_close, :volume)
                    ON CONFLICT(date, ticker) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    adj_close=excluded.adj_close,
                    volume=excluded.volume
                """
                stmts.append(libsql_client.Statement(query, row))
                
            for i in range(0, len(stmts), 500):
                client.batch(stmts[i:i+500])
            print(f"Saved {len(df)} rows to Turso DB (technical_prices)")
    except Exception as e:
        print(f"Database Save Error: {e}")

# CLI
if __name__ == "__main__":
    import argparse
    
    with get_db_connection() as client:
        from src.database.schema import init_tables
        init_tables(client) # Pastikan tabel ada

    parser = argparse.ArgumentParser("Yahoo Technical Price Collector (Turso)")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    if args.ticker:
        fetch_and_store(args.ticker, args.period)
    else:
        tickers = load_tickers("indonesia")
        if args.limit > 0:
            tickers = tickers[:args.limit]

        for i, item in enumerate(tickers, 1):
            t = item["ticker"] if isinstance(item, dict) else item
            print(f"[{i}/{len(tickers)}]")
            fetch_and_store(t, args.period)
