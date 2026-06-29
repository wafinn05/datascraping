import yfinance as yf
import pandas as pd
import os
import sys
import libsql_client

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_PERIOD = "1mo" # Ambil sebulan terakhir untuk pembaruan harian

sys.path.append(PROJECT_ROOT)
from src.database.connection import get_db_connection

MACRO_TICKERS = {
    "USDIDR=X": "usd_idr",
    "^JKSE": "ihsg",
    "GC=F": "gold_price",
    "CL=F": "oil_price"
}

def collect_macro(period=DEFAULT_PERIOD):
    print(f"Fetching Macro Data (period={period})...")
    
    dfs = []
    for ticker, col_name in MACRO_TICKERS.items():
        print(f"  - {ticker} -> {col_name}")
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty: continue
            
            temp_df = hist[['Close']].reset_index()
            # Normalize Date and stringify
            temp_df['Date'] = pd.to_datetime(temp_df['Date']).dt.date.astype(str)
            temp_df = temp_df.rename(columns={'Close': col_name})
            dfs.append(temp_df)
        except Exception as e:
            print(f"    Error fetching {ticker}: {e}")

    from functools import reduce
    if not dfs:
        print("No macro data fetched.")
        return

    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), dfs)
    df_merged = df_merged.sort_values('Date').ffill().bfill().dropna()
    
    # Simpan ke Turso
    try:
        with get_db_connection() as client:
            records = df_merged.to_dict('records')
            stmts = []
            for row in records:
                for k, v in row.items():
                    if pd.isna(v): row[k] = None
                    
                query = """
                    INSERT INTO macro_economic (date, ihsg, usd_idr, gold_price, oil_price)
                    VALUES (:Date, :ihsg, :usd_idr, :gold_price, :oil_price)
                    ON CONFLICT(date) DO UPDATE SET
                    ihsg=excluded.ihsg,
                    usd_idr=excluded.usd_idr,
                    gold_price=excluded.gold_price,
                    oil_price=excluded.oil_price
                """
                stmts.append(libsql_client.Statement(query, row))
                
            # Execute in batches of 500
            for i in range(0, len(stmts), 500):
                client.batch(stmts[i:i+500])
            print(f"Saved {len(df_merged)} macro records to Turso DB (macro_economic) via Batch")
    except Exception as e:
        print(f"Database Save Error: {e}")

if __name__ == "__main__":
    with get_db_connection() as client:
        from src.database.schema import init_tables
        init_tables(client)
    collect_macro()
