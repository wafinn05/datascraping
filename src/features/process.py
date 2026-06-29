import pandas as pd
import numpy as np
import os
import argparse
import sys
import libsql_client
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.features.technical import process_technical
from src.database.connection import get_db_connection

SEQ_LEN = 90
TARGET_HORIZONS = [20]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  
    
    if "price" not in df.columns:
        df["price"] = df.get("close", df.get("adj_close"))

    # RETURNS
    df["return_1d"] = df["price"].pct_change().shift(1)
    df["return_20d"] = df["price"].pct_change(20).shift(1)

    # MOMENTUM
    df["momentum_20"] = (df["price"] / df["price"].shift(20) - 1).shift(1)

    # FUNDAMENTAL RATIOS
    if "net_profit" in df.columns and "revenue" in df.columns:
        df["net_margin"] = df["net_profit"] / df["revenue"].replace(0, np.nan)
    if "total_liabilities" in df.columns and "total_equity" in df.columns:
        df["der"] = df["total_liabilities"] / df["total_equity"].replace(0, np.nan)

    returns = df["price"].pct_change()
    df["vol_10"] = returns.rolling(10).std()
    df["vol_20"] = returns.rolling(20).std()
    df["drawdown"] = df["price"] / df["price"].cummax() - 1
    
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        df["range_pct"] = (df["bb_upper"] - df["bb_lower"]) / df["price"]

    # Macro Momentum
    for col in ["ihsg", "gold_price", "usd_idr", "oil_price"]:
        if col in df.columns:
            prefix = col.split("_")[0]
            df[f"{prefix}_momentum"] = df[col].pct_change(20).ffill().fillna(0)

    # FFT Smoothing
    if "price" in df.columns and len(df) > 30:
        window = 365
        prices_full = df["price"].values
        padded = np.pad(prices_full, (window - 1, 0), mode='edge')
        from numpy.lib.stride_tricks import sliding_window_view
        chunks = sliding_window_view(padded, window_shape=window)
        
        chunks_fft = np.fft.fft(chunks, axis=1)
        freqs = np.fft.fftfreq(window)
        chunks_fft[:, np.abs(freqs) > 0.1] = 0
        smoothed = np.real(np.fft.ifft(chunks_fft, axis=1))[:, -1]
        
        df["price_fft"] = smoothed
        dev = (df["price"] - df["price_fft"]) / df["price"].replace(0, 1.0)
        df["fft_dev"] = np.clip(dev, -0.2, 0.2)

    # SAFETY CLIPS
    for col in df.columns:
        if col in ["usd_idr", "ihsg", "gold_price", "oil_price"]:
            col_med = df[col].median()
            if col_med > 0:
                df[col] = np.clip(df[col], col_med * 0.1, col_med * 10.0)
        if col == "drawdown":
            df[col] = np.clip(df[col], -0.8, 0.0)

    return df

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    for h in TARGET_HORIZONS:
        df[f"target_return_{h}d"] = np.log(df["price"].shift(-h) / df["price"])
    return df

def build_features_for_stock(ticker: str, client) -> Optional[pd.DataFrame]:
    df_tech = process_technical(ticker)
    if df_tech is None or df_tech.empty:
        return None
        
    df_tech.index = pd.to_datetime(df_tech.index)
    
    if "raw_price" not in df_tech.columns:
        df_tech["raw_price"] = df_tech["close"]
    if "price" not in df_tech.columns:
        df_tech["price"] = df_tech["close"]

    # Load Macro
    result_macro = client.execute("SELECT * FROM macro_economic ORDER BY date ASC")
    if result_macro.rows:
        cols = result_macro.columns
        data = [dict(zip(cols, row)) for row in result_macro.rows]
        df_macro = pd.DataFrame(data)
        df_macro["Date"] = pd.to_datetime(df_macro["date"])
        df_macro.set_index("Date", inplace=True)
        if "date" in df_macro.columns: df_macro.drop(columns=["date"], inplace=True)
        if "ticker" in df_macro.columns: df_macro.drop(columns=["ticker"], inplace=True)
        df_tech = df_tech.join(df_macro, how="left")
    
    # Load Sentiment
    base_ticker = ticker.split('.')[0]
    result_sent = client.execute("SELECT * FROM news_sentiment WHERE ticker LIKE ? ORDER BY date ASC", [f"{base_ticker}%"])
    if result_sent.rows:
        cols = result_sent.columns
        data = [dict(zip(cols, row)) for row in result_sent.rows]
        df_sent = pd.DataFrame(data)
        df_sent["date_index"] = pd.to_datetime(df_sent["date"])
        df_sent.set_index("date_index", inplace=True)
        df_sent = df_sent[~df_sent.index.duplicated(keep='last')]
        df_tech = df_tech.join(df_sent[["sentiment_score"]], how="left")
    else:
        df_tech["sentiment_score"] = 0

    # Load Fundamental
    result_fund = client.execute("SELECT * FROM fundamental_quarterly WHERE ticker LIKE ? ORDER BY report_date ASC", [f"{base_ticker}%"])
    if result_fund.rows:
        cols = result_fund.columns
        data = [dict(zip(cols, row)) for row in result_fund.rows]
        df_fund = pd.DataFrame(data)
        df_fund = df_fund.dropna(subset=['report_date'])
        if not df_fund.empty:
            df_fund["report_date"] = pd.to_datetime(df_fund["report_date"])
            df_fund["effective_date"] = df_fund["report_date"] + pd.DateOffset(months=3)
            df_fund = df_fund.sort_values("effective_date").drop_duplicates(subset=["effective_date"], keep="last")
            df_fund.set_index("effective_date", inplace=True)
            
            df_fund_reindexed = df_fund.reindex(df_tech.index, method="ffill", tolerance=pd.Timedelta("120d"))
            df_tech = df_tech.join(df_fund_reindexed[["revenue", "net_profit", "total_assets", "total_liabilities", "total_equity", "roe", "eps"]], how="left")

    fund_cols = ["revenue", "net_profit", "total_assets", "total_liabilities", "total_equity", "roe", "eps"]
    fund_cols = [c for c in fund_cols if c in df_tech.columns]
    if fund_cols:
        df_tech[fund_cols] = df_tech[fund_cols].ffill()

    df_tech = add_features(df_tech)
    df_tech = add_targets(df_tech)
    
    df_tech = df_tech.replace([np.inf, -np.inf], np.nan)
    df_tech = df_tech.ffill().fillna(0)
    
    df_tech = df_tech.dropna(subset=[f"target_return_{h}d" for h in TARGET_HORIZONS])

    min_required = SEQ_LEN + max(TARGET_HORIZONS) + 10
    if len(df_tech) < min_required:
        print(f"Insufficient data: {len(df_tech)} rows, need {min_required}")
        return None
    
    return df_tech

def process_features(ticker: Optional[str] = None):
    if ticker:
        tickers = [ticker]
    else:
        import json
        config_path = os.path.join(PROJECT_ROOT, "config", "tickers.json")
        with open(config_path, "r") as f:
            data = json.load(f)
        tickers = [t["ticker"] if isinstance(t, dict) else t for t in data.get("indonesia", [])]
        
        # Parallel Chunking
        chunk_index = int(os.environ.get("CHUNK_INDEX", "0"))
        num_chunks = int(os.environ.get("NUM_CHUNKS", "1"))
        if num_chunks > 1:
            import numpy as np
            tickers = np.array_split(tickers, num_chunks)[chunk_index].tolist()
            print(f"Robot {chunk_index+1}/{num_chunks} mengerjakan {len(tickers)} saham.")

    all_dfs = []
    
    with get_db_connection() as client:
        for t in tickers:
            df = build_features_for_stock(t, client)
            if df is not None:
                base_ticker = t.split('.')[0]
                df["ticker"] = base_ticker
                
                # Make date a column again
                df = df.reset_index()
                df = df.rename(columns={"Date": "date", "index": "date"})
                df["date"] = df["date"].astype(str) # Convert to string for DB
                
                cols = ["date", "ticker"] + [c for c in df.columns if c not in ["date", "ticker"]]
                df = df[cols]
                all_dfs.append(df)
                print(f"Processed features for {t}")
                
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Total compiled data: {len(final_df)} rows. Saving to Turso main_dataset...")
            
            # Identify columns that exist in the dataframe to form the INSERT query dynamically
            # Our schema explicitly lists ~40 columns. We will construct a dynamic insert 
            # based on what columns actually exist in final_df to avoid missing column errors.
            db_cols = [c for c in final_df.columns if c != "id" and c != "raw_price"]
            
            col_names = ", ".join(db_cols)
            val_names = ", ".join([f":{c}" for c in db_cols])
            
            # Buat update set string "col1=excluded.col1, col2=excluded.col2..."
            update_set = ", ".join([f"{c}=excluded.{c}" for c in db_cols if c not in ["date", "ticker"]])
            
            query = f"""
                INSERT INTO main_dataset ({col_names})
                VALUES ({val_names})
                ON CONFLICT(date, ticker) DO UPDATE SET
                {update_set}
            """
            
            records = final_df.to_dict('records')
            stmts = []
            for row in records:
                for k, v in row.items():
                    if pd.isna(v): row[k] = None
                stmts.append(libsql_client.Statement(query, {k: row[k] for k in db_cols}))
            
            # Batch execute
            for i in range(0, len(stmts), 500):
                client.batch(stmts[i:i+500])
                
            print(f"Successfully saved {len(final_df)} rows to Turso (main_dataset).")
        else:
            print("No data processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str)
    args = parser.parse_args()
    
    # Inisialisasi tabel terlebih dahulu
    from src.database.schema import init_tables
    with get_db_connection() as client:
        init_tables(client)
        
    process_features(args.ticker)