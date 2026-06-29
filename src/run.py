import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load .env FIRST so HF_TOKEN and other secrets are available before any import
load_dotenv(override=True)

from src.collectors.prices import fetch_and_store
from src.collectors.macro import collect_macro
from src.collectors.sentiment import collect_sentiment
from src.modeling.model import custom_loss, UnifiedMultiTaskModel

from src.collectors.fundamental import FundamentalCollector

def get_features_for_ticker(ticker):
    path = "data/processed/features/MainDataset.csv"
    import os, pandas as pd
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Filter for this specific ticker
        base_ticker = ticker.split('.')[0]
        if "ticker" in df.columns:
            df = df[df["ticker"] == base_ticker]
            if len(df) > 0:
                return df
    return None

# ORCHESTRATOR
def run_pipeline(ticker):
    print(f"\n[ORCHESTRATOR] Starting Full Pipeline for {ticker}")
    print("=" * 60)
    
    # 1. Collect Prices (Force 10 years to ensure StockForecast strategy works)
    print("\n1. Collecting Historical Prices (10 Years)...")
    fetch_and_store(ticker, period="10y")
    
    # 2. Collect Macro (Global)
    print("\n2. Collecting Macroeconomic Data (USD/IDR, IHSG)...")
    collect_macro()
    
    # 3. Collect Sentiment (Global/Ticker)
    print(f"\n3. Collecting News Sentiment for {ticker}...")
    collect_sentiment(target_ticker=ticker)
    
    print(f"\n4. Collecting Fundamental Data for {ticker}...")
    fundamental_collector = FundamentalCollector()
    fundamental_collector.collect_quarterly(ticker)
    
    # 5. Compute Indicators and Process Features
    print("\n5. Calculating Technical Indicators and Features...")
    from src.features.process import process_features
    process_features(ticker)
    
    # 6. Build Final Features (Skip CSV generation, we do it in RAM now)
    print("\n6. Skipping CSV generation (In-Memory updates confirmed)...")
    
    print("\n[ORCHESTRATOR] Pipeline Complete. Ready for Training.")
    print("=" * 60)

SEQ_LEN = 90        #CONFIG
FEATURE_DIR = "data/processed/features"

N_RUNS = 5       
TECH_WEIGHT = 0.85    
FUND_WEIGHT = 0.15


def load_features(ticker):  #LOAD FEATURES (NOW IN-MEMORY)
    df = get_features_for_ticker(ticker)
    if df is None:
        raise FileNotFoundError(f"Could not generate features for ticker: {ticker}")
    return df


def get_column_groups(df):
    target_col = "target_return_20d"

    tech_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "rsi", "macd", "sma", "ema",
            "bb", "vol", "return", "momentum",
            "atr", "stoch", "fft"
        ])
        and c not in [target_col, "ticker"]
    ]

    fund_cols = [
        c for c in df.columns
        if c not in tech_cols
        and c not in ["raw_price", "price", target_col, "ticker"]
        and c in df.columns # Ensure existence
    ]
    
    return tech_cols, fund_cols, target_col


def make_sequences(df, tech_cols, fund_cols, target_col):     #MAKE SEQUENCES (NO SCALING HERE)
    X_seq, X_fund, y = [], [], []

    # Ensure we have enough data
    if len(df) <= SEQ_LEN:
        return np.array([]), np.array([]), np.array([])

    for i in range(len(df) - SEQ_LEN):
        # Target alignment: 
        # sequence [i : i+SEQ_LEN] -> target at [i+SEQ_LEN]
        
        target_val = df[target_col].iloc[i + SEQ_LEN]
        
        # Check if target is valid (not NaN)
        if pd.isna(target_val):
            continue

        X_seq.append(df[tech_cols].iloc[i:i + SEQ_LEN].values)
        X_fund.append(df[fund_cols].iloc[i + SEQ_LEN].values)
        y.append(target_val)

    return (
        np.array(X_seq),
        np.array(X_fund),
        np.array(y).reshape(-1, 1)
    )

def make_sequences_unified(df, tech_cols, fund_cols, target_col):
    X_seq, X_fund, y_ret, y_dir = [], [], [], []
    if len(df) <= SEQ_LEN:
        return np.array([]), np.array([]), np.array([]), np.array([])

    for i in range(len(df) - SEQ_LEN):
        target_val = df[target_col].iloc[i + SEQ_LEN]
        if pd.isna(target_val):
            continue

        X_seq.append(df[tech_cols].iloc[i:i + SEQ_LEN].values)
        X_fund.append(df[fund_cols].iloc[i + SEQ_LEN].values)
        y_ret.append(target_val)
        y_dir.append(1 if target_val > 0 else 0)

    return (
        np.array(X_seq),
        np.array(X_fund),
        np.array(y_ret).reshape(-1, 1),
        np.array(y_dir).reshape(-1, 1)
    )



def run(ticker):            #MAIN RUN
    
    # AUTO-RUN PIPELINE
    run_pipeline(ticker)

    print(f"\nMULTI-RUN ENSEMBLE TRAIN & FORECAST : {ticker}")
    print("=" * 60)

    df = load_features(ticker)
    
    # Identify Columns
    tech_cols, fund_cols, target_col = get_column_groups(df)
    
    print(f"Features: {len(tech_cols)} Technical, {len(fund_cols)} Fundamental")

    # SPLIT DATA (Time Series Split)
    # Train = First 80%
    # Test  = Last 20%
    
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    
    # For test set, we need previous SEQ_LEN rows to form the first sequence
    test_start_idx = max(0, split_idx - SEQ_LEN)
    df_test = df.iloc[test_start_idx:].copy()

    # SCALING (Using Fixed Global Statistics)
    from src.modeling.stats import SCALER_STATS
    
    # 1. Technical Scaling
    if tech_cols:
        print(f"Applying fixed scaling to {len(tech_cols)} Tech features...")
        for col in tech_cols:
            if col in SCALER_STATS["tech"]["mean"]:
                m = SCALER_STATS["tech"]["mean"][col]
                s = SCALER_STATS["tech"]["std"][col]
                s = s if s > 0 else 1.0 # Anti-NaN Fix
                df_train[col] = (df_train[col] - m) / s
                df_test[col] = (df_test[col] - m) / s
            else:
                print(f"Warning: No stats for tech col {col}, skipping scaling.")

    # 2. Fundamental Scaling
    if fund_cols:
        print(f"Applying fixed scaling to {len(fund_cols)} Fund features...")
        for col in fund_cols:
            if col in SCALER_STATS["fund"]["mean"]:
                m = SCALER_STATS["fund"]["mean"][col]
                s = SCALER_STATS["fund"]["std"][col]
                s = s if s > 0 else 1.0 # Anti-NaN Fix
                df_train[col] = (df_train[col] - m) / s
                df_test[col] = (df_test[col] - m) / s
            else:
                print(f"Warning: No stats for fund col {col}, skipping scaling.")

    # MAKE SEQUENCES
    print("Building sequences...")
    Xs_tr, Xf_tr, y_tr_ret, y_tr_dir = make_sequences_unified(df_train, tech_cols, fund_cols, target_col)
    Xs_te, Xf_te, y_te_ret, y_te_dir = make_sequences_unified(df_test, tech_cols, fund_cols, target_col)
    
    print(f"Train samples: {len(y_tr_ret)}")
    print(f"Test samples : {len(y_te_ret)}")

    # Target Scaling
    y_scaler = StandardScaler()
    print(f"y_tr stats: Mean={np.mean(y_tr_ret):.4f}, Std={np.std(y_tr_ret):.4f}, Min={np.min(y_tr_ret):.4f}, Max={np.max(y_tr_ret):.4f}")
    y_tr_scaled = y_scaler.fit_transform(y_tr_ret)
    
    import tensorflow as tf
    from src.modeling.model import custom_loss, UnifiedMultiTaskModel
    import json

    try:
        with open("config/sectors.json", "r") as f:
            sector_data = json.load(f)
        ticker_to_sector = sector_data["ticker_to_sector"]
        sector_to_id = sector_data["sector_to_id"]
        num_sectors = sector_data["num_sectors"]
    except:
        ticker_to_sector = {}
        sector_to_id = {"Unknown": 0}
        num_sectors = 1

    sec_name = ticker_to_sector.get(ticker, "Unknown")
    sec_id = sector_to_id.get(sec_name, 0)
    Xsec_te = np.full((len(Xs_te), 1), sec_id, dtype=np.int32)

    print("\n--- LOADING UNIFIED MULTI-TASK MODEL ---")
    model = UnifiedMultiTaskModel(seq_len=SEQ_LEN, num_sectors=num_sectors, embed_dim=8)
    model.model = tf.keras.models.load_model("models/UnifiedModel.h5", custom_objects={"custom_loss": custom_loss})

    print("\n--- EVALUATING ON TEST SET ---")
    if len(Xs_te) > 0:
        preds = model.model.predict({"tech_input": Xs_te, "sector_input": Xsec_te, "fund_input": Xf_te})
        pred_ret_scaled = preds["return_head"].ravel()
        pred_dir_prob = preds["direction_head"].ravel()
        
        pred_ret_inv = y_scaler.inverse_transform(pred_ret_scaled.reshape(-1, 1)).ravel()
        y_te_inv = y_scaler.inverse_transform(y_te_ret).ravel()

        mae = np.mean(np.abs(y_te_inv - pred_ret_inv))
        rmse = np.sqrt(np.mean((y_te_inv - pred_ret_inv) ** 2))
        
        # Dual direction evaluation
        dir_acc_return = np.mean(np.sign(y_te_inv) == np.sign(pred_ret_inv))
        dir_acc_prob = np.mean((y_te_dir.ravel() == 1) == (pred_dir_prob >= 0.5))

        print(f"MAE                   : {mae:.4f}")
        print(f"RMSE                  : {rmse:.4f}")
        print(f"Direction Acc (Return): {dir_acc_return * 100:.2f}%")
        print(f"Direction Acc (Prob)  : {dir_acc_prob * 100:.2f}%")

        forecast_returns = []
        fut_preds = model.model.predict({
            "tech_input": Xs_te[-1:], 
            "sector_input": Xsec_te[-1:], 
            "fund_input": Xf_te[-1:]
        })
        y_hat_final_scaled = fut_preds["return_head"][0][0]
        y_hat_final = y_scaler.inverse_transform([[y_hat_final_scaled]])[0][0]
        prob_up = fut_preds["direction_head"][0][0]

        forecast_returns.append(y_hat_final)
        print(f"\n[Multi-Task Signal] Upward Probability: {prob_up * 100:.2f}%")
    else:
        forecast_returns = [0]
        print("Not enough data to predict.")
    # PREDICT PRICE
    # We need the last RAW price (unscaled). 
    # Since df is loaded from CSV and we modified it in place, let's look at the original df loaded or just the raw_price column if we didn't scale it.
    # checking get_column_groups: tech_cols excludes "raw_price".
    # So "raw_price" in df_test is still original.
    
    last_price = df_test["raw_price"].iloc[-1]          

    mean_return_20d = np.mean(forecast_returns)
    std_return_20d = np.std(forecast_returns)

    price_pred_20d = last_price * np.exp(mean_return_20d)

    print("\n" + "=" * 60)
    print(f"LAST PRICE          : {last_price:.2f}")
    print(f"PREDICTED RETURN 20D: {mean_return_20d:.4f}")
    print(f"PRICE PREDICT IN 20D: {price_pred_20d:.2f}")
    print("=" * 60)

   
    
    # VISUALIZATION (Deferred import to avoid matplotlib dependency during training)
    from src.plot import plot_forecast
    plot_forecast(                          
        dates=df.index, # Use full index for plotting context
        prices=df["raw_price"].values, # Use full raw prices
        future_mean=mean_return_20d,
        future_std=std_return_20d,
        horizon=20
    )


if __name__ == "__main__":              # ENTRY POINT
    parser = argparse.ArgumentParser("Multi-Run Ensemble Forecast")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    run(args.ticker)
