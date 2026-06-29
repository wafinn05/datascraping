import pandas as pd
import numpy as np
import os
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
import sys
sys.path.insert(0, PROJECT_ROOT)
from src.database.connection import get_db_connection

# TECHNICAL
SMA_20 = 20
SMA_50 = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
VOLATILITY_PERIOD = 20
VOLUME_SMA_PERIOD = 20

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure column names match expected formats
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)

    if "close" not in df.columns:
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        else:
            return df # Cannot calculate without close

    close = df["close"]
    volume = df.get("volume", pd.Series(0, index=df.index))

    # SMA
    df["sma_20"] = close.rolling(SMA_20, min_periods=SMA_20).mean()
    df["sma_50"] = close.rolling(SMA_50, min_periods=SMA_50).mean()

    # EMA
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    #Bollinger Bands
    bb_mid = close.rolling(BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).mean()
    bb_std = close.rolling(BOLLINGER_PERIOD, min_periods=BOLLINGER_PERIOD).std()

    df["bb_middle"] = bb_mid
    df["bb_upper"] = bb_mid + (bb_std * BOLLINGER_STD)
    df["bb_lower"] = bb_mid - (bb_std * BOLLINGER_STD)

    # Returns & Volatility
    df["daily_return"] = close.pct_change()
    df["volatility_20"] = df["daily_return"].rolling(
        VOLATILITY_PERIOD,
        min_periods=VOLATILITY_PERIOD
    ).std()

    # Volume
    df["volume_sma_20"] = volume.rolling(
        VOLUME_SMA_PERIOD,
        min_periods=VOLUME_SMA_PERIOD
    ).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

    # ATR & Stoch RSI
    if all(c in df.columns for c in ["high", "low"]):
        prev_close = close.shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
    else:
        df["atr_14"] = np.nan

    min_rsi = df["rsi"].rolling(14).min()
    max_rsi = df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)

    return df

def process_technical(ticker: str) -> pd.DataFrame:
    print(f"\n[TECH] Processing {ticker}")
    
    with get_db_connection() as client:
        result = client.execute("SELECT * FROM technical_prices WHERE ticker = ? ORDER BY date ASC", [ticker])
        
    if not result.rows:
        print(f"[WARN] No raw price data found for {ticker} in Turso DB")
        return None
        
    cols = result.columns
    data = [dict(zip(cols, row)) for row in result.rows]
    df = pd.DataFrame(data)
    
    df["Date"] = pd.to_datetime(df["date"])
    df.set_index("Date", inplace=True)
    
    df = calculate_indicators(df)
    return df
