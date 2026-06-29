import os

def init_tables(client):
    chunk_index = int(os.environ.get("CHUNK_INDEX", "0"))
    if chunk_index != 0:
        return # Hanya Robot 0 yang boleh membuat tabel agar tidak bentrok
        
    print("[DB] Menginisialisasi Skema Tabel...")
    
    queries = [
        """
        CREATE TABLE IF NOT EXISTS technical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL,
            UNIQUE(date, ticker)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fundamental_quarterly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            year INTEGER NOT NULL,
            quarter TEXT NOT NULL,
            report_date TEXT,
            revenue REAL,
            net_profit REAL,
            eps REAL,
            total_assets REAL,
            total_liabilities REAL,
            total_equity REAL,
            roe REAL,
            UNIQUE(ticker, year, quarter)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS macro_economic (
            date TEXT PRIMARY KEY,
            ihsg REAL,
            usd_idr REAL,
            gold_price REAL,
            oil_price REAL,
            macro_sentiment_score REAL DEFAULT 0.0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            sentiment_score REAL,
            news_count INTEGER,
            UNIQUE(date, ticker)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS main_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            
            -- Harga
            open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL, price REAL,
            
            -- Teknikal Asli
            sma_20 REAL, sma_50 REAL, ema_20 REAL, rsi REAL, macd REAL, macd_signal REAL,
            bb_middle REAL, bb_upper REAL, bb_lower REAL,
            daily_return REAL, volatility_20 REAL, volume_sma_20 REAL, volume_ratio REAL, atr_14 REAL, stoch_rsi REAL,
            
            -- Teknikal Lanjutan
            return_1d REAL, return_20d REAL, momentum_20 REAL,
            vol_10 REAL, vol_20 REAL, drawdown REAL, range_pct REAL,
            price_fft REAL, fft_dev REAL,
            
            -- Fundamental
            revenue REAL, net_profit REAL, total_assets REAL, total_liabilities REAL, total_equity REAL, roe REAL, eps REAL,
            net_margin REAL, der REAL,
            
            -- Makro
            ihsg REAL, gold_price REAL, usd_idr REAL, oil_price REAL,
            ihsg_momentum REAL, gold_momentum REAL, usd_momentum REAL, oil_momentum REAL,
            macro_sentiment_score REAL,
            
            -- Sentimen
            sentiment_score REAL,
            
            -- Target
            target_return_20d REAL,
            
            UNIQUE(date, ticker)
        )
        """
    ]
    
    for q in queries:
        client.execute(q)
        
    print("[DB] Skema berhasil diinisialisasi (Tabel siap digunakan).")
