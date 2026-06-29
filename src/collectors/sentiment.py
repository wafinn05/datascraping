import feedparser
import pandas as pd
from datetime import datetime
import urllib.parse
import json
import os
import sys
import libsql_client

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
MAIN_REPO_ROOT = os.path.dirname(PROJECT_ROOT)

sys.path.insert(0, PROJECT_ROOT)
from src.database.connection import get_db_connection
from src.modeling.indobert import get_engine

# Simple Indonesian Sentiment Dictionary
POSITIVE_WORDS = [
    "naik", "melonjak", "tumbuh", "laba", "untung", "dividen", "bullish", 
    "menguat", "positif", "rekor", "tertinggi", "buy", "akumulasi", "kinerja bagus"
]
NEGATIVE_WORDS = [
    "turun", "anjlok", "rugi", "merugi", "bearish", "melemah", "negatif", 
    "terendah", "sell", "jual", "koreksi", "gagal", "bangkrut", "utang"
]


# Initialize AI Engine (Lazy Load)
ai_engine = None

def get_sentiment_score(text):
    global ai_engine
    try:
        if ai_engine is None:
            ai_engine = get_engine()
        return ai_engine.predict(text)
    except Exception as e:
        print(f"AI Failure: {e}, falling back to keywords")
        text = text.lower()
        score = 0
        for w in POSITIVE_WORDS:
            if w in text: score += 1
        for w in NEGATIVE_WORDS:
            if w in text: score -= 1
        return max(min(score, 1.0), -1.0)

def collect_sentiment(target_ticker=None):
    print(f"Collecting Sentiment...")
    
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "tickers.json")
    
    ticker_map = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        all_tickers = data.get("indonesia", [])
        tickers = [t["ticker"] if isinstance(t, dict) else t for t in all_tickers]
        
        # Parallel Chunking
        chunk_index = int(os.environ.get("CHUNK_INDEX", "0"))
        num_chunks = int(os.environ.get("NUM_CHUNKS", "1"))
        if num_chunks > 1:
            import numpy as np
            tickers = np.array_split(tickers, num_chunks)[chunk_index].tolist()
            print(f"Robot {chunk_index+1}/{num_chunks} mengerjakan {len(tickers)} saham.")
            
        for t in tickers:
            ticker_map[t] = t
    
    if target_ticker:
        stocks = [target_ticker]
    else:
        stocks = list(ticker_map.keys())

    today_date = str(datetime.now().date())
    print(f"Processing {len(stocks)} stocks...")
    
    global ai_engine
    
    # Save to Turso Database
    try:
        with get_db_connection() as client:
            stmts = []
            for ticker_raw in stocks:
                ticker_clean = ticker_raw.split(".")[0]
                company_name = ticker_map.get(ticker_raw, ticker_clean)
                
                query_text = f'"{company_name}" OR "{ticker_clean}"'
                encoded_query = urllib.parse.quote(query_text)
                
                print(f"  Searching: {query_text}")
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=id-ID&gl=ID&ceid=ID:id"
                
                feed = feedparser.parse(rss_url)
                titles = [entry.title for entry in feed.entries[:10]] # [TURBO] Batasi 10 berita
                count = len(titles)
                
                if count > 0:
                    if ai_engine is None:
                        ai_engine = get_engine()
                    
                    print(f"  [TURBO] Batch processing {count} news for {ticker_clean}...")
                    scores = ai_engine.predict_batch(titles)
                    avg = sum(scores) / count
                    final_score = max(min(avg, 1.0), -1.0)
                else:
                    final_score = 0
                    
                print(f"  {ticker_clean}: {count} news, Score: {final_score:.2f}")
                
                # Append to stmts
                query = """
                    INSERT INTO news_sentiment (date, ticker, sentiment_score, news_count)
                    VALUES (:date, :ticker, :sentiment_score, :news_count)
                    ON CONFLICT(date, ticker) DO UPDATE SET
                    sentiment_score=excluded.sentiment_score,
                    news_count=excluded.news_count
                """
                stmts.append(libsql_client.Statement(query, {
                    "date": today_date,
                    "ticker": ticker_raw,
                    "sentiment_score": final_score,
                    "news_count": count
                }))
                
            for i in range(0, len(stmts), 500):
                client.batch(stmts[i:i+500])
                
    except Exception as e:
        print(f"Database Save Error: {e}")

    print("Sentiment Collection Complete!")

if __name__ == "__main__":
    import argparse
    
    with get_db_connection() as client:
        from src.database.schema import init_tables
        init_tables(client) # Pastikan tabel ada

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Specific ticker (e.g. ASII.JK)")
    args = parser.parse_args()
    collect_sentiment(target_ticker=args.ticker)
