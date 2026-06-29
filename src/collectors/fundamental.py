import yfinance as yf
import pandas as pd
import time
import os
import sys
import json
import libsql_client
from typing import Dict, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

sys.path.append(PROJECT_ROOT)
from src.database.connection import get_db_connection

class FundamentalCollector:
    def __init__(self):
        self.config = self._load_config()
        self.years_back = self.config["data_collection"]["years_back"]
        self.delay = self.config["data_collection"]["request_delay"]
        self.q_cfg = self.config["fundamental"]["quarterly"]
        print("[OK] FundamentalCollector (Turso DB Mode) initialized")

    def _load_config(self) -> Dict[str, Any]:
        import yaml
        # Config is in main project
        path = os.path.join(os.path.dirname(PROJECT_ROOT), "config", "settings.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError("settings.yaml not found")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_tickers(self, market: str) -> list[str]:
        path = os.path.join(os.path.dirname(PROJECT_ROOT), "config", "tickers.json")
        if not os.path.exists(path):
            raise FileNotFoundError("tickers.json not found")
        with open(path, "r") as f:
            data = json.load(f)
        if market not in data:
            raise ValueError(f"Market '{market}' not found in tickers.json")
        return data[market]

    @staticmethod
    def _extract(df: pd.DataFrame, key: str, col) -> Optional[float]:
        try:
            if df is None or df.empty or key not in df.index or col not in df.columns:
                return None
            val = df.loc[key, col]
            if pd.isna(val): return None
            return float(val)
        except Exception:
            return None

    def collect_quarterly(self, ticker: str):
        time.sleep(self.delay)
        yf_stock = yf.Ticker(ticker)

        income = yf_stock.quarterly_financials
        balance = yf_stock.quarterly_balance_sheet

        if income is None or income.empty:
            print(f"[WARN] No quarterly income data for {ticker}")
            return 0

        max_quarters = min(self.years_back * 4, len(income.columns))
        quarters = list(income.columns)[:max_quarters]

        data = []
        for q in quarters:
            q_date = pd.Timestamp(q)
            year = q_date.year
            quarter = f"Q{(q_date.month - 1) // 3 + 1}"

            revenue = self._extract(income, self.q_cfg["revenue_key"], q)
            net_profit = self._extract(income, self.q_cfg["net_profit_key"], q)
            eps = self._extract(income, self.q_cfg["eps_key"], q)
            assets = self._extract(balance, self.q_cfg["assets_key"], q)
            liabilities = self._extract(balance, self.q_cfg["liabilities_key"], q)

            if all(v is None for v in [revenue, net_profit, assets, liabilities]):
                continue

            equity = None
            roe = None
            if assets is not None and liabilities is not None:
                equity = assets - liabilities
                roe = net_profit / equity if equity and net_profit is not None and equity != 0 else None

            data.append({
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "report_date": str(q_date.date()),
                "revenue": revenue,
                "net_profit": net_profit,
                "eps": eps,
                "total_assets": assets,
                "total_liabilities": liabilities,
                "total_equity": equity,
                "roe": roe
            })

        if not data:
            print(f"[DONE] {ticker}: saved=0")
            return 0
            
        try:
            with get_db_connection() as client:
                stmts = []
                for row in data:
                    query = """
                        INSERT INTO fundamental_quarterly (
                            ticker, year, quarter, report_date, revenue, net_profit, 
                            eps, total_assets, total_liabilities, total_equity, roe
                        ) VALUES (
                            :ticker, :year, :quarter, :report_date, :revenue, :net_profit,
                            :eps, :total_assets, :total_liabilities, :total_equity, :roe
                        )
                        ON CONFLICT(ticker, year, quarter) DO UPDATE SET
                            report_date=excluded.report_date,
                            revenue=excluded.revenue,
                            net_profit=excluded.net_profit,
                            eps=excluded.eps,
                            total_assets=excluded.total_assets,
                            total_liabilities=excluded.total_liabilities,
                            total_equity=excluded.total_equity,
                            roe=excluded.roe
                    """
                    stmts.append(libsql_client.Statement(query, row))
                
                for i in range(0, len(stmts), 500):
                    client.batch(stmts[i:i+500])
            print(f"[DONE] {ticker}: saved {len(data)} rows to Turso DB (fundamental_quarterly)")
            return len(data)
        except Exception as e:
            print(f"[ERROR] Database Save Error for {ticker}: {e}")
            return 0

if __name__ == "__main__":
    import argparse
    
    with get_db_connection() as client:
        from src.database.schema import init_tables
        init_tables(client) # Pastikan tabel ada

    parser = argparse.ArgumentParser("Quarterly Fundamental Collector (Turso)")
    parser.add_argument("--ticker", help="Single ticker (e.g. BBCA.JK)")
    parser.add_argument("--market", help="Market key from tickers.json (e.g. indonesia)")
    args = parser.parse_args()

    collector = FundamentalCollector()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [t["ticker"] if isinstance(t, dict) else t for t in collector.load_tickers("indonesia")]
        
        # Parallel Chunking
        chunk_index = int(os.environ.get("CHUNK_INDEX", "0"))
        num_chunks = int(os.environ.get("NUM_CHUNKS", "1"))
        if num_chunks > 1:
            import numpy as np
            tickers = np.array_split(tickers, num_chunks)[chunk_index].tolist()
            print(f"Robot {chunk_index+1}/{num_chunks} mengerjakan {len(tickers)} saham.")

    for t in tickers:
        collector.collect_quarterly(t)
