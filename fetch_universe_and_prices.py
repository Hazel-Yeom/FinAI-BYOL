# fetch_universe_and_prices.py
import os
import time
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

NASDAQ_TRADER_BASE = "https://www.nasdaqtrader.com/dynamic/SymDir"

def _read_nasdaqtrader_symbol_file(url: str) -> pd.DataFrame:
    """
    Nasdaq Trader symbol directory files are pipe-delimited with header/footer lines.
    """
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text.strip().splitlines()
    # remove footer lines like "File Creation Time: ..."
    # and last line like "Total Records|xxxx"
    rows = [line for line in text if "|" in line and not line.startswith("File Creation") and not line.startswith("Total Records")]
    # first row is header
    header = rows[0].split("|")
    data = [row.split("|") for row in rows[1:]]
    return pd.DataFrame(data, columns=header)

def get_us_listed_tickers() -> pd.DataFrame:
    """
    Returns a DataFrame with tickers from NASDAQ, NYSE, AMEX (via Nasdaq Trader files).
    """
    nasdaq = _read_nasdaqtrader_symbol_file(f"{NASDAQ_TRADER_BASE}/nasdaqlisted.txt")
    other  = _read_nasdaqtrader_symbol_file(f"{NASDAQ_TRADER_BASE}/otherlisted.txt")

    # Normalize column names
    # nasdaqlisted has "Symbol", otherlisted has "ACT Symbol"
    nasdaq = nasdaq.rename(columns={"Symbol": "ticker"})
    other  = other.rename(columns={"ACT Symbol": "ticker"})

    # Filter out test issues / ETFs if you want (optional)
    # These files include ETFs; for MVP it's fine to keep them, or filter by "ETF" column in nasdaq.
    # nasdaq columns: ticker, Security Name, Market Category, Test Issue, Financial Status, Round Lot Size, ETF, NextShares
    if "Test Issue" in nasdaq.columns:
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    if "Test Issue" in other.columns:
        other = other[other["Test Issue"] == "N"]

    # Clean tickers: Yahoo uses '-' instead of '.' for some tickers (e.g., BRK.B -> BRK-B)
    def yahooize(t: str) -> str:
        return t.replace(".", "-").strip()

    nasdaq["ticker_yahoo"] = nasdaq["ticker"].map(yahooize)
    other["ticker_yahoo"]  = other["ticker"].map(yahooize)

    # Add exchange label if available
    if "Exchange" in other.columns:
        other["exchange"] = other["Exchange"]
    else:
        other["exchange"] = "OTHER"
    nasdaq["exchange"] = "NASDAQ"

    universe = pd.concat(
        [nasdaq[["ticker", "ticker_yahoo", "exchange", "Security Name"]],
         other[["ticker", "ticker_yahoo", "exchange", "Security Name"]]],
        ignore_index=True
    ).drop_duplicates(subset=["ticker_yahoo"])

    universe = universe.rename(columns={"Security Name": "name"})
    universe = universe[universe["ticker_yahoo"].str.len() > 0]
    return universe.reset_index(drop=True)

@dataclass
class DownloadConfig:
    start: str = "2010-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    batch_size: int = 200
    sleep_sec: float = 1.0
    max_retries: int = 3

def download_prices_yahoo(
    tickers: List[str],
    out_dir: str,
    cfg: DownloadConfig
) -> None:
    """
    Downloads OHLCV for tickers in batches and stores as parquet per ticker.
    """
    os.makedirs(out_dir, exist_ok=True)

    # skip already downloaded
    tickers = [t for t in tickers if not os.path.exists(os.path.join(out_dir, f"{t}.parquet"))]
    if not tickers:
        print("All tickers already downloaded.")
        return

    total = len(tickers)
    processed = 0
    start_time = time.time()

    num_batches = math.ceil(total / cfg.batch_size)
    for b in range(num_batches):
        batch = tickers[b*cfg.batch_size:(b+1)*cfg.batch_size]
        if not batch:
            continue

        # retry logic
        for attempt in range(cfg.max_retries):
            try:
                df = yf.download(
                    tickers=batch,
                    start=cfg.start,
                    end=cfg.end,
                    interval=cfg.interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                # yf.download returns:
                # - for multiple tickers: columns as MultiIndex (Field, Ticker) OR (Ticker, Field) depending
                # handle both.
                if isinstance(df.columns, pd.MultiIndex):
                    # try to infer orientation
                    # common format: ("Adj Close", "AAPL") etc
                    level0 = df.columns.get_level_values(0)
                    level1 = df.columns.get_level_values(1)
                    # if fields appear in level0, tickers in level1
                    fields = {"Open","High","Low","Close","Adj Close","Volume"}
                    if len(set(level0) & fields) > 0:
                        # (Field, Ticker)
                        for t in batch:
                            path = os.path.join(out_dir, f"{t}.parquet")
                            if t not in set(level1):
                                processed += 1
                                if total and processed and processed <= total:
                                    elapsed = time.time() - start_time
                                    pct = processed/total if total else 1.0
                                    eta = (elapsed / processed * (total - processed)) if processed else 0
                                    print(f"[{processed}/{total}] {pct:.1%} elapsed {int(elapsed)}s ETA {int(eta)}s")
                                continue
                            sub = df.xs(t, axis=1, level=1).copy()
                            sub = sub.dropna(how="all")
                            if len(sub) < 60:
                                processed += 1
                                if total and processed and processed <= total:
                                    elapsed = time.time() - start_time
                                    pct = processed/total if total else 1.0
                                    eta = (elapsed / processed * (total - processed)) if processed else 0
                                    print(f"[{processed}/{total}] {pct:.1%} elapsed {int(elapsed)}s ETA {int(eta)}s")
                                continue
                            try:
                                os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
                                sub.to_parquet(path)
                            except KeyboardInterrupt:
                                print("Download interrupted by user.")
                                raise
                            except Exception as e:
                                print(f"Error writing {path}: {e}")
                            finally:
                                processed += 1
                                if total and processed and processed <= total:
                                    elapsed = time.time() - start_time
                                    pct = processed/total if total else 1.0
                                    eta = (elapsed / processed * (total - processed)) if processed else 0
                                    print(f"[{processed}/{total}] {pct:.1%} elapsed {int(elapsed)}s ETA {int(eta)}s")
                    else:
                        # (Ticker, Field)
                        for t in batch:
                            path = os.path.join(out_dir, f"{t}.parquet")
                            if t not in set(level0):
                                processed += 1
                                if total and processed and processed <= total:
                                    elapsed = time.time() - start_time
                                    pct = processed/total if total else 1.0
                                    eta = (elapsed / processed * (total - processed)) if processed else 0
                                    print(f"[{processed}/{total}] {pct:.1%} elapsed {int(elapsed)}s ETA {int(eta)}s")
                                continue
                            sub = df[t].copy()
                            sub = sub.dropna(how="all")
                            if len(sub) < 60:
                                processed += 1
                                if total and processed and processed <= total:
                                    elapsed = time.time() - start_time
                                    pct = processed/total if total else 1.0
                                    eta = (elapsed / processed * (total - processed)) if processed else 0
                                    print(f"[{processed}/{total}] {pct:.1%} elapsed {int(elapsed)}s ETA {int(eta)}s")
                                continue
                            try:
                                os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
                                sub.to_parquet(path)
                            except KeyboardInterrupt:
                                print("Download interrupted by user.")
                                raise
                            except Exception as e:
                                print(f"Error writing {path}: {e}")
                            finally:
                                processed += 1
                                if total and processed and processed <= total:
                                    elapsed = time.time() - start_time
                                    pct = processed/total if total else 1.0
                                    eta = (elapsed / processed * (total - processed)) if processed else 0
                                    print(f"[{processed}/{total}] {pct:.1%} elapsed {int(elapsed)}s ETA {int(eta)}s")
                else:
                    # single ticker case unlikely here
                    pass

                time.sleep(cfg.sleep_sec + random.random()*0.5)
                break
            except Exception as e:
                if attempt == cfg.max_retries - 1:
                    print(f"[batch {b+1}/{num_batches}] failed after retries: {e}")
                else:
                    time.sleep(cfg.sleep_sec * (attempt + 1) + random.random())
                    continue

if __name__ == "__main__":
    universe = get_us_listed_tickers()
    universe.to_csv("us_universe.csv", index=False)
    print(f"Universe size: {len(universe)}")

    tickers = universe["ticker_yahoo"].tolist()
    cfg = DownloadConfig(start="2023-01-01", batch_size=200, sleep_sec=1.0)
    download_prices_yahoo(tickers, out_dir="data/yahoo_parquet", cfg=cfg)
