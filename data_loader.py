"""
data_loader.py

Step 1 of the pipeline: build the eligible stock universe from the
Nifty 100 / Midcap 100 / Smallcap 100 constituent lists, then pull and
cache daily adjusted-close price history for every stock via yfinance.

Before running:
1. Download these three files into a local `data/` folder:
   https://niftyindices.com/IndexConstituent/ind_nifty100list.csv
   https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv
   https://niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv
2. pip install yfinance pandas
"""

import os
import time
import pandas as pd
import yfinance as yf

START_DATE = "2021-01-01"
END_DATE = "2026-06-30"
RAW_DATA_DIR = "data/raw"

UNIVERSE_FILES = [
    "data/ind_nifty100list.csv",
    "data/ind_niftymidcap100list.csv",
    "data/ind_niftysmallcap100list.csv",
]


def load_universe(csv_paths):
    """
    Reads the NSE index constituent CSVs and returns a sorted list of
    unique tickers with the '.NS' suffix yfinance expects.
    """
    tickers = set()
    for path in csv_paths:
        df = pd.read_csv(path)
        symbols = df["Symbol"].dropna().unique()
        tickers.update(symbols)
    return sorted(f"{sym}.NS" for sym in tickers)


def fetch_and_cache(tickers, start=START_DATE, end=END_DATE, out_dir=RAW_DATA_DIR, max_retries=3):
    """
    Downloads adjusted daily price history for each ticker and saves it
    as its own CSV under out_dir. Skips tickers already cached, so this
    is safe to re-run without re-downloading everything each time.

    Retries each ticker up to max_retries times with an increasing delay
    if the request fails or comes back empty — this handles Yahoo
    Finance's rate limiting, which is the most common reason a batch of
    ~300 rapid requests will show many "failed" tickers on the first run,
    even for large, obviously still-listed companies.

    Returns the list of tickers that still failed after all retries.
    """
    os.makedirs(out_dir, exist_ok=True)
    failed = []

    for ticker in tickers:
        out_path = os.path.join(out_dir, f"{ticker.replace('.', '_')}.csv")
        if os.path.exists(out_path):
            continue

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,  # split/dividend-adjusted prices
                    progress=False,
                )
                if not data.empty:
                    data.to_csv(out_path)
                    success = True
                    break
                print(f"{ticker}: empty response on attempt {attempt}, retrying...")
            except Exception as e:
                print(f"{ticker}: attempt {attempt} error ({e}), retrying...")
            time.sleep(2 * attempt)  # back off longer with each retry

        if not success:
            failed.append(ticker)

        time.sleep(1.5)  # slow down between different tickers to avoid rate limiting

    if failed:
        print(f"\n{len(failed)} tickers still failed after {max_retries} attempts each:")
        print(failed)

    return failed


if __name__ == "__main__":
    universe = load_universe(UNIVERSE_FILES)
    print(f"Universe size: {len(universe)} stocks")
    fetch_and_cache(universe)
    print("Done. Check data/raw/ for one CSV per stock.")
