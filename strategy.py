"""
strategy.py

Step 2 of the pipeline. Reads the cached price CSVs from data/raw
(created by data_loader.py), computes the momentum + volatility score
for every stock as of a given date, and builds a top-10 portfolio
weighted inversely to volatility.

This file only looks at ONE date at a time for now, so you can check
the logic makes sense before wiring it into a full 2021-2025 loop
(that's the next script, after this one is confirmed working).
"""

import os
import pandas as pd

RAW_DATA_DIR = "data/raw"


def load_price_series(raw_dir=RAW_DATA_DIR):
    """
    Loads every cached CSV in raw_dir and returns a single DataFrame of
    adjusted close prices: one column per stock, one row per date.
    """
    frames = {}
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace("_NS.csv", ".NS")
        path = os.path.join(raw_dir, fname)

        # yfinance saves a 2-row column header (Price type / Ticker),
        # so we tell pandas both rows are headers, and the first column
        # (Date) is the index.
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        df.columns = df.columns.droplevel(1)  # drop the repeated ticker level
        df.index = pd.to_datetime(df.index)

        if "Close" in df.columns:
            frames[ticker] = df["Close"]

    prices = pd.DataFrame(frames)
    prices = prices.sort_index()
    return prices


def compute_scores(prices, as_of_date, lookback_months=6, skip_months=1):
    """
    For every stock, as of as_of_date:
      - momentum = price return from (as_of_date - 6 months) to
        (as_of_date - 1 month), i.e. 6-month momentum skipping the
        most recent month
      - volatility = std dev of daily returns over the full 6-month
        window (used both to penalise the score and to weight later)

    Both are converted to z-scores across the universe so they're on
    a comparable scale, then combined into one score:
      score = momentum_z - volatility_z
    Higher score = stronger recent trend with calmer price behaviour.
    """
    as_of_date = pd.Timestamp(as_of_date)
    lookback_start = as_of_date - pd.DateOffset(months=lookback_months)
    skip_end = as_of_date - pd.DateOffset(months=skip_months)

    momentum_window = prices.loc[lookback_start:skip_end]
    daily_returns = prices.pct_change()
    vol_window = daily_returns.loc[lookback_start:as_of_date]

    momentum = momentum_window.iloc[-1] / momentum_window.iloc[0] - 1
    volatility = vol_window.std()

    momentum = momentum.dropna()
    volatility = volatility.dropna()
    common = momentum.index.intersection(volatility.index)
    momentum, volatility = momentum[common], volatility[common]

    momentum_z = (momentum - momentum.mean()) / momentum.std()
    vol_z = (volatility - volatility.mean()) / volatility.std()

    score = (momentum_z - vol_z).sort_values(ascending=False)
    return score, volatility


def build_portfolio(score, volatility, n=10):
    """
    Takes the top n stocks by score and weights them inversely to
    their volatility, so shakier stocks in the top 10 get a smaller
    slice of the capital.
    """
    top = score.head(n)
    top_vol = volatility.loc[top.index]
    inv_vol = 1 / top_vol
    weights = inv_vol / inv_vol.sum()

    portfolio = pd.DataFrame({
        "score": top,
        "volatility": top_vol,
        "weight": weights,
    })
    return portfolio


if __name__ == "__main__":
    prices = load_price_series()
    print(f"Loaded price data for {prices.shape[1]} stocks, "
          f"{prices.shape[0]} trading days "
          f"({prices.index.min().date()} to {prices.index.max().date()})")

    test_date = "2021-07-01"
    score, volatility = compute_scores(prices, test_date)
    portfolio = build_portfolio(score, volatility)

    print(f"\nTop 10 portfolio as of {test_date}:\n")
    print(portfolio.round(4))
    print(f"\nWeights sum to: {portfolio['weight'].sum():.4f}")
