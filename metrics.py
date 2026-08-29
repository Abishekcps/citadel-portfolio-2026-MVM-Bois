"""
metrics.py

Step 4 of the pipeline. Reads the portfolio value history saved by
backtest.py and computes the metrics table your report needs, plus a
comparison against a Nifty 100 buy-and-hold benchmark over the same
period.
"""

import numpy as np
import pandas as pd
import yfinance as yf

HISTORY_PATH = "data/backtest_history.csv"
INITIAL_CAPITAL = 1_00_00_000
BENCHMARK_TICKER = "^CNX100"  # Nifty 100 on Yahoo Finance


def cagr(values, trading_days_per_year=252):
    years = len(values) / trading_days_per_year
    return (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1


def sharpe_ratio(values, trading_days_per_year=252):
    daily_returns = values.pct_change().dropna()
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)


def max_drawdown(values):
    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    return drawdown.min()


def summarize(values, label):
    print(f"\n--- {label} ---")
    print(f"Start value:  Rs {values.iloc[0]:,.0f}")
    print(f"Final value:  Rs {values.iloc[-1]:,.0f}")
    print(f"Total return: {(values.iloc[-1] / values.iloc[0] - 1) * 100:.2f}%")
    print(f"CAGR:         {cagr(values) * 100:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio(values):.2f}")
    print(f"Max drawdown: {max_drawdown(values) * 100:.2f}%")


def build_benchmark(start_date, end_date, initial_capital=INITIAL_CAPITAL):
    """
    Downloads the Nifty 100 index and simulates a simple buy-and-hold
    portfolio of initial_capital, starting on start_date, for comparison.
    """
    index_data = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)
    index_close = index_data["Close"].iloc[:, 0] if index_data["Close"].ndim > 1 else index_data["Close"]
    units = initial_capital / index_close.iloc[0]
    benchmark_value = index_close * units
    return benchmark_value


if __name__ == "__main__":
    history = pd.read_csv(HISTORY_PATH, index_col=0, parse_dates=True)
    portfolio_values = history["portfolio_value"]

    summarize(portfolio_values, "Your strategy")

    benchmark_values = build_benchmark(portfolio_values.index[0], portfolio_values.index[-1])
    summarize(benchmark_values, "Nifty 100 benchmark (buy & hold)")
