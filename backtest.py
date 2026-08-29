"""
backtest.py

Step 3 of the pipeline: runs the strategy from strategy.py across every
quarter from 2021 to 2025 (the main backtest), tracks portfolio value
day by day, and applies the 0.1%-per-trade cost on every rebalance.

Run this only after strategy.py has printed a sensible single-date
portfolio for you (it already has).
"""

import pandas as pd
from strategy import load_price_series, compute_scores, build_portfolio

INITIAL_CAPITAL = 1_00_00_000
COST_RATE = 0.001  # 0.1% per trade, applied on both buys and sells
TOP_N = 10

BACKTEST_START = "2021-01-01"
BACKTEST_END = "2025-12-31"


def get_rebalance_dates(prices, start=BACKTEST_START, end=BACKTEST_END, freq="QS",
                         lookback_months=6):
    """
    Returns the actual trading dates closest to (on or after) each
    quarter start between start and end, based on real dates present
    in the price data, so we never try to rebalance on a holiday.

    Skips any quarter start that falls before a full lookback_months of
    price history exists yet (e.g. the very first quarter of the data),
    since the momentum/volatility signal can't be computed without that
    much history behind it. This means the portfolio effectively starts
    a bit later than the raw data does -- disclose this as a burn-in
    period in your report.
    """
    earliest_usable = prices.index.min() + pd.DateOffset(months=lookback_months)
    quarter_starts = pd.date_range(start=start, end=end, freq=freq)
    trading_days = prices.index
    rebalance_dates = []
    for qs in quarter_starts:
        if qs < earliest_usable:
            continue
        future_days = trading_days[trading_days >= qs]
        if len(future_days) > 0:
            rebalance_dates.append(future_days[0])
    return rebalance_dates


def run_backtest(prices, rebalance_dates, end=BACKTEST_END, initial_capital=INITIAL_CAPITAL,
                  cost_rate=COST_RATE, top_n=TOP_N):
    # Forward-fill small gaps (e.g. a stock not trading for a day or two)
    # so a temporary missing price doesn't wrongly zero out its value.
    prices = prices.ffill()

    holdings = {}   # ticker -> number of shares held
    entry_info = {}  # ticker -> (entry_date, entry_price), for open positions
    trades = []       # completed round-trips: entry -> exit, for accuracy/gain-loss stats
    cash = initial_capital
    history = []
    total_cost_paid = 0.0
    rebalance_set = set(rebalance_dates)

    active_dates = prices.index[
        (prices.index >= rebalance_dates[0]) & (prices.index <= pd.Timestamp(end))
    ]

    for date in active_dates:
        day_prices = prices.loc[date]

        holdings_value = sum(
            shares * day_prices.get(ticker, 0)
            for ticker, shares in holdings.items()
            if pd.notna(day_prices.get(ticker, 0))
        )
        portfolio_value = cash + holdings_value

        if date in rebalance_set:
            score, volatility = compute_scores(prices, date)
            portfolio = build_portfolio(score, volatility, n=top_n)
            target_weights = portfolio["weight"].to_dict()

            all_tickers = set(holdings.keys()) | set(target_weights.keys())
            traded_value = 0.0
            new_holdings = {}

            for ticker in all_tickers:
                price = day_prices.get(ticker, None)
                if price is None or pd.isna(price) or price <= 0:
                    continue  # can't trade a name with no valid price today

                current_shares = holdings.get(ticker, 0)
                current_value = current_shares * price
                target_value = target_weights.get(ticker, 0.0) * portfolio_value

                traded_value += abs(target_value - current_value)

                # A brand-new position is opening today
                if current_shares == 0 and target_value > 0:
                    entry_info[ticker] = (date, price)

                # An existing position is being fully closed today
                if current_shares > 0 and target_value == 0:
                    entry_date, entry_price = entry_info.pop(ticker, (None, None))
                    if entry_price:
                        trades.append({
                            "ticker": ticker,
                            "entry_date": entry_date,
                            "exit_date": date,
                            "entry_price": entry_price,
                            "exit_price": price,
                            "return_pct": price / entry_price - 1,
                        })

                new_shares = target_value / price
                if new_shares > 0:
                    new_holdings[ticker] = new_shares

            cost = traded_value * cost_rate
            total_cost_paid += cost
            portfolio_value -= cost

            holdings = new_holdings
            cash = portfolio_value - sum(
                shares * day_prices.get(t, 0) for t, shares in holdings.items()
            )

        history.append((date, portfolio_value))

    # Close out any still-open positions at the final date, so they count
    # toward trade statistics too (otherwise long-held winners would be
    # invisible to the accuracy/gain-loss numbers).
    final_date = active_dates[-1]
    final_prices = prices.loc[final_date]
    for ticker, (entry_date, entry_price) in entry_info.items():
        exit_price = final_prices.get(ticker, None)
        if exit_price is not None and pd.notna(exit_price):
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": final_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": exit_price / entry_price - 1,
            })

    history_df = pd.DataFrame(history, columns=["date", "portfolio_value"]).set_index("date")
    trades_df = pd.DataFrame(trades)
    return history_df, total_cost_paid, trades_df


if __name__ == "__main__":
    prices = load_price_series()
    rebalance_dates = get_rebalance_dates(prices)
    print(f"Rebalancing on {len(rebalance_dates)} dates:")
    print([d.date() for d in rebalance_dates])

    history, total_cost, trades = run_backtest(prices, rebalance_dates)

    final_value = history["portfolio_value"].iloc[-1]
    print(f"\nStart value: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"Final value ({history.index[-1].date()}): Rs {final_value:,.0f}")
    print(f"Total Net PNL: Rs {final_value - INITIAL_CAPITAL:,.0f}")
    print(f"Total transaction costs paid: Rs {total_cost:,.0f}")

    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]
    accuracy = len(wins) / len(trades) if len(trades) else 0
    avg_win = wins["return_pct"].mean() if len(wins) else 0
    avg_loss = abs(losses["return_pct"].mean()) if len(losses) else 0
    gain_to_loss = avg_win / avg_loss if avg_loss else float("nan")

    print(f"\nTotal completed trades: {len(trades)}")
    print(f"Trades per stock (avg): {len(trades) / trades['ticker'].nunique():.2f}" if len(trades) else "")
    print(f"Accuracy (profitable trades): {accuracy * 100:.1f}%")
    print(f"Average winning trade: {avg_win * 100:.2f}%")
    print(f"Average losing trade: {-avg_loss * 100:.2f}%")
    print(f"Gain-to-loss ratio: {gain_to_loss:.2f}")

    history.to_csv("data/backtest_history.csv")
    trades.to_csv("data/trades.csv", index=False)
    print("\nSaved portfolio value history to data/backtest_history.csv")
    print("Saved full trade log to data/trades.csv")
