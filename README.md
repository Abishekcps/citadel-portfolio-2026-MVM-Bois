# Finesse x Citadel - Round 2 Portfolio Challenge

This is our submission for the Portfolio Construction Challenge. We built a simple rule-based
strategy that picks 10 stocks every quarter from the Nifty 100 + Midcap 100 + Smallcap 100
universe, based on price momentum and volatility, and backtested it from 2021 to 2025.

Full writeup with reasoning, results, and limitations is in `Round2_Report.docx` - that's the
main document, this README is just to help you run the code.

## Files

- `data_loader.py` - downloads 5 years of stock price data
- `strategy.py` - the actual rule: scores stocks and picks the top 10
- `backtest.py` - simulates the strategy from 2021-2025 with transaction costs
- `metrics.py` - calculates CAGR, Sharpe ratio, drawdown, and compares to Nifty 100
- `Round2_Report.docx` - full report

We didn't upload the price data itself since it's ~300 files and can just be re-downloaded
by running the first script below.

## How to run it

```
pip install yfinance pandas numpy matplotlib
```

Then download these 3 files into a `data/` folder:
- https://niftyindices.com/IndexConstituent/ind_nifty100list.csv
- https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv
- https://niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv

Then just run these in order:

```
python data_loader.py
python strategy.py
python backtest.py
python metrics.py
```

The first one takes a few minutes since it's downloading ~300 stocks.

## Results (short version)

Started with 1 crore, ended with about 2.53 crore by end of 2025 (23.4% CAGR), vs Nifty 100's
12.26%. Full numbers, benchmark comparison, and the out-of-sample test on Jan-Jun 2026 are in
the report.

## Known limitations

- Used current index constituents, not historical ones, so there's some survivorship bias
- One stock (ZYDUSLIFE.NS) couldn't be downloaded, probably a naming issue
- Backtest actually starts July 2021, not Jan 2021, since the strategy needs 6 months of price
  history before it can pick anything

More detail on all of this is in the report.
