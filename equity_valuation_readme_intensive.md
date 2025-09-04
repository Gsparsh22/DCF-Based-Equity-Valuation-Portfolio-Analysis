# Equity Valuation with Portfolio Integration — Intensive README

## Project Overview

This repository contains a single-file Python script (`equity_valuation_portfolio.py`) that performs an end-to-end equity valuation and portfolio construction workflow. The script pulls historical price data, computes company valuations (simple DCF and multiples), selects candidate stocks, constructs portfolios (equal-weight and mean-variance optimized), and evaluates portfolio performance versus a benchmark (S&P 500 by default).

This README is an intensive, practical guide covering installation, execution, model assumptions, interpretation of outputs, troubleshooting, and recommended extensions.

---

## Table of Contents

1. Quick start
2. What the script does (high-level)
3. Installation
4. Files produced
5. Usage & CLI options with examples
6. Core methodology and assumptions
   - Price data
   - Beta & WACC
   - Simple DCF (detailed math and fallbacks)
   - Multiples valuation
   - Portfolio construction (Equal weight & MVO)
   - Performance metrics & VaR
7. Interpreting results (common pitfalls and explanations)
8. Troubleshooting & diagnostics
9. How to improve the model (practical suggestions)
10. Testing & reproducibility
11. Security, rate limits and data concerns
12. License & attribution

---

## 1) Quick start

Assuming you have Python 3.8+ and required libraries installed, run:

```bash
python equity_valuation_portfolio.py
```

Default behavior: uses tickers `AAPL`, `MSFT`, `TSLA`, benchmark `^GSPC` (S&P 500), 3 years of historical data, risk-free rate = 3%, market risk premium = 5%.

To see debug information (revenue/shares used in DCF):

```bash
python equity_valuation_portfolio.py --verbose
```

To cap maximum MVO weight (prevent concentration):

```bash
python equity_valuation_portfolio.py --max_weight 0.5
```

To change DCF undervaluation threshold (e.g., 20%):

```bash
python equity_valuation_portfolio.py --dcf_threshold 0.20
```

Combine flags as needed.

---

## 2) What the script does (high-level)

1. Downloads adjusted close price history for chosen tickers and benchmark using `yfinance`.
2. Calculates daily returns and estimates beta (regression vs benchmark).
3. Computes cost of equity using CAPM: `Re = Rf + Beta * MRP`.
4. Attempts to compute WACC (if debt & market cap available); otherwise falls back to cost of equity.
5. Runs a *simple 5-year DCF* using revenue, operating margin and reinvestment rate assumptions.
6. Performs a *multiples valuation* using EPS × P/E (trailing PE if available, otherwise an assumed peer PE).
7. Flags undervalued stocks (DCF intrinsic > market price × (1 + threshold)). If none flagged, uses all tickers for portfolio construction.
8. Builds two portfolios: equal-weight and mean-variance optimized (long-only) via `scipy.optimize`.
9. Computes performance metrics (annualized return, volatility, Sharpe) and 1-day historical VaR (95%).
10. Saves plots: `efficient_frontier.png` and `cumulative_returns.png` and prints tabular summaries.

---

## 3) Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows PowerShell
pip install --upgrade pip
pip install pandas numpy yfinance scipy matplotlib statsmodels
```

Note: On some systems, installing `statsmodels` may require `patsy` and a C compiler for optional speedups; the pip wheel should work on most platforms.

---

## 4) Files produced by script

- `efficient_frontier.png` — scatter of random portfolios approximating the efficient frontier; colored by Sharpe.
- `cumulative_returns.png` — cumulative returns of the chosen portfolio vs benchmark.
- Console output containing: valuation table, selected tickers, allocation weights, and performance table.

(If you run with `--verbose`, additional diagnostic lines showing revenue/shares/fallbacks will be printed.)

---

## 5) Usage & CLI options (complete)

```
--tickers       List of stock tickers (default: AAPL MSFT TSLA)
--benchmark     Benchmark symbol (default: ^GSPC)
--start         Start date for historical data, format YYYY-MM-DD (default: 3 years ago)
--rf            Risk-free rate (decimal, default 0.03)
--mrp           Market risk premium (decimal, default 0.05)
--dcf_threshold DCF undervaluation threshold (decimal, default 0.05)
--max_weight    Maximum weight per asset for MVO (decimal, default 1.0)
--verbose       Toggle verbose debug prints
```

### Examples

Download 5 years of history and cap MVO weights at 40%:

```bash
python equity_valuation_portfolio.py --start 2020-09-01 --max_weight 0.4
```

Use a different benchmark (e.g., NASDAQ Composite symbol if available):

```bash
python equity_valuation_portfolio.py --benchmark ^IXIC
```

---

## 6) Core methodology & assumptions (detailed)

### Price data
- Data source: `yfinance` (Yahoo Finance). Uses `auto_adjust=True` so `Close` is adjusted for splits/dividends.
- Daily returns are arithmetic percentage changes: `r_t = P_t / P_{t-1} - 1`.

### Beta & WACC
- Beta estimated via ordinary least squares regression: `r_stock = alpha + beta * r_benchmark + eps`.
- Cost of equity: `Re = Rf + Beta * MRP`.
- WACC: when `marketCap` and `totalDebt` exist in `yfinance` `info`, we compute:

```
WACC = (E/(D+E))*Re + (D/(D+E))*Rd*(1 - TaxRate)
```

Otherwise, WACC falls back to `Re` (cost of equity). The script uses a default tax rate of 21%.

### Simple DCF (5-year)
- Top-level assumptions (defaults):
  - Revenue growth = 5% p.a.
  - Operating margin = 20%
  - Reinvestment rate = 30% (implying FCF = NOPAT × (1 - reinvestment_rate))
  - Tax rate = 21%
  - Terminal growth = 2%
  - Forecast horizon = 5 years

- Steps (in script):
  1. Start with `last_revenue` (from `info['totalRevenue']` if available). If missing, fallback to `marketCap * 0.25` or a conservative constant.
  2. Project revenues each year using constant growth.
  3. Compute NOPAT = Revenue × Operating Margin × (1 − Tax Rate).
  4. Compute Free Cash Flow ≈ NOPAT × (1 − Reinvestment Rate).
  5. Discount the FCFs at WACC and compute terminal value via Gordon Growth.
  6. Divide enterprise/intrinsic value by shares outstanding (try `sharesOutstanding` or infer from `marketCap / price`), with fallbacks.

**Important notes:** This is a highly simplified DCF. It is designed for demonstration/education, not production valuations. The fallback logic is pragmatic but can lead to unrealistic per-share values if financials are missing; run with `--verbose` to inspect inputs.

### Multiples valuation
- If `trailingPE` and `trailingEps` are available from `yfinance`, the script computes `MultiplesValue = EPS × trailingPE`. Otherwise it uses an assumed peer P/E (default 18×) with provided EPS.
- Multiples tend to reproduce market price when data comes from trailing metrics.

### Portfolio construction
- **Equal-weight**: each selected ticker gets `1/n` weight.
- **Mean-variance optimization (MVO)**:
  - Long-only, sum of weights = 1, bounds `[0, max_weight]` per asset.
  - Objective: maximize Sharpe (implemented as minimize negative Sharpe): `Sharpe = (portfolio_return - rf) / portfolio_volatility`.
  - Inputs: annualized mean returns and annualized covariance computed from daily returns.

### Performance metrics
- Annualized return: `((1 + mean_daily_return) ** 252) - 1`.
- Annualized volatility: `std_daily * sqrt(252)`.
- Sharpe ratio: `(annual_return - rf) / annual_volatility`.
- 95% Historical VaR (1-day): negative of 5th percentile of the portfolio daily returns.

---

## 7) Interpreting results (common pitfalls)

- **DCF << market price**: Often due to missing revenue / shares data and conservative fallbacks (e.g., `1e9` shares), or because DCF assumptions are too conservative for high-growth firms. Check `--verbose` output.
- **MVO concentrated to a single stock**: A mathematical solution when one asset has superior risk-adjusted returns or covariance structure. Use `--max_weight` to limit concentration or add regularization (e.g., penalize large weights).
- **Multiples match market**: When trailing EPS × trailing PE is used, you often reconstruct market price; that’s expected.
- **WACC equals cost of equity**: Happens when debt data is missing; the script falls back to cost-of-equity-only WACC.

---

## 8) Troubleshooting & diagnostics

### a) No data for a ticker
- `yfinance` may return empty results for some tickers or indices, or rate-limited responses. The script warns if a ticker has no price data and skips it.

### b) Check what inputs DCF used
- Run with `--verbose` to see `debug` lines showing `price`, `shares`, `revenue`, `beta`, and `wacc` for each ticker.

### c) `statsmodels` OLS problems
- If OLS fails due to insufficient data, beta defaults to 1.0. Ensure you have at least several months of overlapping returns.

### d) Plots not saved
- Confirm working directory write permissions; the script saves `efficient_frontier.png` and `cumulative_returns.png` in the current working directory.

---

## 9) How to improve the model (prioritized)

1. **Use actual Free Cash Flow**
   - Replace revenue-based FCF with FCF = Operating Cash Flow − Capital Expenditures from `Ticker().cashflow`.
2. **Extend the forecast**
   - Use 7–10 years for high-growth companies and model multi-stage growth.
3. **Scenario analysis**
   - Run DCF under multiple cases (base, bull, bear) and show a sensitivity table for WACC, terminal growth, margin assumptions.
4. **Add transaction costs & turnover assumptions**
   - For portfolio backtests and realistic performance figures.
5. **Bootstrap / robustify MVO**
   - Use shrinkage estimators for covariance (Ledoit-Wolf) or impose weight regularization to avoid extreme concentration.
6. **Add bootstrapped / parametric VaR and stress tests**
   - Supplement historical VaR with parametric VaR and Conditional VaR (CVaR).
7. **Logging & structured outputs**
   - Output JSON/CSV reports with valuations and allocations for downstream use.

---

## 10) Testing & reproducibility

- Fix the random seed when generating random portfolios for the efficient frontier to make plots deterministic (e.g., `np.random.seed(42)`).
- Save data snapshots (CSV of fetched prices) to allow offline reproducibility.

---

## 11) Security, rate limits and data concerns

- `yfinance` relies on Yahoo Finance; repeated automated requests may hit rate limits. For large-scale or production usage, prefer a paid data provider or cached historical files.
- Do not commit API keys or sensitive info to version control.

---

## 12) License & attribution

This code and README are provided for educational purposes. Adapt and extend at your own risk. No warranty is provided. If you reuse parts of the code or README, attribute the original author.

---

## Appendix: Quick reference of default assumptions

- `rf` (risk-free) = 3.0% (configurable)
- `MRP` (market risk premium) = 5.0%
- DCF growth = 5% p.a.
- Operating margin = 20%
- Reinvestment rate = 30%
- Forecast horizon = 5 years
- Terminal growth = 2%
- Tax rate = 21%

---

If you want, I can:
- Integrate a cashflow-based DCF function into the script (single-file), or
- Produce a shorter `README.md` suitable for GitHub front page, or
- Create unit tests for the valuation and optimization functions.

Tell me which next step you prefer.

