#!/usr/bin/env python3
"""
Equity Valuation with Portfolio Integration

Single-file executable script.

Usage:
    python equity_valuation_portfolio.py
    python equity_valuation_portfolio.py --tickers AAPL MSFT TSLA --benchmark ^GSPC --start 2022-09-01 \
        --dcf_threshold 0.05 --max_weight 0.5 --verbose

Requirements:
    pandas, numpy, yfinance, scipy, matplotlib, statsmodels

What's changed from the original:
 - Fixed decision labeling so tickers are only labeled "Buy (Undervalued)" when truly undervalued.
 - Improved revenue / shares outstanding retrieval with better fallbacks and debug logging.
 - Added CLI options for DCF undervaluation threshold and maximum weight per asset (to avoid extreme MVO concentration).
 - Cleaner debug/info prints controlled by --verbose flag.
 - Keeps all earlier functionality: DCF, multiples, beta estimation, WACC fallback, equal weight & MVO portfolios,
   metrics (annualized return, vol, Sharpe), historical VaR, efficient frontier and cumulative return plots.

This script is educational — adjust assumptions for production use.
"""
from __future__ import annotations
import argparse
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
from datetime import date, timedelta

warnings.filterwarnings("ignore")
plt.style.use('default')

# ---------------------------
# Defaults and assumptions
# ---------------------------
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA"]
DEFAULT_BENCHMARK = "^GSPC"
DEFAULT_RISK_FREE = 0.03
DEFAULT_MRP = 0.05
DCF_GROWTH = 0.05
OPERATING_MARGIN = 0.20
REINVESTMENT_RATE = 0.30
TAX_RATE = 0.21
TERMINAL_GROWTH = 0.02
DCF_YEARS = 5
DEFAULT_DCF_THRESHOLD = 0.05
ANNUAL_TRADING_DAYS = 252

# ---------------------------
# Data structures
# ---------------------------
@dataclass
class ValuationResult:
    ticker: str
    price: float
    intrinsic_dcf: float
    intrinsic_pe: float
    eps: float
    trailing_pe: float
    shares_outstanding: float
    wacc: float
    beta: float


# ---------------------------
# Utility / Fetching functions
# ---------------------------
def fetch_price_data(tickers: List[str], benchmark: str, start: str, verbose: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    tickers_all = [t for t in tickers if t != benchmark] + [benchmark]
    if verbose:
        print(f"Downloading price data for: {tickers_all} from {start}")
    df = yf.download(tickers_all, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if benchmark not in df.columns:
        # sometimes yfinance returns benchmark under slightly different key for indices
        pass
    bench = df[benchmark].dropna() if benchmark in df.columns else df.iloc[:, -1].dropna()
    # Filter tickers that have data
    available = [t for t in tickers if t in df.columns and df[t].dropna().size > 0]
    if len(available) < len(tickers):
        missing = set(tickers) - set(available)
        print(f"Warning: No price data for {missing}. They will be skipped.")
    df = df[available + [benchmark]].dropna(how="all")
    return df[available], df[benchmark] if benchmark in df.columns else bench


def get_company_info(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    return {"info": info, "ticker_obj": tk}


def annualize_return(mean_daily_ret: float) -> float:
    return (1 + mean_daily_ret) ** ANNUAL_TRADING_DAYS - 1


def annualize_vol(sd_daily: float) -> float:
    return sd_daily * np.sqrt(ANNUAL_TRADING_DAYS)


# ---------------------------
# Financial calculations
# ---------------------------
def estimate_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if df.shape[0] < 10:
        return 1.0
    y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(y, X).fit()
    beta = float(model.params[1])
    return beta


def compute_wacc(info: Dict, beta: float, rf: float, mrp: float) -> float:
    re = rf + beta * mrp
    market_cap = info.get("marketCap") or info.get("market_cap") or None
    total_debt = info.get("totalDebt") or info.get("longTermDebt") or None
    rd = info.get("costOfDebt") or 0.04
    try:
        if market_cap and total_debt:
            E = float(market_cap)
            D = float(total_debt)
            wacc = (E / (E + D)) * re + (D / (E + D)) * rd * (1 - TAX_RATE)
            return float(wacc)
    except Exception:
        pass
    return float(re)


def simple_dcf_valuation(
    ticker: str,
    info: Dict,
    last_revenue: float,
    shares_outstanding: float,
    wacc: float,
    years: int = DCF_YEARS,
    growth: float = DCF_GROWTH,
    op_margin: float = OPERATING_MARGIN,
    reinvestment_rate: float = REINVESTMENT_RATE,
    tax_rate: float = TAX_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
) -> float:
    # Attempt to produce a reasonable intrinsic value per share; many fallbacks provided.
    # If revenue missing -> try market cap * factor; if shares missing -> try marketCap/price
    market_cap = info.get("marketCap") or None
    last_price = info.get("currentPrice") or info.get("previousClose") or None

    if last_revenue is None or last_revenue <= 0:
        if market_cap:
            # approximate revenue as fraction of market cap (industry-dependent)
            last_revenue = float(market_cap) * 0.25
        else:
            last_revenue = 1.0e9  # fallback
    if shares_outstanding is None or shares_outstanding <= 0:
        if market_cap and last_price:
            try:
                shares_outstanding = float(market_cap) / float(last_price)
            except Exception:
                shares_outstanding = 1.0e9
        else:
            shares_outstanding = 1.0e9

    # Build FCF forecast (very simplified)
    rev = float(last_revenue)
    fcfs = []
    for _ in range(years):
        rev = rev * (1 + growth)
        nopat = rev * op_margin * (1 - tax_rate)
        fcf = nopat * (1 - reinvestment_rate)
        fcfs.append(float(fcf))

    # Terminal value (Gordon)
    fcf_terminal = fcfs[-1] * (1 + terminal_growth)
    if wacc - terminal_growth <= 0:
        terminal_value = fcf_terminal / 0.01
    else:
        terminal_value = fcf_terminal / (wacc - terminal_growth)

    discounted = 0.0
    for i, f in enumerate(fcfs, 1):
        discounted += f / ((1 + wacc) ** i)
    discounted += terminal_value / ((1 + wacc) ** years)

    intrinsic_total = discounted
    intrinsic_per_share = intrinsic_total / float(shares_outstanding)
    return float(intrinsic_per_share)


def multiples_valuation(info: Dict, eps: float, assumed_peer_pe: float = 18.0) -> Tuple[float, float]:
    trailing_pe = info.get("trailingPE") or None
    intrinsic_pe = np.nan
    if eps is not None and not np.isnan(eps):
        if trailing_pe:
            intrinsic_pe = float(eps) * float(trailing_pe)
        else:
            intrinsic_pe = float(eps) * assumed_peer_pe
    return intrinsic_pe, trailing_pe or np.nan


# ---------------------------
# Portfolio construction & optimization
# ---------------------------
def pick_undervalued(results: List[ValuationResult], threshold: float = DEFAULT_DCF_THRESHOLD) -> List[ValuationResult]:
    undervalued = []
    for r in results:
        try:
            if r.intrinsic_dcf and r.price and (r.intrinsic_dcf > r.price * (1 + threshold)):
                undervalued.append(r)
        except Exception:
            continue
    return undervalued


def equal_weight_portfolio(tickers: List[str]) -> Dict[str, float]:
    n = len(tickers)
    if n == 0:
        return {}
    return {t: 1.0 / n for t in tickers}


def optimize_portfolio_mvo(returns: pd.DataFrame, rf: float, max_weight: float = 1.0, verbose: bool = False) -> Dict[str, float]:
    tickers = list(returns.columns)
    n = len(tickers)
    if n == 0:
        return {}
    mean_daily = returns.mean()
    mean_annual = (1 + mean_daily) ** ANNUAL_TRADING_DAYS - 1
    cov_annual = returns.cov() * ANNUAL_TRADING_DAYS

    # Ensure max_weight is feasible
    if max_weight <= 0 or max_weight > 1.0:
        max_weight = 1.0
    if max_weight * n < 1.0:
        if verbose:
            print(f"Warning: max_weight {max_weight:.2f} too small for {n} assets (sum < 1). Ignoring max_weight.")
        max_weight = 1.0

    def objective(w):
        w = np.array(w)
        port_ret = np.dot(mean_annual.values, w)
        port_vol = np.sqrt(w @ cov_annual.values @ w.T)
        if port_vol == 0:
            return 1e6
        sharpe = (port_ret - rf) / port_vol
        return -sharpe

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, max_weight) for _ in range(n))
    x0 = np.array([1.0 / n] * n)
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        if verbose:
            print("MVO optimization failed (fallback to equal weight):", res.message)
        return equal_weight_portfolio(tickers)
    weights = {tickers[i]: float(max(0.0, res.x[i])) for i in range(n)}
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}
    return weights


# ---------------------------
# Portfolio analysis metrics
# ---------------------------
def portfolio_daily_returns(weights: Dict[str, float], returns: pd.DataFrame) -> pd.Series:
    common = [c for c in returns.columns if c in weights]
    if not common:
        return pd.Series(dtype=float)
    w = np.array([weights[c] for c in common])
    sub = returns[common].fillna(0.0)
    daily = sub.dot(w)
    return daily


def compute_metrics(daily_returns: pd.Series, rf: float) -> Dict:
    if daily_returns.shape[0] == 0:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "VaR95": np.nan}
    mean_daily = daily_returns.mean()
    ann_return = annualize_return(mean_daily)
    ann_vol = annualize_vol(daily_returns.std())
    sharpe = (ann_return - rf) / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
    var95 = -np.percentile(daily_returns.dropna(), 5) if daily_returns.dropna().size > 0 else np.nan
    return {"annual_return": ann_return, "annual_vol": ann_vol, "sharpe": sharpe, "VaR95": var95}


# ---------------------------
# Visualization
# ---------------------------
def plot_efficient_frontier(returns: pd.DataFrame, rf: float, title="Efficient Frontier", filename="efficient_frontier.png"):
    n = returns.shape[1]
    if n == 0:
        return
    mean_daily = returns.mean()
    mean_annual = (1 + mean_daily) ** ANNUAL_TRADING_DAYS - 1
    cov_annual = returns.cov() * ANNUAL_TRADING_DAYS

    trials = 4000
    rets = []
    vols = []
    sharpes = []
    for _ in range(trials):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        port_ret = np.dot(mean_annual.values, weights)
        port_vol = np.sqrt(weights @ cov_annual.values @ weights.T)
        rets.append(port_ret)
        vols.append(port_vol)
        sharpes.append((port_ret - rf) / port_vol if port_vol > 0 else 0)

    rets = np.array(rets)
    vols = np.array(vols)
    sharpes = np.array(sharpes)

    plt.figure(figsize=(8, 6))
    plt.scatter(vols, rets, c=sharpes, cmap='viridis', marker='.', alpha=0.6)
    plt.colorbar(label='Sharpe')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Efficient frontier saved to {filename}")


def plot_cumulative(portfolio_daily: pd.Series, benchmark_daily: pd.Series, filename="cumulative_returns.png"):
    if portfolio_daily.shape[0] == 0 or benchmark_daily.shape[0] == 0:
        return
    pf_cum = (1 + portfolio_daily).cumprod()
    bm_cum = (1 + benchmark_daily).cumprod()
    df = pd.concat([pf_cum, bm_cum], axis=1).dropna()
    df.columns = ["Portfolio", "Benchmark"]
    plt.figure(figsize=(10, 6))
    df["Portfolio"].plot(label="Portfolio")
    df["Benchmark"].plot(label="Benchmark")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio vs Benchmark Cumulative Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Cumulative returns plot saved to {filename}")


# ---------------------------
# Main workflow
# ---------------------------
def run_workflow(
    tickers: List[str],
    benchmark: str,
    start: str,
    rf: float,
    mrp: float,
    dcf_threshold: float,
    max_weight: float,
    verbose: bool = False,
):
    if verbose:
        print("Fetching price data...")
    price_df, bench_series = fetch_price_data(tickers, benchmark, start, verbose=verbose)
    if price_df.shape[1] == 0:
        print("No tickers available. Exiting.")
        return

    returns = price_df.pct_change().dropna(how="all")
    bench_returns = bench_series.pct_change().dropna()

    results: List[ValuationResult] = []
    if verbose:
        print("Valuing each company (DCF + Multiples)...")

    for t in price_df.columns:
        if verbose:
            print(f"Processing {t}...")
        company = get_company_info(t)
        info = company["info"] or {}
        tk_obj = company.get("ticker_obj")

        # price
        last_price = None
        try:
            if price_df[t].dropna().size > 0:
                last_price = float(price_df[t].dropna().iloc[-1])
            else:
                last_price = float(info.get("currentPrice") or info.get("previousClose") or np.nan)
        except Exception:
            last_price = float(info.get("currentPrice") or info.get("previousClose") or np.nan)

        # shares outstanding
        shares = info.get("sharesOutstanding") or info.get("shares_outstanding") or None
        if (not shares or shares <= 0) and info.get("marketCap") and last_price:
            try:
                shares = float(info.get("marketCap")) / float(last_price)
            except Exception:
                shares = None

        # revenue extraction attempts
        revenue = info.get("totalRevenue") or info.get("total_revenue") or None
        if revenue is None and tk_obj is not None:
            try:
                fin = tk_obj.financials
                if fin is not None and hasattr(fin, "index"):
                    for label in ["Total Revenue", "TotalRevenue", "Revenue", "Total revenues", "Revenues"]:
                        if label in fin.index:
                            try:
                                revenue = float(fin.loc[label].iloc[0])
                                break
                            except Exception:
                                continue
            except Exception:
                revenue = None

        # Last resort revenue fallback using marketCap
        market_cap = info.get("marketCap") or None
        if revenue is None:
            if market_cap:
                revenue = float(market_cap) * 0.25
            else:
                revenue = None

        # eps
        eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths") or None
        try:
            eps = float(eps) if eps is not None else np.nan
        except Exception:
            eps = np.nan

        # beta estimation
        beta = 1.0
        try:
            if t in returns.columns and bench_returns.size > 0:
                beta = estimate_beta(returns[t], bench_returns.reindex(returns.index))
        except Exception:
            beta = 1.0

        wacc = compute_wacc(info, beta, rf, mrp)

        intrinsic_dcf = simple_dcf_valuation(t, info, revenue, shares, wacc)
        intrinsic_pe, trailing_pe = multiples_valuation(info, eps, assumed_peer_pe=18.0)

        results.append(ValuationResult(
            ticker=t,
            price=last_price if last_price is not None else np.nan,
            intrinsic_dcf=intrinsic_dcf if intrinsic_dcf is not None else np.nan,
            intrinsic_pe=intrinsic_pe if intrinsic_pe is not None else np.nan,
            eps=eps if eps is not None else np.nan,
            trailing_pe=trailing_pe if trailing_pe is not None else np.nan,
            shares_outstanding=shares if shares is not None else np.nan,
            wacc=wacc,
            beta=beta
        ))

        if verbose:
            print(f"  debug: {t} price={last_price:.2f} shares={shares if shares else 'NA'} revenue={revenue if revenue else 'NA'} beta={beta:.2f} wacc={wacc:.4f}")

    # DataFrame summary
    df_val = pd.DataFrame([r.__dict__ for r in results]).set_index("ticker")
    pd.set_option("display.float_format", "{:,.2f}".format)
    print("\nValuation summary (first rows):")
    print(df_val[["price", "intrinsic_dcf", "intrinsic_pe", "eps", "trailing_pe", "wacc", "beta"]])

    # Pick undervalued by DCF
    undervalued = pick_undervalued(results, threshold=dcf_threshold)
    undervalued_tickers = [v.ticker for v in undervalued]
    print(f"\nUndervalued stocks by DCF (> {dcf_threshold*100:.1f}% above market): {undervalued_tickers}")

    # Decide selected set & decisions mapping
    if len(undervalued_tickers) == 0:
        print("No undervalued stocks found by DCF threshold. Using all available tickers for portfolio construction.")
        selected = list(price_df.columns)
        decision_map = {t: "Selected (no undervalued found)" for t in selected}
    else:
        selected = undervalued_tickers
        decision_map = {t: "Buy (Undervalued)" for t in selected}

    print(f"Selected tickers for portfolio: {selected}")

    # Prepare returns for selected
    returns_sub = returns[selected].dropna(how="all").fillna(0.0)

    # Equal weight portfolio
    eq_weights = equal_weight_portfolio(selected)
    eq_daily = portfolio_daily_returns(eq_weights, returns_sub)
    eq_metrics = compute_metrics(eq_daily, rf)

    # MVO portfolio with max_weight constraint
    mvo_weights = optimize_portfolio_mvo(returns_sub, rf, max_weight=max_weight, verbose=verbose)
    mvo_daily = portfolio_daily_returns(mvo_weights, returns_sub)
    mvo_metrics = compute_metrics(mvo_daily, rf)

    # Benchmark alignment
    bench_daily_aligned = bench_returns.reindex(returns.index).fillna(0.0)
    bench_metrics = compute_metrics(bench_daily_aligned, rf)

    # Plots
    if verbose:
        print("Generating plots...")
    plot_efficient_frontier(returns_sub, rf)
    plot_cumulative(mvo_daily.reindex(bench_daily_aligned.index).fillna(0.0), bench_daily_aligned)

    # Final summary table
    summary_table = df_val.copy()
    summary_table["decision"] = [decision_map.get(t, "Skip") for t in summary_table.index]
    print("\nDetailed summary table:")
    print(summary_table[["price", "intrinsic_dcf", "intrinsic_pe", "eps", "trailing_pe", "wacc", "beta", "decision"]])

    # Allocations
    print("\nSuggested allocations:")
    print("Equal-weight allocation:")
    for k, v in eq_weights.items():
        print(f"  {k}: {v:.2%}")
    print("MVO-optimized allocation:")
    for k, v in mvo_weights.items():
        print(f"  {k}: {v:.2%}")

    # Performance summary table
    perf_df = pd.DataFrame({
        "Portfolio": ["Equal Weight", "MVO Optimized", "Benchmark"],
        "Annual Return": [eq_metrics["annual_return"], mvo_metrics["annual_return"], bench_metrics["annual_return"]],
        "Annual Vol": [eq_metrics["annual_vol"], mvo_metrics["annual_vol"], bench_metrics["annual_vol"]],
        "Sharpe": [eq_metrics["sharpe"], mvo_metrics["sharpe"], bench_metrics["sharpe"]],
        "VaR95 (1-day)": [eq_metrics["VaR95"], mvo_metrics["VaR95"], bench_metrics["VaR95"]],
    })
    print("\nPerformance summary:")
    print(perf_df.round(4).to_string(index=False))

    # Final textual summary
    undervalued_list = undervalued_tickers if undervalued_tickers else ["None"]
    print("\n" + "=" * 60)
    print("FINAL SUMMARY".center(60))
    print("=" * 60)
    print(f"Based on a simple {DCF_YEARS}-year DCF (growth={DCF_GROWTH*100:.1f}%, op margin={OPERATING_MARGIN*100:.1f}%, reinvestment={REINVESTMENT_RATE*100:.1f}%) and CAPM (rf={rf*100:.2f}%, MRP={mrp*100:.2f}%), the following stocks are considered undervalued:")
    print(", ".join(undervalued_list))
    print()
    print("Suggested allocation:")
    if selected and selected != ["None"]:
        print(" - Equal-weight allocation: each selected ticker receives equal weight.")
        print(" - Mean-variance optimized allocation (long-only) provided above (MVO Optimized allocation).")
    else:
        print(" - No selected tickers; consider holding cash or broad market exposure.")
    print()
    print("Portfolio risk-return vs Benchmark (annualized):")
    print(f" - MVO Portfolio: Return {mvo_metrics['annual_return']:.2%}, Volatility {mvo_metrics['annual_vol']:.2%}, Sharpe {mvo_metrics['sharpe']:.2f}")
    print(f" - Benchmark:    Return {bench_metrics['annual_return']:.2%}, Volatility {bench_metrics['annual_vol']:.2%}, Sharpe {bench_metrics['sharpe']:.2f}")
    print()
    print(f"95% Historical VaR (1-day) for MVO portfolio: {mvo_metrics['VaR95']:.2%}")
    print("=" * 60)
    print("Notes & caveats:")
    print(" - This is a simplified educational implementation. Real IB/PM valuation uses detailed financials and scenario analysis.")
    print(" - If any company had missing financials, reasonable fallbacks were used (see debug prints if --verbose).")
    print(" - Tune DCF inputs (revenues, margins, reinvestment, WACC) and peer multiples for better results.")
    print("=" * 60)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Equity Valuation with Portfolio Integration (Updated)")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of stock tickers")
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark symbol (e.g., ^GSPC)")
    p.add_argument("--start", default=None, help="Start date for historical data (YYYY-MM-DD). Defaults to 3 years ago.")
    p.add_argument("--rf", type=float, default=DEFAULT_RISK_FREE, help="Risk-free rate (decimal, e.g., 0.03)")
    p.add_argument("--mrp", type=float, default=DEFAULT_MRP, help="Market risk premium (decimal, e.g., 0.05)")
    p.add_argument("--dcf_threshold", type=float, default=DEFAULT_DCF_THRESHOLD, help="DCF undervaluation threshold (decimal, e.g., 0.05)")
    p.add_argument("--max_weight", type=float, default=1.0, help="Maximum weight per asset in MVO (decimal, e.g., 0.5)")
    p.add_argument("--verbose", action="store_true", help="Verbose debug prints")
    return p.parse_args()


def default_start_date(years=3):
    today = date.today()
    try:
        start = today.replace(year=today.year - years)
    except Exception:
        start = today - timedelta(days=365 * years)
    return start.isoformat()


if __name__ == "__main__":
    args = parse_args()
    start_date = args.start or default_start_date(3)
    try:
        run_workflow(args.tickers, args.benchmark, start_date, args.rf, args.mrp, args.dcf_threshold, args.max_weight, verbose=args.verbose)
    except Exception as exc:
        print("An error occurred during execution:", str(exc))
        sys.exit(1)
