"""
PCA residual z-score backtest.

Strategy summary:
- Use PCA loadings (betas) from a component CSV.
- Reconstruct predicted returns for each stock from retained PCs.
- Compute residuals and standardize by 20-day rolling residual volatility.
- On the final 20% of the return history:
    - Enter short when z-score >= +1.5 and long when z-score <= -1.5.
    - Exit short when z-score <= +0.5 and exit long when z-score >= -0.5.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty50_stat_arb.pca import compute_pca


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RETURNS_PATH = os.path.join(PROJECT_ROOT, "data", "nifty50", "returns.csv")
DEFAULT_PCA_COMPONENTS_PATH = os.path.join(PROJECT_ROOT, "data", "nifty50", "pca_components.csv")


@dataclass
class BacktestConfig:
    returns_path: str = DEFAULT_RETURNS_PATH
    pca_components_path: str = DEFAULT_PCA_COMPONENTS_PATH
    train_fraction: float = 0.8
    lookback: int = 60
    long_entry_z: float = -1.5
    short_entry_z: float = 1.5
    long_exit_z: float = 0.0
    short_exit_z: float = 0.0
    refit_months: int = 6
    trade_months: int = 6


def load_returns(path: str) -> pd.DataFrame:
    """Load returns CSV indexed by date."""
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns.sort_index(inplace=True)
    return returns


def load_betas(path: str) -> pd.DataFrame:
    """Load PCA component loadings from CSV."""
    components = pd.read_csv(path)
    required = {"component", "cumulative_variance_pct"}
    missing = required.difference(components.columns)
    if missing:
        raise ValueError(f"PCA components file missing columns: {sorted(missing)}")

    meta_cols = {
        "component",
        "eigenvalue",
        "explained_variance_pct",
        "cumulative_variance_pct",
    }
    asset_cols = [c for c in components.columns if c not in meta_cols]
    if not asset_cols:
        raise ValueError("No asset loading columns found in PCA components file")

    betas = components[asset_cols].copy()
    betas.index = components["component"].astype(str)
    return betas


def compute_predicted_returns(returns: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    """Project daily returns onto PCA components and reconstruct predicted returns."""
    common_assets = [c for c in returns.columns if c in betas.columns]
    if not common_assets:
        raise ValueError("No overlapping assets between returns and PCA component file")

    r = returns[common_assets].copy()
    b = betas[common_assets].to_numpy()  # shape: K x N

    # Daily PC scores: K factors inferred from cross-sectional returns each day.
    factor_scores = r.to_numpy() @ b.T  # shape: T x K
    predicted = factor_scores @ b  # shape: T x N

    return pd.DataFrame(predicted, index=r.index, columns=common_assets)


def _build_betas_from_ranked_table(ranked_table: pd.DataFrame) -> pd.DataFrame:
    """Convert PCA ranked table to the component-loading matrix used by the backtest."""
    meta_cols = {
        "component",
        "eigenvalue",
        "explained_variance_pct",
        "cumulative_variance_pct",
    }
    asset_cols = [c for c in ranked_table.columns if c not in meta_cols]
    betas = ranked_table[asset_cols].copy()
    betas.index = ranked_table["component"].astype(str)
    return betas


def _get_refit_trade_blocks(
    dates: pd.DatetimeIndex,
    refit_months: int,
    trade_months: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Create rolling calendar blocks: fit on M months, trade next N months."""
    if len(dates) == 0:
        return []

    blocks: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    fit_start = dates[0]

    while True:
        fit_end = fit_start + pd.DateOffset(months=refit_months)
        trade_end = fit_end + pd.DateOffset(months=trade_months)

        fit_dates = dates[(dates >= fit_start) & (dates < fit_end)]
        trade_dates = dates[(dates >= fit_end) & (dates < trade_end)]

        if len(fit_dates) < 2 or len(trade_dates) < 2:
            break

        blocks.append((fit_dates, trade_dates))

        # Roll forward exactly by the traded window as requested.
        fit_start = fit_end

    return blocks


def run_backtest_on_df(returns: pd.DataFrame, config: BacktestConfig, verbose: bool = True) -> pd.DataFrame:
    """Core rolling refit / trade PCA residual strategy on a pre-loaded returns DataFrame."""
    blocks = _get_refit_trade_blocks(
        returns.index,
        refit_months=config.refit_months,
        trade_months=config.trade_months,
    )

    if not blocks:
        return pd.DataFrame(columns=["strategy_return", "cumulative_return", "long_count", "short_count"])

    portfolio_returns: list[float] = []
    long_counts: list[int] = []
    short_counts: list[int] = []
    pnl_dates: list[pd.Timestamp] = []

    for block_num, (fit_dates, trade_dates) in enumerate(blocks, start=1):
        fit_returns = returns.loc[fit_dates]

        _, _, ranked_table = compute_pca(
            fit_returns,
            train_fraction=1.0,
            variance_threshold=0.99,
        )
        betas = _build_betas_from_ranked_table(ranked_table)

        # Compute z-scores with context from fit + trade to preserve rolling-vol logic.
        block_slice = returns.loc[fit_dates[0]:trade_dates[-1]].copy()
        predicted = compute_predicted_returns(block_slice, betas)
        aligned_returns = block_slice[predicted.columns].copy()
        residuals = aligned_returns - predicted
        rolling_std = residuals.rolling(config.lookback).std().shift(1)
        zscores = residuals / rolling_std

        test_dates = [d for d in trade_dates if d in zscores.index]
        if len(test_dates) < 2:
            continue

        test_zscores = zscores.loc[test_dates].copy()
        test_returns = aligned_returns.loc[test_dates].copy()

        assets = list(test_zscores.columns)
        positions = {asset: 0 for asset in assets}  # +1 long, -1 short, 0 flat

        if verbose:
            print(
                f"\nBlock {block_num}: fit {fit_dates[0].date()} to {fit_dates[-1].date()} | "
                f"trade {test_dates[0].date()} to {test_dates[-1].date()}"
            )

        for i, date in enumerate(test_dates):
            day_z = test_zscores.loc[date].dropna()
            if day_z.empty:
                continue

            for asset, pos in list(positions.items()):
                z_value = day_z.get(asset, np.nan)
                if np.isnan(z_value):
                    continue

                # Exit rules by z-score reversion.
                if pos == 1 and z_value >= config.long_exit_z:
                    if verbose:
                        print(f"{date.date()} EXIT LONG  {asset:12s} z={z_value: .4f}")
                    positions[asset] = 0
                elif pos == -1 and z_value <= config.short_exit_z:
                    if verbose:
                        print(f"{date.date()} EXIT SHORT {asset:12s} z={z_value: .4f}")
                    positions[asset] = 0

            for asset in day_z.index:
                if positions[asset] != 0:
                    continue

                z_value = day_z[asset]
                if z_value <= config.long_entry_z:
                    positions[asset] = 1
                    if verbose:
                        print(f"{date.date()} ENTER LONG {asset:12s} z={z_value: .4f}")
                elif z_value >= config.short_entry_z:
                    positions[asset] = -1
                    if verbose:
                        print(f"{date.date()} ENTER SHORT {asset:12s} z={z_value: .4f}")

            # Apply end-of-day positions to next-day return.
            if i + 1 < len(test_dates):
                next_date = test_dates[i + 1]
                next_ret = test_returns.loc[next_date]

                long_assets = [asset for asset, pos in positions.items() if pos == 1]
                short_assets = [asset for asset, pos in positions.items() if pos == -1]

                long_pnl = next_ret[long_assets].mean() if long_assets else 0.0
                short_pnl = -next_ret[short_assets].mean() if short_assets else 0.0
                day_pnl = 0.5 * long_pnl + 0.5 * short_pnl

                portfolio_returns.append(float(day_pnl))
                long_counts.append(len(long_assets))
                short_counts.append(len(short_assets))
                pnl_dates.append(next_date)

    pnl = pd.Series(portfolio_returns, index=pnl_dates, name="strategy_return")
    cumulative = (1.0 + pnl).cumprod() - 1.0
    return pd.DataFrame({
        "strategy_return": pnl,
        "cumulative_return": cumulative,
        "long_count": long_counts,
        "short_count": short_counts,
    })


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    """Run rolling refit / trade PCA residual strategy."""
    returns = load_returns(config.returns_path)
    result = run_backtest_on_df(returns, config, verbose=True)
    return result

def run_backtest_baseline_on_df(returns: pd.DataFrame, config: BacktestConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Index mean-reversion baseline strategy on a pre-loaded returns DataFrame.
    
    Instead of PCA residuals per asset, we use a single index-level z-score:
    - Index return = mean return across all assets
    - z-score = (index_return - rolling_mean) / rolling_std
    - Entry/exit rules apply to the entire index (all assets held equally)
    """
    blocks = _get_refit_trade_blocks(
        returns.index,
        refit_months=config.refit_months,
        trade_months=config.trade_months,
    )

    if not blocks:
        return pd.DataFrame(columns=["strategy_return", "cumulative_return", "long_count", "short_count"])

    portfolio_returns: list[float] = []
    long_counts: list[int] = []
    short_counts: list[int] = []
    pnl_dates: list[pd.Timestamp] = []

    for block_num, (fit_dates, trade_dates) in enumerate(blocks, start=1):
        block_slice = returns.loc[fit_dates[0]:trade_dates[-1]].copy()
        index_returns = block_slice.mean(axis=1)
        rolling_mean = index_returns.rolling(config.lookback).mean().shift(1)
        rolling_std = index_returns.rolling(config.lookback).std().shift(1)
        zscores = (index_returns - rolling_mean) / rolling_std
        
        test_dates = [d for d in trade_dates if d in zscores.index]
        if len(test_dates) < 2:
            continue

        test_zscores = zscores.loc[test_dates]
        test_returns = block_slice.loc[test_dates]
        position = 0
        
        if verbose:
            print(f"\nBlock {block_num}: trade {test_dates[0].date()} to {test_dates[-1].date()}")

        for i, date in enumerate(test_dates):
            z_value = test_zscores.loc[date]
            if np.isnan(z_value):
                continue

            if position == 1 and z_value >= config.long_exit_z:
                if verbose:
                    print(f"{date.date()} EXIT LONG  (index) z={z_value: .4f}")
                position = 0
            elif position == -1 and z_value <= config.short_exit_z:
                if verbose:
                    print(f"{date.date()} EXIT SHORT (index) z={z_value: .4f}")
                position = 0

            if position == 0:
                if z_value <= config.long_entry_z:
                    position = 1
                    if verbose:
                        print(f"{date.date()} ENTER LONG (index) z={z_value: .4f}")
                elif z_value >= config.short_entry_z:
                    position = -1
                    if verbose:
                        print(f"{date.date()} ENTER SHORT (index) z={z_value: .4f}")

            if i + 1 < len(test_dates):
                next_date = test_dates[i + 1]
                next_ret = test_returns.loc[next_date].mean()
                day_pnl = position * next_ret
                portfolio_returns.append(float(day_pnl))
                long_count = len(test_returns.columns) if position == 1 else 0
                short_count = len(test_returns.columns) if position == -1 else 0
                long_counts.append(long_count)
                short_counts.append(short_count)
                pnl_dates.append(next_date)

    pnl = pd.Series(portfolio_returns, index=pnl_dates, name="strategy_return")
    cumulative = (1.0 + pnl).cumprod() - 1.0
    return pd.DataFrame({
        "strategy_return": pnl,
        "cumulative_return": cumulative,
        "long_count": long_counts,
        "short_count": short_counts,
    })


def summarize_results(results: pd.DataFrame) -> None:
    """Print simple backtest performance summary."""
    if results.empty:
        print("No backtest returns were generated (insufficient test data after lookback).")
        return

    daily = results["strategy_return"]
    total_return = results["cumulative_return"].iloc[-1]
    annualized_return = (1.0 + total_return) ** (252 / len(daily)) - 1.0
    annualized_vol = daily.std(ddof=1) * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan

    print("\nBacktest Summary")
    print(f"Days traded:       {len(daily)}")
    print(f"Total return:      {total_return: .2%}")
    print(f"Annualized return: {annualized_return: .2%}")
    print(f"Annualized vol:    {annualized_vol: .2%}")
    print(f"Sharpe (rf=0):     {sharpe: .3f}")


def main() -> None:
    """CLI entry point for PCA residual z-score backtest."""
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from nifty50_stat_arb.eigenvalue_plotting import plot_position_counts, DEFAULT_NIFTY50_PLOTS_DIR

    parser = argparse.ArgumentParser(description="Run PCA residual z-score backtest")
    parser.add_argument("--returns-path", type=str, default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--pca-components-path", type=str, default=DEFAULT_PCA_COMPONENTS_PATH)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--long-entry-z", type=float, default=-2.5)
    parser.add_argument("--short-entry-z", type=float, default=2.5)
    parser.add_argument("--long-exit-z", type=float, default=-1.5)
    parser.add_argument("--short-exit-z", type=float, default=1.5)
    parser.add_argument(
        "--save-results-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "nifty50", "backtest_results.csv"),
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital in rupees used to convert returns to PnL on the plot (default: 10,00,000)",
    )
    args = parser.parse_args()

    config = BacktestConfig(
        returns_path=args.returns_path,
        pca_components_path=args.pca_components_path,
        train_fraction=args.train_fraction,
        lookback=args.lookback,
        long_entry_z=args.long_entry_z,
        short_entry_z=args.short_entry_z,
        long_exit_z=args.long_exit_z,
        short_exit_z=args.short_exit_z,
    )

    results = run_backtest(config)
    summarize_results(results)

    if args.save_results_path:
        out_dir = os.path.dirname(args.save_results_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        results.to_csv(args.save_results_path)
        print(f"Saved daily backtest results to {args.save_results_path}")

    plot_path = plot_position_counts(results, plots_dir=DEFAULT_NIFTY50_PLOTS_DIR, initial_capital=args.capital)
    print(f"Saved position counts plot to {plot_path}")


if __name__ == "__main__":
    main()