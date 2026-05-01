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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RETURNS_PATH = os.path.join(PROJECT_ROOT, "data", "nifty50_returns.csv")
DEFAULT_PCA_COMPONENTS_PATH = os.path.join(PROJECT_ROOT, "data", "nifty50_pca_components.csv")


@dataclass
class BacktestConfig:
    returns_path: str = DEFAULT_RETURNS_PATH
    pca_components_path: str = DEFAULT_PCA_COMPONENTS_PATH
    train_fraction: float = 0.8
    lookback: int = 60
    long_entry_z: float = -2.0
    short_entry_z: float = 2.0
    long_exit_z: float = -0.5
    short_exit_z: float = 0.5


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


def run_backtest(config: BacktestConfig) -> pd.DataFrame:
    """Run the PCA residual strategy and return daily portfolio returns."""
    returns = load_returns(config.returns_path)
    betas = load_betas(config.pca_components_path)
    predicted = compute_predicted_returns(returns, betas)

    aligned_returns = returns[predicted.columns].copy()
    residuals = aligned_returns - predicted
    rolling_std = residuals.rolling(config.lookback).std().shift(1)
    zscores = residuals / rolling_std

    split_idx = int(len(zscores) * config.train_fraction)
    test_dates = zscores.index[split_idx:]
    test_zscores = zscores.loc[test_dates].copy()
    test_returns = aligned_returns.loc[test_dates].copy()

    assets = list(test_zscores.columns)

    positions = {asset: 0 for asset in assets}  # +1 long, -1 short, 0 flat
    portfolio_returns: list[float] = []
    pnl_dates: list[pd.Timestamp] = []

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
                print(f"{date.date()} EXIT LONG  {asset:12s} z={z_value: .4f}")
                positions[asset] = 0
            elif pos == -1 and z_value <= config.short_exit_z:
                print(f"{date.date()} EXIT SHORT {asset:12s} z={z_value: .4f}")
                positions[asset] = 0

        for asset in day_z.index:
            if positions[asset] != 0:
                continue

            z_value = day_z[asset]
            if z_value <= config.long_entry_z:
                positions[asset] = 1
                print(f"{date.date()} ENTER LONG {asset:12s} z={z_value: .4f}")
            elif z_value >= config.short_entry_z:
                positions[asset] = -1
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
            pnl_dates.append(next_date)

    pnl = pd.Series(portfolio_returns, index=pnl_dates, name="strategy_return")
    cumulative = (1.0 + pnl).cumprod() - 1.0
    result = pd.DataFrame({"strategy_return": pnl, "cumulative_return": cumulative})
    return result


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
    parser = argparse.ArgumentParser(description="Run PCA residual z-score backtest")
    parser.add_argument("--returns-path", type=str, default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--pca-components-path", type=str, default=DEFAULT_PCA_COMPONENTS_PATH)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--long-entry-z", type=float, default=-2.5)
    parser.add_argument("--short-entry-z", type=float, default=2.5)
    parser.add_argument("--long-exit-z", type=float, default=-1.0)
    parser.add_argument("--short-exit-z", type=float, default=1.0)
    parser.add_argument(
        "--save-results-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "pca_backtest_results.csv"),
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


if __name__ == "__main__":
    main()