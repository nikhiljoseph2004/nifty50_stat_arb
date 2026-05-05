#!/usr/bin/env python3
"""
Compare PCA strategy vs. index mean-reversion baseline on test set.

For each sector index:
  1. Load tuning results (best entry_z, exit_z from validation).
  2. Split returns chronologically 70 / 15 / 15.
  3. Run PCA strategy on test slice with tuned params.
  4. Run baseline (index mean-reversion) on test slice with same params.
  5. Plot cumulative PnL comparison with Sharpe ratios in legend.

Usage:
    python compare_strategies.py
    python compare_strategies.py --index nifty_bank nifty_it
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from nifty50_stat_arb.pca_backtest import (
    BacktestConfig,
    load_returns,
    run_backtest_on_df,
    run_backtest_baseline_on_df,
)

# ---------------------------------------------------------------------------
# Index registry
# ---------------------------------------------------------------------------
ALL_INDICES: list[tuple[str, str]] = [
    ("nifty50",                  "nifty50.txt"),
    ("nifty_auto",               "nifty_auto.txt"),
    ("nifty_bank",               "nifty_bank.txt"),
    ("nifty_financial_services", "nifty_financial_services.txt"),
    ("nifty_it",                 "nifty_it.txt"),
    ("nifty_healthcare",         "nifty_healthcare.txt"),
    ("nifty_pharma",             "nifty_pharma.txt"),
    ("nifty_oil_gas",            "nifty_oil_gas.txt"),
    ("nifty_fmcg",               "nifty_fmcg.txt"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_returns(
    returns: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70 / 15 / 15 split."""
    n = len(returns)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train = returns.iloc[:train_end]
    val   = returns.iloc[train_end:val_end]
    test  = returns.iloc[val_end:]
    return train, val, test


def compute_sharpe(results: pd.DataFrame) -> float:
    """Annualised Sharpe (rf = 0) from a backtest result DataFrame."""
    if results.empty or len(results) == 0:
        return np.nan
    daily = results["strategy_return"]
    if daily.std(ddof=1) == 0:
        return np.nan
    ann_ret = (1.0 + daily.mean()) ** 252 - 1.0
    ann_vol = daily.std(ddof=1) * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 0 else np.nan


def plot_comparison(
    index_name: str,
    pca_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Plot cumulative PnL comparison between PCA and baseline strategies.
    
    Returns path to saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute Sharpe ratios.
    pca_sharpe = compute_sharpe(pca_results)
    baseline_sharpe = compute_sharpe(baseline_results)
    
    # Create figure.
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot cumulative returns.
    ax.plot(
        pca_results.index,
        pca_results["cumulative_return"] * 100,
        color="#4e79a7",
        linewidth=2.0,
        label=f"PCA Strategy (Sharpe: {pca_sharpe:.3f})",
    )
    ax.plot(
        baseline_results.index,
        baseline_results["cumulative_return"] * 100,
        color="#e15759",
        linewidth=2.0,
        label=f"Baseline Index Mean-Reversion (Sharpe: {baseline_sharpe:.3f})",
    )
    
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(
        pca_results.index,
        pca_results["cumulative_return"] * 100,
        0,
        alpha=0.1,
        color="#4e79a7",
    )
    ax.fill_between(
        baseline_results.index,
        baseline_results["cumulative_return"] * 100,
        0,
        alpha=0.1,
        color="#e15759",
    )
    
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title(f"{index_name} — PCA vs. Baseline on Test Set", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    
    # Save.
    output_file = os.path.join(output_dir, f"{index_name}_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file


def compare_index(
    index_name: str,
    entry_z: float,
    exit_z: float,
    lookback: int,
    refit_months: int,
    trade_months: int,
    output_dir: str,
) -> dict:
    """
    Compare PCA and baseline strategies for a single index on test set.
    Returns a dict with Sharpe comparison results.
    """
    returns_path = os.path.join(PROJECT_ROOT, "data", index_name, "returns.csv")
    pca_path = os.path.join(PROJECT_ROOT, "data", index_name, "pca_components.csv")
    
    if not os.path.exists(returns_path):
        print(f"  [SKIP] returns file not found: {returns_path}")
        return {}
    
    returns = load_returns(returns_path)
    train, val, test = split_returns(returns)
    
    print(f"  Test period: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")
    
    # Create config with tuned parameters (symmetric).
    config = BacktestConfig(
        returns_path=returns_path,
        pca_components_path=pca_path,
        lookback=lookback,
        long_entry_z=-entry_z,
        short_entry_z=+entry_z,
        long_exit_z=+exit_z,
        short_exit_z=-exit_z,
        refit_months=refit_months,
        trade_months=trade_months,
    )
    
    # Run PCA strategy on test.
    print("  Running PCA strategy...")
    pca_results = run_backtest_on_df(test, config, verbose=False)
    pca_sharpe = compute_sharpe(pca_results)
    print(f"    PCA Sharpe: {pca_sharpe:.3f}")
    
    # Run baseline strategy on test.
    print("  Running baseline strategy...")
    baseline_results = run_backtest_baseline_on_df(test, config, verbose=False)
    baseline_sharpe = compute_sharpe(baseline_results)
    print(f"    Baseline Sharpe: {baseline_sharpe:.3f}")
    
    # Plot comparison.
    print("  Generating comparison plot...")
    plot_path = plot_comparison(index_name, pca_results, baseline_results, output_dir)
    print(f"    Saved to {plot_path}")
    
    return {
        "index": index_name,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "pca_sharpe": pca_sharpe,
        "baseline_sharpe": baseline_sharpe,
        "pca_total_return": pca_results["cumulative_return"].iloc[-1] if not pca_results.empty else np.nan,
        "baseline_total_return": baseline_results["cumulative_return"].iloc[-1] if not baseline_results.empty else np.nan,
        "sharpe_diff": pca_sharpe - baseline_sharpe,
        "return_diff": (pca_results["cumulative_return"].iloc[-1] if not pca_results.empty else 0.0) - 
                       (baseline_results["cumulative_return"].iloc[-1] if not baseline_results.empty else 0.0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PCA strategy vs. index mean-reversion baseline on test set."
    )
    parser.add_argument(
        "--index",
        nargs="+",
        metavar="NAME",
        help="Run only these indices. Default: all.",
    )
    parser.add_argument(
        "--tuning-results",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "tuning_results.csv"),
        help="Path to tuning_results.csv from tune_hyperparams.py.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Rolling z-score lookback window (default: 60).",
    )
    parser.add_argument(
        "--refit-months",
        type=int,
        default=6,
        help="PCA refit window in months (default: 6).",
    )
    parser.add_argument(
        "--trade-months",
        type=int,
        default=6,
        help="Trade window in months (default: 6).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "plots"),
        help="Directory to save comparison plots.",
    )
    args = parser.parse_args()
    
    # Load tuning results.
    if not os.path.exists(args.tuning_results):
        print(f"Error: tuning results file not found: {args.tuning_results}")
        sys.exit(1)
    
    tuning_df = pd.read_csv(args.tuning_results, index_col="index")
    
    selected = {name for name, _ in ALL_INDICES}
    if args.index:
        selected = set(args.index)
    
    results_rows: list[dict] = []
    
    for index_name, _ in ALL_INDICES:
        if index_name not in selected:
            continue
        
        if index_name not in tuning_df.index:
            print(f"\n[SKIP] {index_name} not found in tuning results")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Comparing: {index_name}")
        print(f"{'=' * 60}")
        
        tuning_row = tuning_df.loc[index_name]
        entry_z = float(tuning_row["entry_z"])
        exit_z = float(tuning_row["exit_z"])
        
        row = compare_index(
            index_name=index_name,
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=args.lookback,
            refit_months=args.refit_months,
            trade_months=args.trade_months,
            output_dir=args.output_dir,
        )
        
        if row:
            results_rows.append(row)
    
    if not results_rows:
        print("\nNo results to summarize.")
        return
    
    # Summary table.
    summary = pd.DataFrame(results_rows).set_index("index")
    print(f"\n{'=' * 80}")
    print("STRATEGY COMPARISON SUMMARY (Test Set)")
    print(f"{'=' * 80}")
    print(summary[["entry_z", "exit_z", "pca_sharpe", "baseline_sharpe", "sharpe_diff"]].to_string())
    
    summary_path = os.path.join(args.output_dir, "strategy_comparison.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary.to_csv(summary_path)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
