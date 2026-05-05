#!/usr/bin/env python3
"""
Coarse-to-fine grid search for PCA residual strategy hyperparameters.

For each sector index:
  1. Load returns and split chronologically into 70 / 15 / 15 (train / val / test).
  2. Run the rolling refit/trade strategy on the TRAIN + VAL slices only.
     - PCA is refit every 6 months within each slice.
     - The Sharpe ratio on the VAL slice is the optimisation objective.
  3. Coarse pass: broad grid over (entry_z, exit_z).
  4. Fine pass: zoom in ±step around the coarse winner with tighter spacing.
  5. Report best params and val Sharpe per index.
  6. Evaluate best params on the held-out TEST slice and report test Sharpe.

Symmetry enforced:
  long_entry_z  = -entry_z
  short_entry_z = +entry_z
  long_exit_z   = +exit_z   (long exits when z reverts above exit_z)
  short_exit_z  = -exit_z   (short exits when z reverts below -exit_z)
  Constraint: exit_z < entry_z

Usage:
    python tune_hyperparams.py
    python tune_hyperparams.py --index nifty_bank nifty_it
    python tune_hyperparams.py --index nifty50 --lookback 60
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from nifty50_stat_arb.pca_backtest import BacktestConfig, load_returns, run_backtest_on_df

# ---------------------------------------------------------------------------
# Index registry (mirrors run_indices.py)
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
# Grid definitions
# ---------------------------------------------------------------------------
COARSE_ENTRY = [1.0, 1.5, 2.0, 2.5, 3.0]        # entry_z candidates
COARSE_EXIT  = [0.0, 0.25, 0.5, 0.75, 1.0]       # exit_z candidates

FINE_STEP    = 0.1                                 # fine grid resolution
FINE_RADIUS  = 0.3                                 # search ± this around coarse best


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
    if results.empty or results["strategy_return"].std(ddof=1) == 0:
        return -np.inf
    daily = results["strategy_return"]
    ann_ret = (1.0 + daily.mean()) ** 252 - 1.0
    ann_vol = daily.std(ddof=1) * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 0 else -np.inf


def _make_config(
    index_name: str,
    entry_z: float,
    exit_z: float,
    lookback: int,
    refit_months: int,
    trade_months: int,
) -> BacktestConfig:
    returns_path = os.path.join(PROJECT_ROOT, "data", index_name, "returns.csv")
    pca_path     = os.path.join(PROJECT_ROOT, "data", index_name, "pca_components.csv")
    return BacktestConfig(
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


def evaluate_on_slice(
    returns_slice: pd.DataFrame,
    config: BacktestConfig,
) -> float:
    """Run backtest on a returns slice and return its Sharpe ratio."""
    results = run_backtest_on_df(returns_slice, config, verbose=False)
    return compute_sharpe(results)


def grid_search(
    val_returns: pd.DataFrame,
    entry_candidates: list[float],
    exit_candidates: list[float],
    config_template: BacktestConfig,
) -> tuple[float, float, float]:
    """
    Exhaustive grid search over (entry_z, exit_z) pairs.
    Returns (best_entry_z, best_exit_z, best_val_sharpe).
    """
    best_entry, best_exit, best_sharpe = entry_candidates[0], exit_candidates[0], -np.inf

    total = sum(1 for e, x in product(entry_candidates, exit_candidates) if x < e)
    done  = 0

    for entry_z, exit_z in product(entry_candidates, exit_candidates):
        if exit_z >= entry_z:          # constraint: exit must be less extreme than entry
            continue

        cfg = BacktestConfig(
            returns_path=config_template.returns_path,
            pca_components_path=config_template.pca_components_path,
            lookback=config_template.lookback,
            long_entry_z=-entry_z,
            short_entry_z=+entry_z,
            long_exit_z=+exit_z,
            short_exit_z=-exit_z,
            refit_months=config_template.refit_months,
            trade_months=config_template.trade_months,
        )

        sharpe = evaluate_on_slice(val_returns, cfg)
        done += 1
        print(
            f"    [{done:>3}/{total}] entry={entry_z:.2f}  exit={exit_z:.2f}"
            f"  val_sharpe={sharpe: .3f}",
            end="\r",
        )

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_entry  = entry_z
            best_exit   = exit_z

    print()  # newline after \r progress
    return best_entry, best_exit, best_sharpe


def fine_grid(
    best_entry: float,
    best_exit: float,
    step: float,
    radius: float,
) -> tuple[list[float], list[float]]:
    """Build a fine grid centred on the coarse best."""
    entry_lo = round(best_entry - radius, 4)
    entry_hi = round(best_entry + radius + step * 0.5, 4)   # inclusive upper
    exit_lo  = round(max(0.0, best_exit - radius), 4)
    exit_hi  = round(best_exit + radius + step * 0.5, 4)

    entry_vals = [round(v, 4) for v in np.arange(entry_lo, entry_hi, step) if v > 0]
    exit_vals  = [round(v, 4) for v in np.arange(exit_lo, exit_hi, step)  if v >= 0]

    return entry_vals, exit_vals


def tune_index(
    index_name: str,
    lookback: int,
    refit_months: int,
    trade_months: int,
) -> dict:
    """Full coarse-to-fine tuning for a single index. Returns a results dict."""
    returns_path = os.path.join(PROJECT_ROOT, "data", index_name, "returns.csv")

    if not os.path.exists(returns_path):
        print(f"  [SKIP] returns file not found: {returns_path}")
        return {}

    returns = load_returns(returns_path)
    train, val, test = split_returns(returns)

    n_total = len(returns)
    n_train = len(train)
    n_val   = len(val)
    n_test  = len(test)
    print(
        f"  Data split: total={n_total} days  "
        f"train={n_train} ({train.index[0].date()} to {train.index[-1].date()})  "
        f"val={n_val} ({val.index[0].date()} to {val.index[-1].date()})  "
        f"test={n_test} ({test.index[0].date()} to {test.index[-1].date()})"
    )

    config_template = _make_config(index_name, 1.5, 0.0, lookback, refit_months, trade_months)

    # ------------------------------------------------------------------
    # COARSE pass — evaluate on val slice
    # ------------------------------------------------------------------
    print("  [Coarse grid]")
    best_entry_c, best_exit_c, best_sharpe_c = grid_search(
        val, COARSE_ENTRY, COARSE_EXIT, config_template
    )
    print(
        f"  Coarse best: entry_z={best_entry_c:.2f}  exit_z={best_exit_c:.2f}"
        f"  val_sharpe={best_sharpe_c:.3f}"
    )

    # ------------------------------------------------------------------
    # FINE pass — zoom in around coarse winner
    # ------------------------------------------------------------------
    print("  [Fine grid]")
    fine_entry_vals, fine_exit_vals = fine_grid(best_entry_c, best_exit_c, FINE_STEP, FINE_RADIUS)
    best_entry_f, best_exit_f, best_sharpe_f = grid_search(
        val, fine_entry_vals, fine_exit_vals, config_template
    )
    print(
        f"  Fine best:   entry_z={best_entry_f:.2f}  exit_z={best_exit_f:.2f}"
        f"  val_sharpe={best_sharpe_f:.3f}"
    )

    # ------------------------------------------------------------------
    # TEST evaluation — held-out, never seen during tuning
    # ------------------------------------------------------------------
    best_cfg = BacktestConfig(
        returns_path=config_template.returns_path,
        pca_components_path=config_template.pca_components_path,
        lookback=lookback,
        long_entry_z=-best_entry_f,
        short_entry_z=+best_entry_f,
        long_exit_z=+best_exit_f,
        short_exit_z=-best_exit_f,
        refit_months=refit_months,
        trade_months=trade_months,
    )
    test_sharpe = evaluate_on_slice(test, best_cfg)
    print(f"  Test Sharpe (held-out): {test_sharpe:.3f}")

    return {
        "index":        index_name,
        "entry_z":      best_entry_f,
        "exit_z":       best_exit_f,
        "val_sharpe":   best_sharpe_f,
        "test_sharpe":  test_sharpe,
        "train_start":  str(train.index[0].date()),
        "train_end":    str(train.index[-1].date()),
        "val_start":    str(val.index[0].date()),
        "val_end":      str(val.index[-1].date()),
        "test_start":   str(test.index[0].date()),
        "test_end":     str(test.index[-1].date()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coarse-to-fine grid search for PCA stat-arb z-score thresholds."
    )
    parser.add_argument(
        "--index",
        nargs="+",
        metavar="NAME",
        help="Run only these indices. Default: all.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Rolling z-score lookback window in trading days (default: 60).",
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
        help="Trade window in months after each refit (default: 6).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "tuning_results.csv"),
        help="Path to save tuning results CSV.",
    )
    args = parser.parse_args()

    selected = {name for name, _ in ALL_INDICES}
    if args.index:
        selected = set(args.index)

    results_rows: list[dict] = []

    for index_name, _ in ALL_INDICES:
        if index_name not in selected:
            continue
        print(f"\n{'=' * 60}")
        print(f"Tuning: {index_name}")
        print(f"{'=' * 60}")
        row = tune_index(
            index_name=index_name,
            lookback=args.lookback,
            refit_months=args.refit_months,
            trade_months=args.trade_months,
        )
        if row:
            results_rows.append(row)

    if not results_rows:
        print("\nNo results to save.")
        return

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    summary = pd.DataFrame(results_rows).set_index("index")
    print(f"\n{'=' * 60}")
    print("TUNING SUMMARY")
    print(f"{'=' * 60}")
    print(summary[["entry_z", "exit_z", "val_sharpe", "test_sharpe"]].to_string())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary.to_csv(args.output)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
