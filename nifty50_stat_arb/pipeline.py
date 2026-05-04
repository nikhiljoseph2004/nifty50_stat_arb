"""
Index pipeline: fetch → PCA → backtest → plots for any symbol list.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure the package is importable when the file is run directly
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class PipelineConfig:
    """Configuration for a single index pipeline run."""

    # Identification
    index_name: str  # e.g. "nifty_bank"

    # Symbol source — one of these must be set
    symbols_file: Optional[str] = None  # path to .txt file
    symbols: Optional[list[str]] = None  # explicit list

    # Date range for price fetch (pass both or neither)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: str = "5y"  # used only when start_date/end_date are not given

    # Backtest parameters
    train_fraction: float = 0.8
    variance_threshold: float = 0.99
    lookback: int = 60
    long_entry_z: float = -2.0
    short_entry_z: float = 2.0
    long_exit_z: float = -0.5
    short_exit_z: float = 0.5

    # Capital used for the PnL overlay on the position counts plot (₹)
    initial_capital: float = 1_000_000.0

    # Derived paths — auto-populated in __post_init__ if left empty
    data_dir: str = ""
    plots_dir: str = ""

    def __post_init__(self) -> None:
        if not self.data_dir:
            self.data_dir = os.path.join(PROJECT_ROOT, "data", self.index_name)
        if not self.plots_dir:
            self.plots_dir = os.path.join(PROJECT_ROOT, "plots", self.index_name)

    # Convenience path properties
    @property
    def prices_path(self) -> str:
        return os.path.join(self.data_dir, "prices.csv")

    @property
    def returns_path(self) -> str:
        return os.path.join(self.data_dir, "returns.csv")

    @property
    def pca_components_path(self) -> str:
        return os.path.join(self.data_dir, "pca_components.csv")

    @property
    def backtest_results_path(self) -> str:
        return os.path.join(self.data_dir, "backtest_results.csv")


def run_pipeline(cfg: PipelineConfig, refresh_cache: bool = False) -> pd.DataFrame:
    """
    Run the full pipeline for a single index.

    Steps:
        1. Fetch prices (cached to ``cfg.prices_path``)
        2. Compute and save log returns (``cfg.returns_path``)
        3. Run PCA, save components CSV (``cfg.pca_components_path``),
           save eigenvalue profile plot
        4. Run z-score backtest, save results CSV (``cfg.backtest_results_path``),
           save position-counts+PnL plot

    Returns:
        The backtest results DataFrame.
    """
    from nifty50_stat_arb.data_fetcher import DataFetcher
    from nifty50_stat_arb.pca import compute_pca, load_returns
    from nifty50_stat_arb.pca_backtest import BacktestConfig, run_backtest, summarize_results
    from nifty50_stat_arb.eigenvalue_plotting import plot_eigenvalue_profile, plot_position_counts

    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Pipeline: {cfg.index_name}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Fetch prices
    # ------------------------------------------------------------------
    if cfg.symbols_file:
        fetcher = DataFetcher(symbols_file=cfg.symbols_file)
    elif cfg.symbols:
        fetcher = DataFetcher(symbols=cfg.symbols)
    else:
        raise ValueError(f"[{cfg.index_name}] Either symbols_file or symbols must be set.")

    prices = fetcher.fetch_data(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        period=cfg.period,
        cache_path=cfg.prices_path,
        returns_cache_path=cfg.returns_path,
        refresh_cache=refresh_cache,
    )
    print(f"[{cfg.index_name}] Prices: {prices.shape[0]} days x {prices.shape[1]} stocks")

    # ------------------------------------------------------------------
    # 2. PCA
    # ------------------------------------------------------------------
    returns = load_returns(cfg.returns_path)
    _, _, ranked_table = compute_pca(
        returns,
        train_fraction=cfg.train_fraction,
        variance_threshold=cfg.variance_threshold,
    )
    ranked_table.to_csv(cfg.pca_components_path, index=False)
    n_components = len(ranked_table)
    cum_var = ranked_table["cumulative_variance_pct"].iloc[-1]
    print(
        f"[{cfg.index_name}] PCA: {n_components} components -> {cum_var:.2f}% variance "
        f"(saved to {cfg.pca_components_path})"
    )

    # Eigenvalue profile plot
    eig_plot_path = plot_eigenvalue_profile(
        csv_path=cfg.pca_components_path,
        plots_dir=cfg.plots_dir,
    )
    print(f"[{cfg.index_name}] Eigenvalue plot -> {eig_plot_path}")

    # ------------------------------------------------------------------
    # 3. Backtest
    # ------------------------------------------------------------------
    bt_config = BacktestConfig(
        returns_path=cfg.returns_path,
        pca_components_path=cfg.pca_components_path,
        train_fraction=cfg.train_fraction,
        lookback=cfg.lookback,
        long_entry_z=cfg.long_entry_z,
        short_entry_z=cfg.short_entry_z,
        long_exit_z=cfg.long_exit_z,
        short_exit_z=cfg.short_exit_z,
    )
    results = run_backtest(bt_config)
    summarize_results(results)
    results.to_csv(cfg.backtest_results_path)
    print(f"[{cfg.index_name}] Backtest results -> {cfg.backtest_results_path}")

    # Position counts + PnL plot
    pos_plot_path = plot_position_counts(
        results=results,
        plots_dir=cfg.plots_dir,
        initial_capital=cfg.initial_capital,
    )
    print(f"[{cfg.index_name}] Position counts plot -> {pos_plot_path}")

    return results
