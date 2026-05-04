#!/usr/bin/env python3
"""
Run the full stat-arb pipeline across multiple Nifty sector indices.

Usage:
    python run_indices.py
    python run_indices.py --start-date 2021-01-01 --end-date 2025-12-31
    python run_indices.py --index nifty_bank nifty_it          # subset only
    python run_indices.py --refresh-cache                      # re-download prices
    python run_indices.py --capital 500000                     # ₹5L starting capital
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from nifty50_stat_arb.pipeline import PipelineConfig, run_pipeline

SYMBOLS_DIR = os.path.join(PROJECT_ROOT, "data", "symbols")

# ---------------------------------------------------------------------------
# Index registry
# Each entry: (index_name, symbols_file_basename)
# ---------------------------------------------------------------------------
ALL_INDICES: list[tuple[str, str]] = [
    ("nifty50",                "nifty50.txt"),
    ("nifty_auto",             "nifty_auto.txt"),
    ("nifty_bank",             "nifty_bank.txt"),
    ("nifty_financial_services", "nifty_financial_services.txt"),
    ("nifty_it",               "nifty_it.txt"),
    ("nifty_healthcare",       "nifty_healthcare.txt"),
    ("nifty_pharma",           "nifty_pharma.txt"),
    ("nifty_oil_gas",          "nifty_oil_gas.txt"),
    ("nifty_fmcg",             "nifty_fmcg.txt"),
]


def build_config(
    index_name: str,
    symbols_filename: str,
    start_date: str | None,
    end_date: str | None,
    period: str,
    capital: float,
) -> PipelineConfig:
    return PipelineConfig(
        index_name=index_name,
        symbols_file=os.path.join(SYMBOLS_DIR, symbols_filename),
        start_date=start_date,
        end_date=end_date,
        period=period,
        initial_capital=capital,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PCA stat-arb pipeline for multiple Nifty sector indices."
    )
    parser.add_argument(
        "--index",
        nargs="+",
        metavar="NAME",
        help=(
            "Run only these indices (by name). "
            f"Available: {', '.join(n for n, _ in ALL_INDICES)}"
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Price history start date (YYYY-MM-DD). Default: 2021-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Price history end date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="yfinance period string used when start/end dates are not given. Default: 5y",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-download of price data even if a cache exists.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Starting capital in rupees for the PnL overlay (default: ₹10,00,000).",
    )
    args = parser.parse_args()

    # Filter to requested indices
    selected = ALL_INDICES
    if args.index:
        available = {n: f for n, f in ALL_INDICES}
        missing = [n for n in args.index if n not in available]
        if missing:
            parser.error(f"Unknown index name(s): {', '.join(missing)}")
        selected = [(n, available[n]) for n in args.index]

    print(f"Running pipeline for {len(selected)} index/indices: {[n for n, _ in selected]}")

    failed: list[str] = []
    for index_name, symbols_file in selected:
        cfg = build_config(
            index_name=index_name,
            symbols_filename=symbols_file,
            start_date=args.start_date,
            end_date=args.end_date,
            period=args.period,
            capital=args.capital,
        )
        try:
            run_pipeline(cfg, refresh_cache=args.refresh_cache)
        except Exception as exc:
            print(f"\n[ERROR] {index_name}: {exc}")
            failed.append(index_name)

    print(f"\n{'='*60}")
    if failed:
        print(f"Completed with errors in: {failed}")
    else:
        print(f"All {len(selected)} pipeline(s) completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
