#!/usr/bin/env python3
"""
Main script for fetching and inspecting Nifty 50 price data.
"""

import argparse
import os
import sys

# Ensure the repo root is on the path when running this script directly from misc/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nifty50_stat_arb import DataFetcher


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Nifty 50 data fetch utility')

    parser.add_argument(
        '--period',
        type=str,
        default='1y',
        help='Period for historical data (e.g., 1y, 2y, 5y). Default: 1y'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--cache-path',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nifty50', 'prices.csv'),
        help='Path to cache CSV for fetched price data. Default: data/nifty50/prices.csv'
    )

    parser.add_argument(
        '--symbols-file',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'symbols', 'nifty50.txt'),
        help='Path to text file listing ticker symbols (one per line). Default: data/symbols/nifty50.txt'
    )

    parser.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Force data refresh even if a cache exists'
    )

    parser.add_argument(
        '--head',
        type=int,
        default=5,
        help='Number of rows to print from the top of the price table. Default: 5'
    )

    parser.add_argument(
        '--show-returns',
        action='store_true',
        help='Also compute and print log-return diagnostics'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Nifty 50 Data Fetch Utility")
    print("=" * 80)

    print("\nFetching historical data...")
    fetcher = DataFetcher(symbols_file=args.symbols_file)

    try:
        if args.start_date and args.end_date:
            prices = fetcher.fetch_data(
                start_date=args.start_date,
                end_date=args.end_date,
                cache_path=args.cache_path,
                refresh_cache=args.refresh_cache
            )
        else:
            prices = fetcher.fetch_data(
                period=args.period,
                cache_path=args.cache_path,
                refresh_cache=args.refresh_cache
            )
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    print(f"\nRows: {prices.shape[0]}")
    print(f"Columns (symbols): {prices.shape[1]}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    print("\nPrice sample:")
    print(prices.head(args.head).to_string())

    if args.show_returns:
        returns = fetcher.get_returns(prices)
        print("\nLog-return sample:")
        print(returns.head(args.head).to_string())
        print("\nAverage daily log return (first 10 symbols):")
        print(returns.mean().sort_values(ascending=False).head(10).to_string())

    print("\n" + "=" * 80)
    print("Fetch completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
