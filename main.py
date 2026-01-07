#!/usr/bin/env python3
"""
Main script for running the Nifty 50 Statistical Arbitrage Strategy.
"""

import argparse
import sys
from datetime import datetime, timedelta

from nifty50_stat_arb import DataFetcher, CointegrationAnalyzer, PairsTrading, Backtester


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Nifty 50 Statistical Arbitrage Strategy'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        help='Period for historical data (e.g., 1y, 2y, 5y). Default: 2y'
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
        '--top-pairs',
        type=int,
        default=5,
        help='Number of top cointegrated pairs to analyze. Default: 5'
    )
    
    parser.add_argument(
        '--significance',
        type=float,
        default=0.05,
        help='Significance level for cointegration test. Default: 0.05'
    )
    
    parser.add_argument(
        '--entry-threshold',
        type=float,
        default=2.0,
        help='Z-score threshold for entry. Default: 2.0'
    )
    
    parser.add_argument(
        '--exit-threshold',
        type=float,
        default=0.5,
        help='Z-score threshold for exit. Default: 0.5'
    )
    
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=4.0,
        help='Z-score threshold for stop loss. Default: 4.0'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=60,
        help='Lookback period for rolling statistics. Default: 60'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots for results'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Nifty 50 Statistical Arbitrage Strategy")
    print("=" * 80)
    
    # Step 1: Fetch data
    print("\n[Step 1/5] Fetching historical data...")
    fetcher = DataFetcher()
    
    try:
        if args.start_date and args.end_date:
            prices = fetcher.fetch_data(start_date=args.start_date, end_date=args.end_date)
        else:
            prices = fetcher.fetch_data(period=args.period)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Step 2: Find cointegrated pairs
    print(f"\n[Step 2/5] Analyzing cointegration...")
    analyzer = CointegrationAnalyzer(significance_level=args.significance)
    
    cointegrated_pairs = analyzer.test_cointegration(prices)
    
    if not cointegrated_pairs:
        print("No cointegrated pairs found!")
        sys.exit(0)
    
    # Get top pairs with analysis
    print(f"\n[Step 3/5] Analyzing top {args.top_pairs} pairs...")
    top_pairs_df = analyzer.get_top_pairs(prices, n_pairs=args.top_pairs)
    
    print("\nTop Cointegrated Pairs:")
    print("-" * 80)
    for idx, row in top_pairs_df.iterrows():
        print(f"{idx+1}. {row['stock1']} vs {row['stock2']}")
        print(f"   Cointegration p-value: {row['coint_pvalue']:.6f}")
        print(f"   Hedge ratio: {row['hedge_ratio']:.4f}")
        print(f"   Correlation: {row['correlation']:.4f}")
        print(f"   Spread mean: {row['spread_mean']:.4f}, std: {row['spread_std']:.4f}")
        print()
    
    # Step 4: Run strategy
    print(f"[Step 4/5] Running pairs trading strategy...")
    strategy = PairsTrading(
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss=args.stop_loss,
        lookback_period=args.lookback
    )
    
    pairs_results = {}
    
    for idx, row in top_pairs_df.iterrows():
        stock1 = row['stock1']
        stock2 = row['stock2']
        hedge_ratio = row['hedge_ratio']
        
        pair_name = f"{stock1}_vs_{stock2}"
        
        print(f"  Running strategy for {pair_name}...")
        
        results = strategy.run_strategy(prices, stock1, stock2, hedge_ratio)
        pairs_results[pair_name] = results
    
    # Step 5: Backtest
    print(f"\n[Step 5/5] Backtesting results...")
    backtester = Backtester(initial_capital=100000)
    
    metrics_df = backtester.backtest_multiple_pairs(pairs_results)
    
    print("\nBacktest Results Summary:")
    print("-" * 80)
    
    for pair_name in pairs_results.keys():
        backtester.print_summary(pair_name)
    
    # Print comparison table
    print("\nComparative Performance Metrics:")
    print("-" * 80)
    
    comparison_cols = [
        'pair_name', 'total_return', 'annualized_return',
        'sharpe_ratio', 'max_drawdown', 'win_rate'
    ]
    
    print(metrics_df[comparison_cols].to_string(index=False))
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        for pair_name in pairs_results.keys():
            save_path = f"{pair_name}_backtest.png"
            backtester.plot_results(pair_name, save_path=save_path)
    
    print("\n" + "=" * 80)
    print("Strategy execution completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
