"""
Example: Basic usage of the Nifty 50 Statistical Arbitrage Strategy.

This example demonstrates how to:
1. Fetch historical data
2. Find cointegrated pairs
3. Run a pairs trading strategy
4. Backtest the results
"""

from nifty50_stat_arb import DataFetcher, CointegrationAnalyzer, PairsTrading, Backtester


def main():
    print("=" * 80)
    print("Basic Example: Nifty 50 Statistical Arbitrage")
    print("=" * 80)
    
    # Step 1: Fetch data
    print("\n1. Fetching data...")
    fetcher = DataFetcher()
    prices = fetcher.fetch_data(period='1y')
    
    print(f"Fetched data for {prices.shape[1]} stocks over {prices.shape[0]} days")
    
    # Step 2: Find cointegrated pairs
    print("\n2. Finding cointegrated pairs...")
    analyzer = CointegrationAnalyzer(significance_level=0.05)
    cointegrated_pairs = analyzer.test_cointegration(prices)
    
    if not cointegrated_pairs:
        print("No cointegrated pairs found!")
        return
    
    # Get the best pair
    best_pair = cointegrated_pairs[0]
    stock1, stock2, pvalue = best_pair
    
    print(f"\nBest pair: {stock1} vs {stock2}")
    print(f"P-value: {pvalue:.6f}")
    
    # Analyze the pair
    analysis = analyzer.analyze_pair(prices, stock1, stock2)
    
    print(f"Hedge ratio: {analysis['hedge_ratio']:.4f}")
    print(f"Correlation: {analysis['correlation']:.4f}")
    print(f"Spread is stationary: {analysis['is_stationary']}")
    
    # Step 3: Run strategy
    print("\n3. Running trading strategy...")
    strategy = PairsTrading(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=4.0,
        lookback_period=60
    )
    
    results = strategy.run_strategy(
        prices,
        stock1,
        stock2,
        analysis['hedge_ratio']
    )
    
    # Step 4: Backtest
    print("\n4. Backtesting...")
    backtester = Backtester(initial_capital=100000)
    
    pair_name = f"{stock1}_vs_{stock2}"
    backtest_results = backtester.backtest_pair(results, pair_name)
    
    # Print results
    backtester.print_summary(pair_name)
    
    # Generate plot
    print("\n5. Generating plot...")
    backtester.plot_results(pair_name, save_path=f"{pair_name}_example.png")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
