"""
Example: Advanced usage with custom parameters and multiple pairs.

This example demonstrates:
1. Custom date ranges
2. Analyzing multiple top pairs
3. Comparing performance across pairs
4. Custom strategy parameters
"""

from nifty50_stat_arb import DataFetcher, CointegrationAnalyzer, PairsTrading, Backtester
import pandas as pd


def main():
    print("=" * 80)
    print("Advanced Example: Multiple Pairs Analysis")
    print("=" * 80)
    
    # Step 1: Fetch data with custom date range
    print("\n1. Fetching data for custom date range...")
    fetcher = DataFetcher()
    prices = fetcher.fetch_data(period='2y')
    
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Step 2: Find and analyze top pairs
    print("\n2. Finding top 5 cointegrated pairs...")
    analyzer = CointegrationAnalyzer(significance_level=0.05)
    analyzer.test_cointegration(prices)
    
    top_pairs_df = analyzer.get_top_pairs(prices, n_pairs=5)
    
    print("\nTop 5 Cointegrated Pairs:")
    print("-" * 80)
    print(top_pairs_df[['stock1', 'stock2', 'coint_pvalue', 'hedge_ratio', 'correlation']])
    
    # Step 3: Run strategy with different parameters
    print("\n3. Running strategies with different parameters...")
    
    strategies = {
        'Conservative': PairsTrading(entry_threshold=2.5, exit_threshold=0.5, stop_loss=5.0),
        'Standard': PairsTrading(entry_threshold=2.0, exit_threshold=0.5, stop_loss=4.0),
        'Aggressive': PairsTrading(entry_threshold=1.5, exit_threshold=0.3, stop_loss=3.0),
    }
    
    # Use the best pair
    best_pair = top_pairs_df.iloc[0]
    stock1 = best_pair['stock1']
    stock2 = best_pair['stock2']
    hedge_ratio = best_pair['hedge_ratio']
    
    print(f"\nTesting strategies on: {stock1} vs {stock2}")
    
    all_results = []
    
    for strategy_name, strategy in strategies.items():
        print(f"\nRunning {strategy_name} strategy...")
        
        results = strategy.run_strategy(prices, stock1, stock2, hedge_ratio)
        
        backtester = Backtester(initial_capital=100000)
        pair_name = f"{stock1}_vs_{stock2}_{strategy_name}"
        backtest_results = backtester.backtest_pair(results, pair_name)
        
        metrics = backtest_results['metrics']
        metrics['strategy'] = strategy_name
        all_results.append(metrics)
    
    # Step 4: Compare strategies
    print("\n4. Strategy Comparison:")
    print("-" * 80)
    
    comparison_df = pd.DataFrame(all_results)
    
    print("\nPerformance Comparison:")
    print(comparison_df[['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']])
    
    # Find best strategy
    best_strategy = comparison_df.loc[comparison_df['sharpe_ratio'].idxmax()]
    print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy['strategy']}")
    print(f"Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_strategy['total_return']:.2%}")
    
    # Step 5: Analyze multiple pairs with best strategy
    print("\n5. Running best strategy on all top pairs...")
    
    best_strategy_obj = strategies[best_strategy['strategy']]
    backtester_multi = Backtester(initial_capital=100000)
    
    pairs_results = {}
    
    for idx, row in top_pairs_df.iterrows():
        s1 = row['stock1']
        s2 = row['stock2']
        hr = row['hedge_ratio']
        
        pair_name = f"{s1}_vs_{s2}"
        
        results = best_strategy_obj.run_strategy(prices, s1, s2, hr)
        pairs_results[pair_name] = results
    
    metrics_df = backtester_multi.backtest_multiple_pairs(pairs_results)
    
    print("\nMultiple Pairs Performance:")
    print("-" * 80)
    print(metrics_df[['pair_name', 'total_return', 'sharpe_ratio', 'max_drawdown']])
    
    # Calculate portfolio performance (equal weight)
    print("\n6. Portfolio Performance (equal weight across all pairs):")
    print("-" * 80)
    
    portfolio_returns = pd.DataFrame({
        name: results['returns']
        for name, results in pairs_results.items()
    }).mean(axis=1)
    
    portfolio_metrics = backtester_multi.calculate_metrics(portfolio_returns)
    
    print(f"Total Return:      {portfolio_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio:      {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:      {portfolio_metrics['max_drawdown']:.2%}")
    
    print("\nAdvanced example completed successfully!")


if __name__ == "__main__":
    main()
