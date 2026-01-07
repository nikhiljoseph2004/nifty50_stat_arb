"""
Backtesting module for pairs trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class Backtester:
    """Backtests pairs trading strategies."""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the Backtester.
        
        Args:
            initial_capital: Initial capital for backtesting
        """
        self.initial_capital = initial_capital
        self.results = {}
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        positions: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            returns: Series with strategy returns
            positions: Series with position signals (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Annualized return (assuming 252 trading days)
        n_days = len(returns)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and number of trades
        win_rate = None
        n_trades = None
        avg_win = None
        avg_loss = None
        
        if positions is not None:
            # Identify trades (position changes)
            position_changes = positions.diff().fillna(0)
            trades = returns[position_changes != 0]
            
            if len(trades) > 0:
                n_trades = len(trades)
                winning_trades = trades[trades > 0]
                losing_trades = trades[trades < 0]
                
                win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
                avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        return metrics
    
    def backtest_pair(
        self,
        strategy_results: Dict,
        pair_name: str = "Pair"
    ) -> Dict:
        """
        Backtest a single pair strategy.
        
        Args:
            strategy_results: Results from PairsTrading.run_strategy()
            pair_name: Name identifier for the pair
            
        Returns:
            Dictionary with backtest results
        """
        returns = strategy_results['returns']
        positions = strategy_results['positions']['signal']
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns, positions)
        
        # Calculate equity curve
        equity = self.initial_capital * (1 + strategy_results['cumulative_returns'])
        
        results = {
            'pair_name': pair_name,
            'metrics': metrics,
            'equity_curve': equity,
            'returns': returns,
            'positions': positions,
            'spread': strategy_results['spread'],
            'zscore': strategy_results['zscore']
        }
        
        self.results[pair_name] = results
        
        return results
    
    def backtest_multiple_pairs(
        self,
        pairs_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Backtest multiple pairs and aggregate results.
        
        Args:
            pairs_results: Dictionary mapping pair names to strategy results
            
        Returns:
            DataFrame with metrics for all pairs
        """
        all_metrics = []
        
        for pair_name, strategy_results in pairs_results.items():
            result = self.backtest_pair(strategy_results, pair_name)
            metrics = result['metrics'].copy()
            metrics['pair_name'] = pair_name
            all_metrics.append(metrics)
        
        return pd.DataFrame(all_metrics)
    
    def plot_results(
        self,
        pair_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot backtest results for a pair.
        
        Args:
            pair_name: Name of the pair to plot
            save_path: Path to save the plot (optional)
        """
        if pair_name not in self.results:
            raise ValueError(f"No results found for {pair_name}")
        
        results = self.results[pair_name]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot equity curve
        axes[0].plot(results['equity_curve'])
        axes[0].set_title(f'Equity Curve - {pair_name}')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True)
        
        # Plot spread and signals
        axes[1].plot(results['spread'], label='Spread', alpha=0.7)
        axes[1].set_title('Spread')
        axes[1].set_ylabel('Spread Value')
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot z-score and thresholds
        axes[2].plot(results['zscore'], label='Z-Score', color='blue', alpha=0.7)
        axes[2].axhline(y=2.0, color='r', linestyle='--', label='Entry Threshold')
        axes[2].axhline(y=-2.0, color='r', linestyle='--')
        axes[2].axhline(y=0.5, color='g', linestyle='--', label='Exit Threshold')
        axes[2].axhline(y=-0.5, color='g', linestyle='--')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_title('Z-Score with Entry/Exit Thresholds')
        axes[2].set_ylabel('Z-Score')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_summary(self, pair_name: str):
        """
        Print summary statistics for a pair.
        
        Args:
            pair_name: Name of the pair
        """
        if pair_name not in self.results:
            raise ValueError(f"No results found for {pair_name}")
        
        metrics = self.results[pair_name]['metrics']
        
        print(f"\n{'='*60}")
        print(f"Backtest Summary: {pair_name}")
        print(f"{'='*60}")
        print(f"Total Return:           {metrics['total_return']:.2%}")
        print(f"Annualized Return:      {metrics['annualized_return']:.2%}")
        print(f"Annualized Volatility:  {metrics['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown:       {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio:           {metrics['calmar_ratio']:.2f}")
        
        if metrics['n_trades'] is not None:
            print(f"\nTrading Statistics:")
            print(f"Number of Trades:       {metrics['n_trades']}")
            print(f"Win Rate:               {metrics['win_rate']:.2%}")
            if metrics['avg_win'] is not None:
                print(f"Average Win:            {metrics['avg_win']:.4f}")
            if metrics['avg_loss'] is not None:
                print(f"Average Loss:           {metrics['avg_loss']:.4f}")
        
        print(f"{'='*60}\n")
