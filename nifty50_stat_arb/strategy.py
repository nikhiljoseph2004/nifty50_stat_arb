"""
Pairs trading strategy module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from enum import Enum


class Signal(Enum):
    """Trading signals."""
    LONG = 1
    SHORT = -1
    CLOSE = 0
    NEUTRAL = 0


class PairsTrading:
    """Implements a pairs trading strategy based on z-scores."""
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 4.0,
        lookback_period: int = 60
    ):
        """
        Initialize the PairsTrading strategy.
        
        Args:
            entry_threshold: Z-score threshold for entry (absolute value)
            exit_threshold: Z-score threshold for exit (absolute value)
            stop_loss: Z-score threshold for stop loss (absolute value)
            lookback_period: Number of days for rolling statistics
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.lookback_period = lookback_period
    
    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate z-score of the spread.
        
        Args:
            spread: Series with spread values
            lookback: Lookback period for rolling mean/std
            
        Returns:
            Series with z-scores
        """
        if lookback is None:
            lookback = self.lookback_period
        
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def generate_signals(
        self,
        zscore: pd.Series
    ) -> pd.Series:
        """
        Generate trading signals based on z-scores.
        
        Args:
            zscore: Series with z-score values
            
        Returns:
            Series with trading signals
        """
        signals = pd.Series(index=zscore.index, data=0)
        
        position = 0  # Track current position
        
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            
            if pd.isna(z):
                signals.iloc[i] = position
                continue
            
            # Entry signals
            if position == 0:
                if z > self.entry_threshold:
                    # Spread is high - short the spread (short stock1, long stock2)
                    position = -1
                    signals.iloc[i] = position
                elif z < -self.entry_threshold:
                    # Spread is low - long the spread (long stock1, short stock2)
                    position = 1
                    signals.iloc[i] = position
                else:
                    signals.iloc[i] = 0
            
            # Exit signals
            elif position == 1:
                if z > -self.exit_threshold or z > self.stop_loss:
                    # Exit long position
                    position = 0
                    signals.iloc[i] = 0
                else:
                    signals.iloc[i] = position
            
            elif position == -1:
                if z < self.exit_threshold or z < -self.stop_loss:
                    # Exit short position
                    position = 0
                    signals.iloc[i] = 0
                else:
                    signals.iloc[i] = position
        
        return signals
    
    def calculate_positions(
        self,
        prices: pd.DataFrame,
        stock1: str,
        stock2: str,
        hedge_ratio: float
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Calculate positions and signals for a pair.
        
        Args:
            prices: DataFrame with price data
            stock1: First stock symbol
            stock2: Second stock symbol
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Tuple of (positions_df, spread, zscore)
        """
        # Calculate spread
        spread = prices[stock1] - hedge_ratio * prices[stock2]
        
        # Calculate z-score
        zscore = self.calculate_zscore(spread)
        
        # Generate signals
        signals = self.generate_signals(zscore)
        
        # Create positions DataFrame
        positions = pd.DataFrame(index=prices.index)
        positions['signal'] = signals
        positions[stock1] = signals  # Long/short stock1
        positions[stock2] = -signals * hedge_ratio  # Opposite position in stock2
        
        return positions, spread, zscore
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        positions: pd.DataFrame,
        stock1: str,
        stock2: str
    ) -> pd.Series:
        """
        Calculate strategy returns.
        
        Args:
            prices: DataFrame with price data
            positions: DataFrame with position data
            stock1: First stock symbol
            stock2: Second stock symbol
            
        Returns:
            Series with strategy returns
        """
        # Calculate price returns
        returns = prices[[stock1, stock2]].pct_change()
        
        # Calculate strategy returns (positions taken at previous close)
        strategy_returns = (
            positions[stock1].shift(1) * returns[stock1] +
            positions[stock2].shift(1) * returns[stock2]
        )
        
        return strategy_returns.fillna(0)
    
    def run_strategy(
        self,
        prices: pd.DataFrame,
        stock1: str,
        stock2: str,
        hedge_ratio: float
    ) -> Dict:
        """
        Run the complete strategy for a pair.
        
        Args:
            prices: DataFrame with price data
            stock1: First stock symbol
            stock2: Second stock symbol
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Dictionary with strategy results
        """
        # Calculate positions
        positions, spread, zscore = self.calculate_positions(
            prices, stock1, stock2, hedge_ratio
        )
        
        # Calculate returns
        returns = self.calculate_returns(prices, positions, stock1, stock2)
        
        results = {
            'positions': positions,
            'spread': spread,
            'zscore': zscore,
            'returns': returns,
            'cumulative_returns': (1 + returns).cumprod() - 1
        }
        
        return results
