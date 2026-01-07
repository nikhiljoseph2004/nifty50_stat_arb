"""
Cointegration analysis module for identifying pairs.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class CointegrationAnalyzer:
    """Analyzes stock pairs for cointegration relationships."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the CointegrationAnalyzer.
        
        Args:
            significance_level: P-value threshold for cointegration test
        """
        self.significance_level = significance_level
        self.cointegrated_pairs = []
        self.pair_stats = {}
    
    def test_cointegration(
        self,
        prices: pd.DataFrame,
        method: str = 'engle-granger'
    ) -> List[Tuple[str, str, float]]:
        """
        Test all possible pairs for cointegration.
        
        Args:
            prices: DataFrame with price data for multiple stocks
            method: Cointegration test method ('engle-granger')
            
        Returns:
            List of tuples (stock1, stock2, p_value) for cointegrated pairs
        """
        stocks = prices.columns.tolist()
        n_stocks = len(stocks)
        
        print(f"Testing cointegration for {n_stocks * (n_stocks - 1) // 2} pairs...")
        
        cointegrated = []
        
        # Test all possible pairs
        for stock1, stock2 in combinations(stocks, 2):
            try:
                # Perform Engle-Granger cointegration test
                score, pvalue, _ = coint(prices[stock1], prices[stock2])
                
                if pvalue < self.significance_level:
                    cointegrated.append((stock1, stock2, pvalue))
                    
            except Exception as e:
                continue
        
        # Sort by p-value (most significant first)
        cointegrated.sort(key=lambda x: x[2])
        
        self.cointegrated_pairs = cointegrated
        
        print(f"Found {len(cointegrated)} cointegrated pairs at {self.significance_level} significance level")
        
        return cointegrated
    
    def calculate_hedge_ratio(
        self,
        prices: pd.DataFrame,
        stock1: str,
        stock2: str
    ) -> float:
        """
        Calculate the hedge ratio for a pair using OLS regression.
        
        Args:
            prices: DataFrame with price data
            stock1: First stock symbol
            stock2: Second stock symbol
            
        Returns:
            Hedge ratio (beta coefficient)
        """
        from scipy import stats
        
        y = prices[stock1].values
        x = prices[stock2].values
        
        # Linear regression: y = alpha + beta * x
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return slope
    
    def calculate_spread(
        self,
        prices: pd.DataFrame,
        stock1: str,
        stock2: str,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate the spread between two stocks.
        
        Args:
            prices: DataFrame with price data
            stock1: First stock symbol
            stock2: Second stock symbol
            hedge_ratio: Hedge ratio. If None, calculates it.
            
        Returns:
            Series with spread values
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(prices, stock1, stock2)
        
        spread = prices[stock1] - hedge_ratio * prices[stock2]
        
        return spread
    
    def test_spread_stationarity(
        self,
        spread: pd.Series
    ) -> Tuple[bool, float, Dict]:
        """
        Test if the spread is stationary using Augmented Dickey-Fuller test.
        
        Args:
            spread: Series with spread values
            
        Returns:
            Tuple of (is_stationary, p_value, test_stats)
        """
        result = adfuller(spread.dropna(), autolag='AIC')
        
        adf_stat = result[0]
        pvalue = result[1]
        critical_values = result[4]
        
        is_stationary = pvalue < self.significance_level
        
        test_stats = {
            'adf_statistic': adf_stat,
            'p_value': pvalue,
            'critical_values': critical_values,
            'n_lags': result[2]
        }
        
        return is_stationary, pvalue, test_stats
    
    def analyze_pair(
        self,
        prices: pd.DataFrame,
        stock1: str,
        stock2: str
    ) -> Dict:
        """
        Perform comprehensive analysis on a pair.
        
        Args:
            prices: DataFrame with price data
            stock1: First stock symbol
            stock2: Second stock symbol
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(prices, stock1, stock2)
        
        # Calculate spread
        spread = self.calculate_spread(prices, stock1, stock2, hedge_ratio)
        
        # Test spread stationarity
        is_stationary, pvalue, adf_stats = self.test_spread_stationarity(spread)
        
        # Calculate spread statistics
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        analysis = {
            'stock1': stock1,
            'stock2': stock2,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'is_stationary': is_stationary,
            'adf_pvalue': pvalue,
            'adf_statistic': adf_stats['adf_statistic'],
            'correlation': prices[[stock1, stock2]].corr().iloc[0, 1]
        }
        
        return analysis
    
    def get_top_pairs(
        self,
        prices: pd.DataFrame,
        n_pairs: int = 10
    ) -> pd.DataFrame:
        """
        Get the top N cointegrated pairs with full analysis.
        
        Args:
            prices: DataFrame with price data
            n_pairs: Number of top pairs to return
            
        Returns:
            DataFrame with pair analysis results
        """
        if not self.cointegrated_pairs:
            self.test_cointegration(prices)
        
        top_pairs = self.cointegrated_pairs[:n_pairs]
        
        results = []
        for stock1, stock2, coint_pvalue in top_pairs:
            analysis = self.analyze_pair(prices, stock1, stock2)
            analysis['coint_pvalue'] = coint_pvalue
            results.append(analysis)
        
        return pd.DataFrame(results)


from typing import Optional
