"""
Nifty 50 Statistical Arbitrage Strategy

A pairs trading strategy that identifies cointegrated pairs
in the Nifty 50 index and trades based on mean reversion.
"""

__version__ = "0.1.0"
__author__ = "Nikhil Joseph"

from .data_fetcher import DataFetcher
from .cointegration import CointegrationAnalyzer
from .strategy import PairsTrading
from .backtest import Backtester

__all__ = [
    "DataFetcher",
    "CointegrationAnalyzer",
    "PairsTrading",
    "Backtester",
]
