"""
Nifty 50 data tools.

Currently exposes the DataFetcher utility for downloading and caching
Nifty 50 prices from Yahoo Finance.
"""

__version__ = "0.1.0"
__author__ = "Nikhil Joseph"

from .data_fetcher import DataFetcher
from .pca import compute_pca, load_returns, print_pca_summary

__all__ = [
    "DataFetcher",
    "load_returns",
    "compute_pca",
    "print_pca_summary",
]
