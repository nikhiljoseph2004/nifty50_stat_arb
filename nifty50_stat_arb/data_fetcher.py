"""
Data fetcher module for downloading stock data for arbitrary symbol lists.
"""

import os
from typing import List, Optional

import pandas as pd
import yfinance as yf


class DataFetcher:
    """Fetches historical price data for an arbitrary list of stock symbols."""

    @staticmethod
    def load_symbols_from_file(path: str) -> List[str]:
        """Load ticker symbols from a plain-text file (one per line).

        Lines beginning with ``#`` and blank lines are ignored, so the file
        can contain comments describing the index.
        """
        with open(path) as fh:
            symbols = [
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            ]
        if not symbols:
            raise ValueError(f"No symbols found in {path}")
        return symbols

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        symbols_file: Optional[str] = None,
    ):
        """
        Initialize the DataFetcher.

        Exactly one of *symbols* or *symbols_file* must be provided.

        Args:
            symbols: Explicit list of Yahoo Finance ticker symbols.
            symbols_file: Path to a text file with one ticker per line.
        """
        if symbols_file is not None:
            self.symbols = self.load_symbols_from_file(symbols_file)
        elif symbols is not None:
            self.symbols = symbols
        else:
            raise ValueError("Provide either 'symbols' or 'symbols_file'.")
    
    def fetch_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        cache_path: Optional[str] = None,
        returns_cache_path: Optional[str] = None,
        refresh_cache: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical price data for the stocks.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if start_date not provided (e.g., '1y', '2y', '5y')
            returns_cache_path: Path to CSV file where log returns will be saved
            
        Returns:
            DataFrame with closing prices for all symbols
        """
        date_filter_start = pd.to_datetime(start_date) if start_date else None
        date_filter_end = pd.to_datetime(end_date) if end_date else None

        cache_file = None
        cached_df = None

        if cache_path:
            cache_file = os.path.abspath(cache_path)
            cache_exists = os.path.exists(cache_file)
            if cache_exists and not refresh_cache:
                try:
                    cached_df = self._load_cache(cache_file)
                except Exception as exc:
                    print(f"Failed to read cache {cache_file}: {exc}")
                    cached_df = None

            if cached_df is not None:
                if ((date_filter_start and date_filter_start < cached_df.index[0]) or
                        (date_filter_end and date_filter_end > cached_df.index[-1])):
                    print("Cached data range does not cover requested dates; refreshing cache.")
                    cached_df = None
                else:
                    filtered = self._filter_by_dates(cached_df, date_filter_start, date_filter_end)
                    if filtered.empty:
                        print("Cached data does not contain any rows for the requested range; refetching...")
                        cached_df = None
                    else:
                        self._save_returns(filtered, returns_cache_path)
                        print(f"Loaded data from cache: {cache_file}")
                        return filtered

        print(f"Fetching data for {len(self.symbols)} stocks...")

        series_list = []
        failed_symbols = []

        for symbol in self.symbols:
            try:
                if start_date and end_date:
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                else:
                    data = yf.download(symbol, period=period, progress=False)

                if not data.empty and 'Close' in data.columns:
                    series = data['Close'].copy()
                    series.name = symbol
                    series_list.append(series)
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Warning: Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        if not series_list:
            raise ValueError("No data fetched for any symbols")
        
        # Combine into a single DataFrame
        prices_df = pd.concat(series_list, axis=1)
        
        # Drop rows with too many missing values
        prices_df = prices_df.dropna(thresh=len(prices_df.columns) * 0.8)
        
        # Forward fill remaining NaN values
        prices_df = prices_df.ffill().bfill()
        
        print(f"Successfully fetched data: {prices_df.shape[0]} days, {prices_df.shape[1]} stocks")

        if cache_file:
            self._save_cache(prices_df, cache_file)

        filtered = self._filter_by_dates(prices_df, date_filter_start, date_filter_end)
        if filtered.empty:
            raise ValueError("No data available for the requested date range")

        self._save_returns(filtered, returns_cache_path)
        return filtered
    
    def get_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from price data.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            DataFrame with log returns
        """
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def _filter_by_dates(
        prices: pd.DataFrame,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp]
    ) -> pd.DataFrame:
        """Return data that falls within the requested date range."""
        filtered = prices

        if start_date is not None:
            filtered = filtered.loc[filtered.index >= start_date]
        if end_date is not None:
            filtered = filtered.loc[filtered.index <= end_date]

        if filtered.empty:
            return pd.DataFrame()

        return filtered

    def _save_cache(self, prices: pd.DataFrame, cache_path: str):
        """Persist the downloaded prices to disk."""
        directory = os.path.dirname(cache_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        prices.to_csv(cache_path)
        print(f"Saved data cache to {cache_path}")

    @staticmethod
    def _load_cache(cache_path: str) -> pd.DataFrame:
        """Load cached prices from disk."""
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.sort_index(inplace=True)
        return cached

    def _save_returns(self, prices: pd.DataFrame, returns_cache_path: Optional[str]):
        """Compute and persist log returns if a returns path is provided."""
        if not returns_cache_path:
            return

        returns = self.get_returns(prices)
        directory = os.path.dirname(returns_cache_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        returns.to_csv(returns_cache_path)
        print(f"Saved returns cache to {returns_cache_path}")

