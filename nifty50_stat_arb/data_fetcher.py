"""
Data fetcher module for downloading Nifty 50 stock data.
"""

import os
from typing import List, Optional

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_PATH = os.path.join("data", "nifty50_prices.csv")


class DataFetcher:
    """Fetches historical price data for Nifty 50 stocks."""
    
    # Nifty 50 stocks with NSE symbols (Yahoo Finance format)
    NIFTY50_SYMBOLS = [
        'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS',
        'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
        'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
        'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
        'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
        'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS',
        'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
        'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
        'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS',
        'SUNPHARMA.NS', 'TATACONSUM.NS', 'TMPV.NS', 'TATASTEEL.NS',
        'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
        'UPL.NS', 'WIPRO.NS'
    ]
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize the DataFetcher.
        
        Args:
            symbols: List of stock symbols. If None, uses all Nifty 50 stocks.
        """
        self.symbols = symbols if symbols is not None else self.NIFTY50_SYMBOLS
    
    def fetch_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        cache_path: Optional[str] = DEFAULT_CACHE_PATH,
        refresh_cache: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical price data for the stocks.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch if start_date not provided (e.g., '1y', '2y', '5y')
            
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

