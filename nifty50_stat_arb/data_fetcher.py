"""
Data fetcher module for downloading Nifty 50 stock data.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta


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
        'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
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
        period: str = "1y"
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
        print(f"Fetching data for {len(self.symbols)} stocks...")
        
        prices_dict = {}
        failed_symbols = []
        
        for symbol in self.symbols:
            try:
                if start_date and end_date:
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                else:
                    data = yf.download(symbol, period=period, progress=False)
                
                if not data.empty and 'Close' in data.columns:
                    prices_dict[symbol] = data['Close']
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Warning: Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        if not prices_dict:
            raise ValueError("No data fetched for any symbols")
        
        # Combine into a single DataFrame
        prices_df = pd.DataFrame(prices_dict)
        
        # Drop rows with too many missing values
        prices_df = prices_df.dropna(thresh=len(prices_df.columns) * 0.8)
        
        # Forward fill remaining NaN values
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Successfully fetched data: {prices_df.shape[0]} days, {prices_df.shape[1]} stocks")
        
        return prices_df
    
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
