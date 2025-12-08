"""
EODHD Data Client Module
Handles API communication and data normalization.
"""

import pandas as pd
import requests
import time
from typing import Optional


class EODHDDataClient:
    """
    Manages historical data retrieval from EODHD API with error handling and rate limits.
    Includes smart logic to select Adjusted Close for equities.
    """
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, api_key: str):
        """
        Initialize the client with API credentials.
        
        Args:
            api_key: EODHD API key
        """
        self.api_key = api_key
    
    def get_historical_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        exchange: str = 'US'
    ) -> pd.DataFrame:
        """
        Download EOD data and create a normalized 'Price' column (Adjusted vs Close).
        
        Args:
            ticker: Stock/ETF/Crypto symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchange: Exchange code (default: 'US')
            
        Returns:
            DataFrame with OHLCV data and normalized 'Price' column
        """
        url = f"{self.BASE_URL}/eod/{ticker}.{exchange}"
        
        params = {
            'api_token': self.api_key,
            'from': start_date,
            'to': end_date,
            'fmt': 'json'
        }
        
        attempt = 0
        max_retries = 3
        
        while attempt < max_retries:
            try:
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Rename columns to standard format
                    rename_map = {
                        'open': 'Open', 
                        'high': 'High', 
                        'low': 'Low',
                        'close': 'Close', 
                        'adjusted_close': 'Adj Close', 
                        'volume': 'Volume'
                    }
                    df = df.rename(columns=rename_map)
                    
                    # Cast to numeric
                    cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    for c in cols:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                    
                    # Smart Price Selection Logic
                    # Use Adjusted Close for stocks/ETFs, Close for crypto
                    if 'Adj Close' in df.columns and df['Adj Close'].notna().all():
                        df['Price'] = df['Adj Close']
                        df['price_type'] = 'Adjusted Close'
                    else:
                        df['Price'] = df['Close']
                        df['price_type'] = 'Close'
                    
                    return df.sort_index()
                else:
                    return pd.DataFrame()
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    attempt += 1
                    time.sleep(2 ** attempt)
                else:
                    raise ConnectionError(f"Failed to fetch data after {max_retries} attempts: {e}")
        
        return pd.DataFrame()
    
    def get_available_exchanges(self) -> list:
        """
        Return list of common exchange codes for user reference.
        """
        return [
            ('US', 'United States'),
            ('LSE', 'London Stock Exchange'),
            ('PA', 'Euronext Paris'),
            ('XETRA', 'German Exchange'),
            ('TO', 'Toronto Stock Exchange'),
            ('HK', 'Hong Kong'),
            ('CC', 'Cryptocurrency'),
            ('FOREX', 'Foreign Exchange'),
            ('INDX', 'Indices'),
        ]
    
    def validate_api_key(self) -> bool:
        """
        Test if the API key is valid.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            url = f"{self.BASE_URL}/exchange-symbol-list/US"
            params = {'api_token': self.api_key, 'fmt': 'json'}
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False
