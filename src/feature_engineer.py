"""
Feature Engineering Module
Calculates rolling median, distance percentages, and statistical bands.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Calculates Rolling Median and Percentile Ranks using the normalized 'Price' column.
    """
    
    # Band definitions (percentile ranges)
    BAND_DEFINITIONS = [
        (0.95, 1.00, "Extreme Overbought (>95%)"),
        (0.80, 0.95, "Overbought (80-95%)"),
        (0.55, 0.80, "High Neutral (55-80%)"),
        (0.45, 0.55, "Neutral (45-55%)"),
        (0.20, 0.45, "Low Neutral (20-45%)"),
        (0.05, 0.20, "Oversold (5-20%)"),
        (0.00, 0.05, "Extreme Oversold (<5%)"),
    ]
    
    def __init__(self, median_window: int = 252):
        """
        Initialize with rolling window size.
        
        Args:
            median_window: Rolling window for median calculation (default: 252 trading days = 1 year)
        """
        self.median_window = median_window
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features: median, distance, percentile rank, and bands.
        
        Args:
            df: DataFrame with 'Price' column
            
        Returns:
            DataFrame with additional calculated columns
        """
        df_feat = df.copy()
        
        # Ensure Price column exists
        if 'Price' not in df_feat.columns:
            if 'Close' in df_feat.columns:
                df_feat['Price'] = df_feat['Close']
            else:
                raise ValueError("DataFrame must contain 'Price' or 'Close' column")
        
        # 1. Rolling Median (252 days)
        df_feat['Median_252'] = df_feat['Price'].rolling(
            window=self.median_window, 
            min_periods=self.median_window
        ).median()
        
        # 2. Distance Percentage from Median
        df_feat['Distance_Pct'] = (df_feat['Price'] / df_feat['Median_252']) - 1
        
        # 3. Percentile Rank (Expanding) - How current distance compares to history
        min_periods_rank = self.median_window * 2
        df_feat['Percentile_Rank'] = df_feat['Distance_Pct'].expanding(
            min_periods=min_periods_rank
        ).rank(pct=True)
        
        # 4. Assign Statistical Bands
        df_feat['Band'] = df_feat['Percentile_Rank'].apply(self._assign_band)
        
        # Clean NaN rows (initial period without enough data)
        df_feat.dropna(subset=['Median_252'], inplace=True)
        
        return df_feat
    
    def _assign_band(self, percentile: float) -> str:
        """
        Map percentile rank to descriptive band name.
        
        Args:
            percentile: Value between 0 and 1
            
        Returns:
            Band name string
        """
        if pd.isna(percentile):
            return "N/A"
        
        for lower, upper, name in self.BAND_DEFINITIONS:
            if lower <= percentile < upper:
                return name
        
        # Edge case: exactly 1.0
        if percentile >= 0.95:
            return "Extreme Overbought (>95%)"
        
        return "N/A"
    
    def get_band_order(self) -> list:
        """
        Return bands in logical order (from oversold to overbought).
        
        Returns:
            List of band names in order
        """
        return [
            "Extreme Oversold (<5%)",
            "Oversold (5-20%)",
            "Low Neutral (20-45%)",
            "Neutral (45-55%)",
            "High Neutral (55-80%)",
            "Overbought (80-95%)",
            "Extreme Overbought (>95%)"
        ]
    
    def calculate_dynamic_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate expanding quantile bands for visualization.
        
        Args:
            df: DataFrame with 'Distance_Pct' and 'Median_252' columns
            
        Returns:
            DataFrame with band price levels
        """
        dist = df['Distance_Pct']
        median = df['Median_252']
        min_periods = self.median_window
        
        bands = pd.DataFrame(index=df.index)
        
        # Calculate quantiles of distance distribution
        bands['q05'] = dist.expanding(min_periods=min_periods).quantile(0.05)
        bands['q20'] = dist.expanding(min_periods=min_periods).quantile(0.20)
        bands['q80'] = dist.expanding(min_periods=min_periods).quantile(0.80)
        bands['q95'] = dist.expanding(min_periods=min_periods).quantile(0.95)
        
        # Project bands onto price: Band = Median * (1 + Quantile_Distance)
        bands['band_05'] = median * (1 + bands['q05'])
        bands['band_20'] = median * (1 + bands['q20'])
        bands['band_80'] = median * (1 + bands['q80'])
        bands['band_95'] = median * (1 + bands['q95'])
        
        return bands
    
    def get_current_status(self, df: pd.DataFrame) -> dict:
        """
        Get current market status summary.
        
        Args:
            df: Processed DataFrame with all features
            
        Returns:
            Dictionary with current status info
        """
        if df.empty:
            return {}
        
        last_row = df.iloc[-1]
        
        return {
            'date': last_row.name.strftime('%Y-%m-%d'),
            'price': last_row['Price'],
            'median': last_row['Median_252'],
            'distance_pct': last_row['Distance_Pct'] * 100,
            'percentile_rank': last_row['Percentile_Rank'] * 100,
            'band': last_row['Band'],
        }
