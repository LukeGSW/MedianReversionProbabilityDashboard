"""
Signal Processing Module
Handles turning point detection and reversal score calculation.
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Optional


class SignalProcessor:
    """
    Manages identification of turning points (ex-post) and 
    calculation of real-time reversal score.
    """
    
    def __init__(self, turning_point_order: int = 20):
        """
        Initialize signal processor.
        
        Args:
            turning_point_order: Days of look-ahead/look-back to confirm local max/min.
                                 Value of 20 identifies medium-term trends.
        """
        self.order = turning_point_order
    
    def detect_ex_post_turning_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify historical local maxima and minima.
        
        NOTE: This function uses future data (look-ahead). It's meant ONLY for 
        educational charts, NOT for generating trading signals in backtests.
        
        Args:
            df: DataFrame with 'Price' and 'Distance_Pct' columns
            
        Returns:
            DataFrame with turning points (Date, Price, Type, Distance_Pct)
        """
        df_sig = df.copy()
        price_arr = df_sig['Price'].values
        
        # Identify indices of local maxima and minima
        # argrelextrema compares current point with its neighbors (order)
        highs_idx = argrelextrema(price_arr, np.greater, order=self.order)[0]
        lows_idx = argrelextrema(price_arr, np.less, order=self.order)[0]
        
        turning_points = []
        
        # Extract dates and prices of highs
        for idx in highs_idx:
            turning_points.append({
                'Date': df_sig.index[idx],
                'Price': df_sig.iloc[idx]['Price'],
                'Type': 'Top',
                'Distance_Pct': df_sig.iloc[idx]['Distance_Pct']
            })
        
        # Extract dates and prices of lows
        for idx in lows_idx:
            turning_points.append({
                'Date': df_sig.index[idx],
                'Price': df_sig.iloc[idx]['Price'],
                'Type': 'Bottom',
                'Distance_Pct': df_sig.iloc[idx]['Distance_Pct']
            })
        
        if turning_points:
            result = pd.DataFrame(turning_points).sort_values(by='Date')
            result.set_index('Date', inplace=True)
            return result
        
        return pd.DataFrame()
    
    def calculate_real_time_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Reversal Score based on percentiles.
        This is a real-time indicator (no look-ahead bias).
        
        Reversal Score (0-100):
        - > 80: High probability of bearish reversal (price extended above median)
        - < 20: High probability of bullish bounce (price depressed below median)
        
        Args:
            df: DataFrame with 'Percentile_Rank' column
            
        Returns:
            DataFrame with 'Reversal_Score' column added
        """
        df_sig = df.copy()
        
        # Map Percentile Rank (0-1) to 0-100 scale
        df_sig['Reversal_Score'] = df_sig['Percentile_Rank'] * 100
        
        return df_sig
    
    def get_signal_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary of current signal status.
        
        Args:
            df: DataFrame with reversal score
            
        Returns:
            Dictionary with signal interpretation
        """
        if df.empty or 'Reversal_Score' not in df.columns:
            return {}
        
        last_score = df['Reversal_Score'].iloc[-1]
        
        if last_score >= 95:
            signal = "EXTREME OVERBOUGHT"
            interpretation = "Very high probability of correction. Consider reducing exposure."
            color = "red"
        elif last_score >= 80:
            signal = "OVERBOUGHT"
            interpretation = "Price extended above median. Reversal risk elevated."
            color = "orange"
        elif last_score <= 5:
            signal = "EXTREME OVERSOLD"
            interpretation = "Very high probability of bounce. Consider accumulation."
            color = "darkgreen"
        elif last_score <= 20:
            signal = "OVERSOLD"
            interpretation = "Price depressed below median. Mean reversion opportunity."
            color = "green"
        else:
            signal = "NEUTRAL"
            interpretation = "Price in fair value zone. Follow prevailing trend."
            color = "gray"
        
        return {
            'score': round(last_score, 1),
            'signal': signal,
            'interpretation': interpretation,
            'color': color
        }
    
    def count_turning_points_by_band(
        self, 
        df_scored: pd.DataFrame, 
        df_turning_points: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Count how many turning points occurred in each band.
        Useful for validating band effectiveness.
        
        Args:
            df_scored: Main DataFrame with 'Band' column
            df_turning_points: Turning points DataFrame
            
        Returns:
            DataFrame with counts per band and type
        """
        if df_turning_points.empty:
            return pd.DataFrame()
        
        # Match turning points to their bands
        tp_bands = []
        
        for date, row in df_turning_points.iterrows():
            if date in df_scored.index:
                band = df_scored.loc[date, 'Band']
                tp_bands.append({
                    'Band': band,
                    'Type': row['Type']
                })
        
        if tp_bands:
            tp_df = pd.DataFrame(tp_bands)
            return tp_df.groupby(['Band', 'Type']).size().unstack(fill_value=0)
        
        return pd.DataFrame()
