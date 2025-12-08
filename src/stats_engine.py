"""
Statistical Engine Module
Performs backtesting by calculating forward returns conditioned on band membership.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class StatsEngine:
    """
    Executes statistical backtesting by calculating future returns
    conditioned on the band at entry.
    """
    
    # Default time horizons (trading days)
    DEFAULT_HORIZONS = [10, 21, 63]  # ~2 weeks, 1 month, 3 months
    
    # Logical band order for display
    BAND_ORDER = [
        "Extreme Oversold (<5%)",
        "Oversold (5-20%)",
        "Low Neutral (20-45%)",
        "Neutral (45-55%)",
        "High Neutral (55-80%)",
        "Overbought (80-95%)",
        "Extreme Overbought (>95%)"
    ]
    
    def __init__(self, horizons: Optional[List[int]] = None):
        """
        Initialize with time horizons for return calculation.
        
        Args:
            horizons: List of forward-looking periods in trading days.
                     10=~2 weeks, 21=1 month, 63=3 months
        """
        self.horizons = horizons or self.DEFAULT_HORIZONS
    
    def calculate_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward returns for each horizon.
        
        Args:
            df: DataFrame with 'Price' column
            
        Returns:
            DataFrame with forward return columns added
        """
        df_stats = df.copy()
        
        for h in self.horizons:
            # Shift(-h) gets the price 'h' days in the future
            # Return = (Future Price / Current Price) - 1
            df_stats[f'Fwd_Ret_{h}d'] = df_stats['Price'].shift(-h) / df_stats['Price'] - 1
        
        return df_stats
    
    def generate_performance_matrix(self, df_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results by Band.
        
        Args:
            df_stats: DataFrame with forward returns and 'Band' column
            
        Returns:
            Performance matrix DataFrame indexed by Band
        """
        ret_cols = [f'Fwd_Ret_{h}d' for h in self.horizons]
        results = []
        
        for band in self.BAND_ORDER:
            subset = df_stats[df_stats['Band'] == band]
            row = {'Band': band, 'Count': len(subset)}
            
            for col in ret_cols:
                if len(subset) > 0:
                    # Mean Return %
                    mean_ret = subset[col].mean() * 100
                    row[f'Avg_Ret_{col.split("_")[-1]}'] = round(mean_ret, 2) if pd.notna(mean_ret) else None
                    
                    # Win Rate % (Positive Returns)
                    valid_returns = subset[col].dropna()
                    if len(valid_returns) > 0:
                        win_rate = (valid_returns > 0).mean() * 100
                        row[f'Win_Rate_{col.split("_")[-1]}'] = round(win_rate, 1)
                    else:
                        row[f'Win_Rate_{col.split("_")[-1]}'] = None
                else:
                    row[f'Avg_Ret_{col.split("_")[-1]}'] = None
                    row[f'Win_Rate_{col.split("_")[-1]}'] = None
            
            results.append(row)
        
        return pd.DataFrame(results).set_index('Band')
    
    def get_band_statistics(
        self, 
        df_stats: pd.DataFrame, 
        band: str
    ) -> dict:
        """
        Get detailed statistics for a specific band.
        
        Args:
            df_stats: DataFrame with forward returns
            band: Band name to analyze
            
        Returns:
            Dictionary with statistics for the band
        """
        subset = df_stats[df_stats['Band'] == band]
        
        if len(subset) == 0:
            return {'error': 'No data for this band'}
        
        stats = {
            'count': len(subset),
            'avg_distance_pct': round(subset['Distance_Pct'].mean() * 100, 2),
        }
        
        for h in self.horizons:
            col = f'Fwd_Ret_{h}d'
            valid = subset[col].dropna()
            
            if len(valid) > 0:
                stats[f'avg_return_{h}d'] = round(valid.mean() * 100, 2)
                stats[f'win_rate_{h}d'] = round((valid > 0).mean() * 100, 1)
                stats[f'max_return_{h}d'] = round(valid.max() * 100, 2)
                stats[f'min_return_{h}d'] = round(valid.min() * 100, 2)
                stats[f'std_return_{h}d'] = round(valid.std() * 100, 2)
        
        return stats
    
    def calculate_sharpe_by_band(
        self, 
        df_stats: pd.DataFrame, 
        horizon: int = 21,
        risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Calculate Sharpe-like ratio for each band.
        
        Args:
            df_stats: DataFrame with forward returns
            horizon: Time horizon to use
            risk_free_rate: Annual risk-free rate (default 0)
            
        Returns:
            DataFrame with Sharpe ratios by band
        """
        col = f'Fwd_Ret_{horizon}d'
        results = []
        
        # Annualization factor
        periods_per_year = 252 / horizon
        
        for band in self.BAND_ORDER:
            subset = df_stats[df_stats['Band'] == band]
            valid = subset[col].dropna()
            
            if len(valid) > 1:
                mean_ret = valid.mean() * periods_per_year
                std_ret = valid.std() * np.sqrt(periods_per_year)
                
                if std_ret > 0:
                    sharpe = (mean_ret - risk_free_rate) / std_ret
                else:
                    sharpe = None
            else:
                sharpe = None
            
            results.append({
                'Band': band,
                'Sharpe_Ratio': round(sharpe, 2) if sharpe else None
            })
        
        return pd.DataFrame(results).set_index('Band')
    
    def validate_mean_reversion(self, performance_matrix: pd.DataFrame) -> dict:
        """
        Check if mean reversion hypothesis holds for this asset.
        
        Args:
            performance_matrix: Performance matrix DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': False,
            'oversold_edge': False,
            'overbought_edge': False,
            'details': []
        }
        
        try:
            # Check oversold edge (should have positive returns)
            extreme_oversold = performance_matrix.loc["Extreme Oversold (<5%)"]
            oversold = performance_matrix.loc["Oversold (5-20%)"]
            
            oversold_avg_21d = (extreme_oversold['Avg_Ret_21d'] + oversold['Avg_Ret_21d']) / 2
            oversold_wr_21d = (extreme_oversold['Win_Rate_21d'] + oversold['Win_Rate_21d']) / 2
            
            if oversold_avg_21d > 0 and oversold_wr_21d > 50:
                validation['oversold_edge'] = True
                validation['details'].append(
                    f"Oversold zones show positive edge: Avg Return {oversold_avg_21d:.2f}%, Win Rate {oversold_wr_21d:.1f}%"
                )
            
            # Check overbought edge (should have lower/negative returns)
            extreme_overbought = performance_matrix.loc["Extreme Overbought (>95%)"]
            overbought = performance_matrix.loc["Overbought (80-95%)"]
            
            overbought_avg_21d = (extreme_overbought['Avg_Ret_21d'] + overbought['Avg_Ret_21d']) / 2
            
            # Compare to neutral
            neutral = performance_matrix.loc["Neutral (45-55%)"]
            
            if overbought_avg_21d < neutral['Avg_Ret_21d']:
                validation['overbought_edge'] = True
                validation['details'].append(
                    f"Overbought zones show compression: Avg Return {overbought_avg_21d:.2f}% vs Neutral {neutral['Avg_Ret_21d']:.2f}%"
                )
            
            # Overall validation
            validation['is_valid'] = validation['oversold_edge'] and validation['overbought_edge']
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
