"""
Visualizer Module
Generates interactive Plotly charts for Mean Reversion analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple


class Visualizer:
    """
    Generates interactive Plotly charts for Mean Reversion analysis.
    Dynamically calculates probability bands (quantiles) for the main chart.
    """
    
    # Color scheme
    COLORS = {
        'price': '#2c3e50',
        'median': 'black',
        'overbought': 'rgba(231, 76, 60, 0.5)',
        'oversold': 'rgba(39, 174, 96, 0.5)',
        'extreme_overbought_fill': 'rgba(231, 76, 60, 0.2)',
        'extreme_oversold_fill': 'rgba(39, 174, 96, 0.2)',
        'neutral': 'rgba(52, 152, 219, 0.3)',
        'tops': 'red',
        'bottoms': 'green',
        'score_overbought': 'red',
        'score_oversold': 'green',
        'score_neutral': 'gray',
    }
    
    def __init__(self, ticker: str):
        """
        Initialize visualizer with ticker name.
        
        Args:
            ticker: Stock/ETF symbol for titles
        """
        self.ticker = ticker
    
    def plot_analysis(
        self, 
        df: pd.DataFrame, 
        turning_points: Optional[pd.DataFrame] = None,
        show_distributions: bool = True
    ) -> go.Figure:
        """
        Create a complete dashboard with multiple panels.
        
        Panels:
        1. Main Chart: Price, Median, and Probability Bands (5%, 20%, 80%, 95%)
        2. Oscillator: Reversal Score
        3. Distribution: Fat Tails histograms
        
        Args:
            df: DataFrame with Price, Median_252, Distance_Pct, Reversal_Score
            turning_points: Optional DataFrame with historical tops/bottoms
            show_distributions: Whether to show distribution panels
            
        Returns:
            Plotly Figure object
        """
        # Pre-calculate statistical bands
        bands = self._calculate_bands(df)
        
        # Create figure with subplots
        if show_distributions:
            fig = make_subplots(
                rows=3, cols=2,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                specs=[
                    [{"colspan": 2}, None],
                    [{"colspan": 2}, None],
                    [{"type": "xy"}, {"type": "xy"}]
                ],
                subplot_titles=(
                    f"{self.ticker} Price Action & Probability Bands",
                    "Reversal Score (Real-Time)",
                    "Price Distance Distribution",
                    "Reversal Score Distribution"
                )
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.75, 0.25],
                subplot_titles=(
                    f"{self.ticker} Price Action & Probability Bands",
                    "Reversal Score (Real-Time)"
                )
            )
        
        # Add traces to main chart
        self._add_main_chart(fig, df, bands, turning_points)
        
        # Add reversal score oscillator
        self._add_oscillator(fig, df, row=2)
        
        # Add distributions if requested
        if show_distributions:
            self._add_distributions(fig, df)
        
        # Update layout
        self._update_layout(fig, df, show_distributions)
        
        return fig
    
    def _calculate_bands(self, df: pd.DataFrame) -> dict:
        """
        Calculate expanding quantile bands.
        
        Returns:
            Dictionary with band series
        """
        dist = df['Distance_Pct']
        median = df['Median_252']
        min_periods = 252
        
        # Calculate quantiles
        q05 = dist.expanding(min_periods=min_periods).quantile(0.05)
        q20 = dist.expanding(min_periods=min_periods).quantile(0.20)
        q80 = dist.expanding(min_periods=min_periods).quantile(0.80)
        q95 = dist.expanding(min_periods=min_periods).quantile(0.95)
        
        # Project onto price
        return {
            'band_05': median * (1 + q05),
            'band_20': median * (1 + q20),
            'band_80': median * (1 + q80),
            'band_95': median * (1 + q95),
        }
    
    def _add_main_chart(
        self, 
        fig: go.Figure, 
        df: pd.DataFrame, 
        bands: dict,
        turning_points: Optional[pd.DataFrame]
    ):
        """Add main price chart with bands."""
        
        # Extreme Overbought Zone fill (>95%)
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=bands['band_95'],
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Price'] * 1.5,  # Upper bound for fill
            fill='tonexty',
            fillcolor=self.COLORS['extreme_overbought_fill'],
            line=dict(width=0),
            name='Extreme Overbought Zone',
            hoverinfo='skip'
        ), row=1, col=1)
        
        # 80% Resistance Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=bands['band_80'],
            line=dict(color=self.COLORS['overbought'], width=1, dash='dot'),
            name='80% Resistance'
        ), row=1, col=1)
        
        # 20% Support Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=bands['band_20'],
            line=dict(color=self.COLORS['oversold'], width=1, dash='dot'),
            name='20% Support'
        ), row=1, col=1)
        
        # 5% Extreme Support Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=bands['band_05'],
            line=dict(color='rgba(39, 174, 96, 0.8)', width=1),
            name='5% Extreme Support'
        ), row=1, col=1)
        
        # Median (Gravity)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Median_252'],
            line=dict(color=self.COLORS['median'], width=2),
            name='Median (252d)'
        ), row=1, col=1)
        
        # Price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Price'],
            line=dict(color=self.COLORS['price'], width=1.5),
            name='Price'
        ), row=1, col=1)
        
        # Turning Points
        if turning_points is not None and not turning_points.empty:
            tops = turning_points[turning_points['Type'] == 'Top']
            bottoms = turning_points[turning_points['Type'] == 'Bottom']
            
            if not tops.empty:
                fig.add_trace(go.Scatter(
                    mode='markers',
                    x=tops.index,
                    y=tops['Price'],
                    marker=dict(symbol='triangle-down', size=10, color=self.COLORS['tops']),
                    name='Historical Tops'
                ), row=1, col=1)
            
            if not bottoms.empty:
                fig.add_trace(go.Scatter(
                    mode='markers',
                    x=bottoms.index,
                    y=bottoms['Price'],
                    marker=dict(symbol='triangle-up', size=10, color=self.COLORS['bottoms']),
                    name='Historical Bottoms'
                ), row=1, col=1)
    
    def _add_oscillator(self, fig: go.Figure, df: pd.DataFrame, row: int = 2):
        """Add reversal score oscillator panel."""
        
        score = df['Reversal_Score']
        colors = [
            self.COLORS['score_overbought'] if v >= 80 
            else self.COLORS['score_oversold'] if v <= 20 
            else self.COLORS['score_neutral'] 
            for v in score
        ]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=score,
            marker_color=colors,
            name='Reversal Score'
        ), row=row, col=1)
        
        # Threshold lines
        fig.add_hline(
            y=80, 
            line_dash="dash", 
            line_color="red", 
            row=row, col=1,
            annotation_text="Overbought Risk"
        )
        fig.add_hline(
            y=20, 
            line_dash="dash", 
            line_color="green", 
            row=row, col=1,
            annotation_text="Oversold Opportunity"
        )
    
    def _add_distributions(self, fig: go.Figure, df: pd.DataFrame):
        """Add distribution histograms."""
        
        # Distance % Distribution
        fig.add_trace(go.Histogram(
            x=df['Distance_Pct'] * 100,
            nbinsx=100,
            marker_color='#3498db',
            name='Distance Dist'
        ), row=3, col=1)
        
        # Reversal Score Distribution
        fig.add_trace(go.Histogram(
            x=df['Reversal_Score'],
            nbinsx=50,
            marker_color='#9b59b6',
            name='Score Dist'
        ), row=3, col=2)
    
    def _update_layout(
        self, 
        fig: go.Figure, 
        df: pd.DataFrame,
        show_distributions: bool
    ):
        """Update figure layout."""
        
        height = 1200 if show_distributions else 800
        
        fig.update_layout(
            height=height,
            template='plotly_white',
            showlegend=True,
            title_text=f"Kriterion Quant Analysis: {self.ticker}",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Zoom to last 3 years for clarity
        if len(df) > 0:
            last_date = df.index[-1]
            start_zoom = last_date - pd.Timedelta(days=365*3)
            fig.update_xaxes(range=[start_zoom, last_date], row=1, col=1)
    
    def plot_performance_heatmap(
        self, 
        performance_matrix: pd.DataFrame
    ) -> go.Figure:
        """
        Create a heatmap of performance by band.
        
        Args:
            performance_matrix: Performance matrix DataFrame
            
        Returns:
            Plotly Figure object
        """
        # Extract average return columns
        avg_cols = [c for c in performance_matrix.columns if 'Avg_Ret' in c]
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix[avg_cols].values,
            x=avg_cols,
            y=performance_matrix.index,
            colorscale='RdYlGn',
            zmid=0,
            text=performance_matrix[avg_cols].round(2).values,
            texttemplate='%{text}%',
            textfont={"size": 12},
            hovertemplate='Band: %{y}<br>Horizon: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Mean Return by Band and Horizon',
            xaxis_title='Time Horizon',
            yaxis_title='Statistical Band',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_win_rate_chart(
        self, 
        performance_matrix: pd.DataFrame
    ) -> go.Figure:
        """
        Create a bar chart of win rates by band.
        
        Args:
            performance_matrix: Performance matrix DataFrame
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        wr_cols = [c for c in performance_matrix.columns if 'Win_Rate' in c]
        horizons = [c.split('_')[-1] for c in wr_cols]
        
        for col, horizon in zip(wr_cols, horizons):
            fig.add_trace(go.Bar(
                name=f'{horizon} Win Rate',
                x=performance_matrix.index,
                y=performance_matrix[col],
                text=performance_matrix[col].round(1),
                textposition='auto'
            ))
        
        fig.add_hline(y=50, line_dash="dash", line_color="black", 
                      annotation_text="50% Baseline")
        
        fig.update_layout(
            title='Win Rate by Statistical Band',
            xaxis_title='Band',
            yaxis_title='Win Rate (%)',
            barmode='group',
            height=500,
            template='plotly_white'
        )
        
        return fig
