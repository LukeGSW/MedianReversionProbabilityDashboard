"""
Unit tests for Median Reversion Dashboard core modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineer import FeatureEngineer
from src.signal_processor import SignalProcessor
from src.stats_engine import StatsEngine


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
    
    # Generate random walk with drift
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Price': prices,
        'Close': prices,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.abs(np.random.uniform(0, 0.02, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.uniform(0, 0.02, len(dates)))),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return df


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_initialization(self):
        """Test default initialization."""
        engineer = FeatureEngineer()
        assert engineer.median_window == 252
    
    def test_custom_window(self):
        """Test custom window initialization."""
        engineer = FeatureEngineer(median_window=126)
        assert engineer.median_window == 126
    
    def test_calculate_features(self, sample_price_data):
        """Test feature calculation."""
        engineer = FeatureEngineer(median_window=252)
        df = engineer.calculate_features(sample_price_data)
        
        # Check required columns exist
        assert 'Median_252' in df.columns
        assert 'Distance_Pct' in df.columns
        assert 'Percentile_Rank' in df.columns
        assert 'Band' in df.columns
        
        # Check no NaN in output (after initial warmup)
        assert df['Median_252'].notna().all()
    
    def test_band_assignment(self, sample_price_data):
        """Test band assignment logic."""
        engineer = FeatureEngineer(median_window=252)
        df = engineer.calculate_features(sample_price_data)
        
        # All bands should be valid
        valid_bands = engineer.get_band_order() + ['N/A']
        assert df['Band'].isin(valid_bands).all()
    
    def test_missing_price_column(self):
        """Test handling of missing Price column."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({'Close': [100, 101, 102]})
        
        # Should use Close as fallback
        result = engineer.calculate_features(df)
        assert 'Price' in result.columns
    
    def test_get_current_status(self, sample_price_data):
        """Test current status extraction."""
        engineer = FeatureEngineer(median_window=252)
        df = engineer.calculate_features(sample_price_data)
        
        status = engineer.get_current_status(df)
        
        assert 'date' in status
        assert 'price' in status
        assert 'median' in status
        assert 'band' in status


class TestSignalProcessor:
    """Tests for SignalProcessor class."""
    
    def test_initialization(self):
        """Test default initialization."""
        processor = SignalProcessor()
        assert processor.order == 20
    
    def test_reversal_score_calculation(self, sample_price_data):
        """Test reversal score calculation."""
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        processor = SignalProcessor()
        df_scored = processor.calculate_real_time_score(df_features)
        
        # Check Reversal Score exists and is in range
        assert 'Reversal_Score' in df_scored.columns
        valid_scores = df_scored['Reversal_Score'].dropna()
        assert (valid_scores >= 0).all()
        assert (valid_scores <= 100).all()
    
    def test_turning_point_detection(self, sample_price_data):
        """Test turning point detection."""
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        processor = SignalProcessor(turning_point_order=20)
        df_scored = processor.calculate_real_time_score(df_features)
        df_tp = processor.detect_ex_post_turning_points(df_scored)
        
        # Should have some turning points
        assert len(df_tp) > 0
        
        # Check required columns
        assert 'Price' in df_tp.columns
        assert 'Type' in df_tp.columns
        
        # Types should be Top or Bottom
        assert df_tp['Type'].isin(['Top', 'Bottom']).all()
    
    def test_signal_summary(self, sample_price_data):
        """Test signal summary generation."""
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        processor = SignalProcessor()
        df_scored = processor.calculate_real_time_score(df_features)
        
        summary = processor.get_signal_summary(df_scored)
        
        assert 'score' in summary
        assert 'signal' in summary
        assert 'interpretation' in summary
        assert 'color' in summary


class TestStatsEngine:
    """Tests for StatsEngine class."""
    
    def test_initialization(self):
        """Test default initialization."""
        engine = StatsEngine()
        assert engine.horizons == [10, 21, 63]
    
    def test_custom_horizons(self):
        """Test custom horizons initialization."""
        engine = StatsEngine(horizons=[5, 10, 20])
        assert engine.horizons == [5, 10, 20]
    
    def test_forward_returns_calculation(self, sample_price_data):
        """Test forward returns calculation."""
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        processor = SignalProcessor()
        df_scored = processor.calculate_real_time_score(df_features)
        
        engine = StatsEngine(horizons=[10, 21])
        df_stats = engine.calculate_forward_returns(df_scored)
        
        # Check forward return columns exist
        assert 'Fwd_Ret_10d' in df_stats.columns
        assert 'Fwd_Ret_21d' in df_stats.columns
    
    def test_performance_matrix_generation(self, sample_price_data):
        """Test performance matrix generation."""
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        processor = SignalProcessor()
        df_scored = processor.calculate_real_time_score(df_features)
        
        engine = StatsEngine(horizons=[10, 21])
        df_stats = engine.calculate_forward_returns(df_scored)
        matrix = engine.generate_performance_matrix(df_stats)
        
        # Check structure
        assert isinstance(matrix.index, pd.Index)
        assert 'Count' in matrix.columns
        
        # Check all bands present
        for band in engine.BAND_ORDER:
            assert band in matrix.index
    
    def test_validation(self, sample_price_data):
        """Test mean reversion validation."""
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        processor = SignalProcessor()
        df_scored = processor.calculate_real_time_score(df_features)
        
        engine = StatsEngine(horizons=[10, 21])
        df_stats = engine.calculate_forward_returns(df_scored)
        matrix = engine.generate_performance_matrix(df_stats)
        
        validation = engine.validate_mean_reversion(matrix)
        
        assert 'is_valid' in validation
        assert 'oversold_edge' in validation
        assert 'overbought_edge' in validation


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_full_pipeline(self, sample_price_data):
        """Test complete analysis pipeline."""
        # Step 1: Feature Engineering
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(sample_price_data)
        
        # Step 2: Signal Processing
        processor = SignalProcessor(turning_point_order=20)
        df_scored = processor.calculate_real_time_score(df_features)
        df_turning_points = processor.detect_ex_post_turning_points(df_scored)
        
        # Step 3: Statistical Backtesting
        engine = StatsEngine(horizons=[10, 21, 63])
        df_stats = engine.calculate_forward_returns(df_scored)
        matrix = engine.generate_performance_matrix(df_stats)
        
        # Final validations
        assert len(df_scored) > 0
        assert len(df_turning_points) > 0
        assert len(matrix) == 7  # 7 bands
    
    def test_edge_cases_short_data(self):
        """Test handling of short data series."""
        # Create very short dataset
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        df = pd.DataFrame({
            'Price': np.random.uniform(100, 110, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates))
        }, index=dates)
        
        engineer = FeatureEngineer(median_window=50)
        df_features = engineer.calculate_features(df)
        
        # Should still work but with fewer rows
        assert len(df_features) < len(df)
        assert len(df_features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
