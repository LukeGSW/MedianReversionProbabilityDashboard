"""
Median Reversion Probability Analysis
Kriterion Quant Project

Core modules for quantitative mean reversion analysis.
"""

from .data_client import EODHDDataClient
from .feature_engineer import FeatureEngineer
from .signal_processor import SignalProcessor
from .stats_engine import StatsEngine
from .visualizer import Visualizer
from .data_exporter import DataExporter  # <--- AGGIUNTA

__all__ = [
    'EODHDDataClient',
    'FeatureEngineer', 
    'SignalProcessor',
    'StatsEngine',
    'Visualizer',
    'DataExporter'  # <--- AGGIUNTA
]

__version__ = '1.0.0'
__author__ = 'Kriterion Quant'
