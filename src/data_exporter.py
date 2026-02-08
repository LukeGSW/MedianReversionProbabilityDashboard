"""
Data Exporter Module
Handles data serialization and export to JSON format.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional

class DataExporter:
    """
    Utility class to package analysis results into a structured JSON format.
    """

    @staticmethod
    def _convert_dates(obj: Any) -> Any:
        """Helper to serialize datetime objects for JSON."""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d')
        raise TypeError(f"Type {type(obj)} not serializable")

    @classmethod
    def prepare_full_export(
        cls, 
        ticker: str,
        df_scored: pd.DataFrame, 
        df_turning_points: Optional[pd.DataFrame], 
        performance_matrix: pd.DataFrame,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Combines all analysis data into a single JSON string.
        
        Args:
            ticker: The asset symbol analyzed
            df_scored: Main time-series dataframe
            df_turning_points: Detected tops/bottoms
            performance_matrix: Statistical backtest results
            metadata: Optional dictionary with run parameters
            
        Returns:
            String containing the full JSON object
        """
        
        # 1. Prepare Metadata
        export_data = {
            "meta": {
                "ticker": ticker,
                "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_observations": len(df_scored),
                **(metadata or {})
            }
        }

        # 2. Process Time Series Data (df_scored)
        # Convert index to column for explicit date handling
        df_scored_clean = df_scored.reset_index().rename(columns={'index': 'Date', 'date': 'Date'})
        # Convert to dict records (list of objects)
        export_data["time_series"] = df_scored_clean.to_dict(orient='records')

        # 3. Process Turning Points
        if df_turning_points is not None and not df_turning_points.empty:
            df_tp_clean = df_turning_points.reset_index().rename(columns={'index': 'Date', 'date': 'Date'})
            export_data["turning_points"] = df_tp_clean.to_dict(orient='records')
        else:
            export_data["turning_points"] = []

        # 4. Process Performance Matrix
        # Reset index to include 'Band' as a field
        perf_clean = performance_matrix.reset_index().rename(columns={'index': 'Band'})
        export_data["statistics"] = perf_clean.to_dict(orient='records')

        # 5. Serialize to JSON with custom date handler
        return json.dumps(
            export_data, 
            default=cls._convert_dates, 
            indent=4
        )
