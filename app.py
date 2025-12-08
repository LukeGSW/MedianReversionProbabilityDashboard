"""
Median Reversion Probability Dashboard
Kriterion Quant Project

Main Streamlit application for quantitative mean reversion analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import custom modules
from src.data_client import EODHDDataClient
from src.feature_engineer import FeatureEngineer
from src.signal_processor import SignalProcessor
from src.stats_engine import StatsEngine
from src.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="Median Reversion Dashboard | Kriterion Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_scored' not in st.session_state:
    st.session_state.df_scored = None
if 'df_turning_points' not in st.session_state:
    st.session_state.df_turning_points = None
if 'performance_matrix' not in st.session_state:
    st.session_state.performance_matrix = None


def get_api_key() -> str:
    """Get API key from environment or secrets."""
    # Try Streamlit secrets first
    try:
        return st.secrets["EODHD_API_KEY"]
    except:
        pass
    
    # Try environment variable
    api_key = os.getenv("EODHD_API_KEY", "")
    return api_key


def generate_insight_html(ticker: str, df_scored: pd.DataFrame, performance_matrix: pd.DataFrame) -> tuple:
    """
    Generate dynamic insight based on current position and historical statistics.
    
    Returns:
        Tuple of (insight_text, sentiment, color)
    """
    last_row = df_scored.iloc[-1]
    current_band = last_row['Band']
    current_score = last_row['Reversal_Score']
    current_price = last_row['Price']
    
    # Get statistics for current band
    try:
        stats_row = performance_matrix.loc[current_band]
        win_rate_21d = stats_row.get('Win_Rate_21d', 'N/A')
        avg_ret_21d = stats_row.get('Avg_Ret_21d', 'N/A')
    except:
        win_rate_21d = "N/A"
        avg_ret_21d = "N/A"
    
    # Determine sentiment and advice
    if "Oversold" in current_band:
        sentiment = "BULLISH (Mean Reversion Opportunity)"
        advice = f"Historically, when {ticker} is in this zone, it tends to bounce back."
        color = "green"
    elif "Overbought" in current_band:
        sentiment = "BEARISH / CAUTION (Extended)"
        advice = "Price is statistically extended. Future returns tend to compress."
        color = "red"
    else:
        sentiment = "NEUTRAL (Trend Following)"
        advice = "Price is in fair value zone. Follow the prevailing trend without statistical extremes."
        color = "blue"
    
    insight = {
        'date': last_row.name.strftime('%Y-%m-%d'),
        'price': current_price,
        'band': current_band,
        'score': current_score,
        'win_rate': win_rate_21d,
        'avg_return': avg_ret_21d,
        'sentiment': sentiment,
        'advice': advice,
        'color': color
    }
    
    return insight


def run_analysis(ticker: str, api_key: str, start_date: datetime, end_date: datetime) -> bool:
    """
    Execute the complete analysis pipeline.
    
    Returns:
        True if successful, False otherwise
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Download
        status_text.text("📥 Downloading data from EODHD...")
        progress_bar.progress(10)
        
        client = EODHDDataClient(api_key=api_key)
        df_raw = client.get_historical_data(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if df_raw.empty:
            st.error("❌ No data returned. Check ticker symbol and API key.")
            return False
        
        progress_bar.progress(30)
        
        # Step 2: Feature Engineering
        status_text.text("⚙️ Calculating median and percentiles...")
        engineer = FeatureEngineer(median_window=252)
        df_features = engineer.calculate_features(df_raw)
        progress_bar.progress(50)
        
        # Step 3: Signal Processing
        status_text.text("🔍 Detecting signals and turning points...")
        processor = SignalProcessor(turning_point_order=20)
        df_scored = processor.calculate_real_time_score(df_features)
        df_turning_points = processor.detect_ex_post_turning_points(df_scored)
        progress_bar.progress(70)
        
        # Step 4: Statistical Backtesting
        status_text.text("📊 Running statistical backtest...")
        stats_engine = StatsEngine(horizons=[10, 21, 63])
        df_stats = stats_engine.calculate_forward_returns(df_scored)
        performance_matrix = stats_engine.generate_performance_matrix(df_stats)
        progress_bar.progress(90)
        
        # Store in session state
        st.session_state.df_scored = df_scored
        st.session_state.df_turning_points = df_turning_points
        st.session_state.performance_matrix = performance_matrix
        st.session_state.ticker = ticker
        st.session_state.data_loaded = True
        st.session_state.price_type = df_raw['price_type'].iloc[0] if 'price_type' in df_raw.columns else 'Close'
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return False
    finally:
        progress_bar.empty()
        status_text.empty()


def main():
    # Header
    st.title("📊 Median Reversion Probability Dashboard")
    st.markdown("*Kriterion Quant Project - Quantitative Mean Reversion Analysis*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "EODHD API Key",
            value=get_api_key(),
            type="password",
            help="Get your free API key at eodhd.com"
        )
        
        st.divider()
        
        # Ticker Selection
        ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Enter US stock/ETF symbol (e.g., SPY, AAPL, QQQ)"
        ).upper()
        
        # Exchange Selection
        exchange = st.selectbox(
            "Exchange",
            options=['US', 'LSE', 'PA', 'XETRA', 'TO', 'HK', 'CC', 'FOREX', 'INDX'],
            index=0,
            help="Select the exchange (US for American stocks)"
        )
        
        st.divider()
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2010, 1, 1),
                min_value=datetime(2000, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                min_value=datetime(2000, 1, 1),
                max_value=datetime.now()
            )
        
        st.divider()
        
        # Analysis Parameters
        st.subheader("📐 Parameters")
        median_window = st.slider(
            "Median Window (days)",
            min_value=50,
            max_value=504,
            value=252,
            help="Rolling window for median calculation (252 = 1 trading year)"
        )
        
        turning_point_order = st.slider(
            "Turning Point Sensitivity",
            min_value=5,
            max_value=50,
            value=20,
            help="Days of confirmation for local highs/lows"
        )
        
        st.divider()
        
        # Run Button
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_button:
            if not api_key:
                st.error("Please enter an API key")
            elif not ticker:
                st.error("Please enter a ticker symbol")
            else:
                run_analysis(ticker, api_key, start_date, end_date)
    
    # Main Content Area
    if st.session_state.data_loaded:
        df_scored = st.session_state.df_scored
        df_turning_points = st.session_state.df_turning_points
        performance_matrix = st.session_state.performance_matrix
        ticker = st.session_state.ticker
        
        # AI Insight Box
        st.header("⚡ AI Insight")
        insight = generate_insight_html(ticker, df_scored, performance_matrix)
        
        # Color-coded insight box
        if insight['color'] == 'green':
            box_color = "#d4efdf"
            border_color = "#27ae60"
        elif insight['color'] == 'red':
            box_color = "#fadbd8"
            border_color = "#c0392b"
        else:
            box_color = "#ebf5fb"
            border_color = "#2980b9"
        
        st.markdown(f"""
        <div style="background-color: {box_color}; border-left: 6px solid {border_color}; 
                    padding: 20px; margin-bottom: 25px; border-radius: 4px;">
            <h3 style="margin-top: 0; color: {border_color};">📌 {insight['sentiment']}</h3>
            <p><strong>Current Status ({insight['date']}):</strong>
               {ticker} price ({insight['price']:.2f}) is in the <strong>{insight['band']}</strong> band
               with a Reversal Score of <strong>{insight['score']:.1f}/100</strong>.</p>
            <p><strong>Historical Statistics (Backtest):</strong><br>
               When entering this band ({insight['band']}), the asset historically showed:
               <ul>
                   <li>Win Rate at 1 month: <strong>{insight['win_rate']}%</strong></li>
                   <li>Average Return (21 days): <strong>{insight['avg_return']}%</strong></li>
               </ul>
            </p>
            <p style="font-style: italic; font-size: 0.9em;">💡 <strong>Interpretation:</strong> {insight['advice']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        st.header("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        last_row = df_scored.iloc[-1]
        with col1:
            st.metric(
                "Current Price",
                f"${last_row['Price']:.2f}",
                delta=f"{((last_row['Price']/df_scored['Price'].iloc[-2])-1)*100:.2f}%"
            )
        with col2:
            st.metric(
                "252d Median",
                f"${last_row['Median_252']:.2f}"
            )
        with col3:
            distance = last_row['Distance_Pct'] * 100
            st.metric(
                "Distance from Median",
                f"{distance:+.2f}%"
            )
        with col4:
            score = last_row['Reversal_Score']
            st.metric(
                "Reversal Score",
                f"{score:.1f}/100"
            )
        
        # Main Chart
        st.header("📊 Price Analysis Chart")
        
        viz = Visualizer(ticker=ticker)
        fig_main = viz.plot_analysis(df_scored, df_turning_points, show_distributions=False)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Interpretation Guide
        with st.expander("📖 How to Read This Chart"):
            st.markdown("""
            **Understanding the Visualization:**
            
            - **Black Line**: 252-day Rolling Median - the "gravity center" of price
            - **Red Zone (top)**: Extreme overbought territory (>95th percentile) - high correction risk
            - **Green Zone (bottom)**: Extreme oversold territory (<5th percentile) - high bounce probability
            - **Dotted Red Line**: 80% resistance level
            - **Dotted Green Line**: 20% support level
            - **🔻 Red Triangles**: Historical market tops
            - **🔺 Green Triangles**: Historical market bottoms
            
            **Reversal Score Interpretation:**
            - **Score > 80**: High probability of bearish reversal (overbought)
            - **Score < 20**: High probability of bullish reversal (oversold)
            - **Score 20-80**: Neutral zone - follow prevailing trend
            """)
        
        # Statistical Backtest Results
        st.header("📊 Statistical Backtest Matrix")
        
        # Performance Matrix Display
        st.dataframe(
            performance_matrix.style.background_gradient(
                cmap='RdYlGn',
                subset=[c for c in performance_matrix.columns if 'Avg_Ret' in c]
            ).format({
                c: '{:.2f}%' for c in performance_matrix.columns if 'Ret' in c
            }).format({
                c: '{:.1f}%' for c in performance_matrix.columns if 'Win_Rate' in c
            }),
            use_container_width=True
        )
        
        # Mean Reversion Validation
        stats_engine = StatsEngine()
        validation = stats_engine.validate_mean_reversion(performance_matrix)
        
        if validation.get('is_valid'):
            st.success("✅ **Mean Reversion Hypothesis VALIDATED** for this asset")
        else:
            st.warning("⚠️ **Mean Reversion Hypothesis PARTIAL** - Check individual band performance")
        
        for detail in validation.get('details', []):
            st.info(detail)
        
        # Additional Charts in Tabs
        st.header("📈 Additional Analysis")
        tab1, tab2, tab3 = st.tabs(["Win Rate Chart", "Return Heatmap", "Data Summary"])
        
        with tab1:
            fig_wr = viz.plot_win_rate_chart(performance_matrix)
            st.plotly_chart(fig_wr, use_container_width=True)
        
        with tab2:
            fig_hm = viz.plot_performance_heatmap(performance_matrix)
            st.plotly_chart(fig_hm, use_container_width=True)
        
        with tab3:
            st.subheader("Dataset Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Observations:** {len(df_scored):,}")
                st.write(f"**Date Range:** {df_scored.index[0].strftime('%Y-%m-%d')} to {df_scored.index[-1].strftime('%Y-%m-%d')}")
                st.write(f"**Price Type:** {st.session_state.get('price_type', 'N/A')}")
            with col2:
                st.write(f"**Turning Points Detected:** {len(df_turning_points)}")
                if not df_turning_points.empty:
                    tops = len(df_turning_points[df_turning_points['Type'] == 'Top'])
                    bottoms = len(df_turning_points[df_turning_points['Type'] == 'Bottom'])
                    st.write(f"  - Tops: {tops}")
                    st.write(f"  - Bottoms: {bottoms}")
            
            # Show recent data
            st.subheader("Recent Data (Last 10 Days)")
            display_cols = ['Price', 'Median_252', 'Distance_Pct', 'Percentile_Rank', 'Reversal_Score', 'Band']
            st.dataframe(df_scored[display_cols].tail(10))
        
        # Export Section
        st.header("💾 Export Report")
        
        col1, col2 = st.columns(2)
        with col1:
            # CSV Export
            csv = performance_matrix.to_csv()
            st.download_button(
                label="📄 Download Performance Matrix (CSV)",
                data=csv,
                file_name=f"performance_matrix_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Full data CSV
            full_csv = df_scored.to_csv()
            st.download_button(
                label="📊 Download Full Dataset (CSV)",
                data=full_csv,
                file_name=f"full_data_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen when no data loaded
        st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")
        
        st.markdown("""
        ### 🎯 About This Dashboard
        
        This tool implements a **Median Reversion Strategy** analysis that:
        
        1. **Calculates Rolling Median** - Uses 252-day rolling median as the "gravity center"
        2. **Identifies Statistical Bands** - Computes expanding quantiles to define overbought/oversold zones
        3. **Generates Reversal Score** - A real-time indicator (0-100) based on percentile rank
        4. **Backtests Historical Performance** - Validates mean reversion with forward returns analysis
        
        ### 📊 Key Concepts
        
        | Band | Percentile | Interpretation |
        |------|------------|----------------|
        | Extreme Oversold | < 5% | Strong buy signal |
        | Oversold | 5-20% | Buy opportunity |
        | Neutral | 45-55% | Fair value |
        | Overbought | 80-95% | Sell consideration |
        | Extreme Overbought | > 95% | Strong sell signal |
        
        ### ⚠️ Disclaimer
        
        This tool is for **educational purposes only**. Past performance does not guarantee future results.
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.85em;">
        © 2025 Kriterion Quant Project | Powered by Python, Streamlit, Plotly & EODHD
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
