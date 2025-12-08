# 📊 Median Reversion Probability Dashboard

**Kriterion Quant Project** - A quantitative analysis tool for identifying mean reversion opportunities in financial markets.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This dashboard implements a **Median Reversion Strategy** analysis tool that:

1. **Calculates Rolling Median** - Uses 252-day (1 trading year) rolling median as the "gravity center"
2. **Identifies Statistical Bands** - Computes expanding quantiles (5%, 20%, 80%, 95%) to define overbought/oversold zones
3. **Generates Reversal Score** - A real-time indicator (0-100) based on percentile rank
4. **Backtests Historical Performance** - Validates the mean reversion hypothesis with forward returns analysis
5. **Detects Turning Points** - Identifies historical tops and bottoms for visual validation

---

## 📈 Key Features

- **Dynamic Ticker Analysis**: Analyze any US stock, ETF, or cryptocurrency
- **Interactive Charts**: Plotly-powered visualizations with zoom and hover
- **Statistical Validation**: Performance matrix showing win rates and average returns per band
- **AI Insights**: Automated interpretation of current market position
- **Export Capabilities**: Download reports in HTML format

---

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/median-reversion-dashboard.git
cd median-reversion-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Environment Variables

Create a `.env` file or set the following environment variable:

```bash
EODHD_API_KEY=your_api_key_here
```

You can get a free API key from [EODHD](https://eodhd.com/).

---

## 📁 Project Structure

```
median-reversion-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── data_client.py        # EODHD API data fetching
│   ├── feature_engineer.py   # Median & percentile calculations
│   ├── signal_processor.py   # Reversal score & turning points
│   ├── stats_engine.py       # Statistical backtesting
│   └── visualizer.py         # Plotly chart generation
├── assets/
│   └── style.css             # Custom CSS styling
└── tests/
    └── test_core.py          # Unit tests
```

---

## 🧠 Methodology

### 1. Rolling Median (252 days)

The median is more robust than the mean against outliers and represents the "fair value" around which prices oscillate.

### 2. Distance Percentage

```
Distance % = (Price / Median) - 1
```

Measures how far the current price is from its statistical center.

### 3. Percentile Rank (Expanding)

Ranks the current distance against all historical distances, producing a value between 0-1.

### 4. Statistical Bands

| Band | Percentile Range | Interpretation |
|------|------------------|----------------|
| Extreme Oversold | < 5% | Strong buy signal |
| Oversold | 5-20% | Buy opportunity |
| Low Neutral | 20-45% | Slightly undervalued |
| Neutral | 45-55% | Fair value |
| High Neutral | 55-80% | Slightly overvalued |
| Overbought | 80-95% | Sell consideration |
| Extreme Overbought | > 95% | Strong sell signal |

### 5. Reversal Score

```
Reversal Score = Percentile Rank × 100
```

- **Score > 80**: High probability of bearish reversal
- **Score < 20**: High probability of bullish reversal

---

## 📊 Interpreting the Dashboard

### Main Chart
- **Black Line**: 252-day Rolling Median (gravity center)
- **Red Zone**: Extreme overbought territory (>95th percentile)
- **Green Zone**: Extreme oversold territory (<5th percentile)
- **Dotted Lines**: 20% and 80% probability bands
- **Triangles**: Historical turning points (ex-post validation)

### Reversal Score Panel
- Real-time oscillator showing current market positioning
- Color-coded: Red (>80), Green (<20), Gray (neutral)

### Performance Matrix
- Historical win rates and average returns for each band
- Validates mean reversion hypothesis for the specific asset

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

**Kriterion Quant** - [kriterionquant.com](https://kriterionquant.com)

---

*Powered by Python, Streamlit, Plotly & EODHD*
