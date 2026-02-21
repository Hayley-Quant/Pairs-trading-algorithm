# Pairs Trading Algorithm: GOOGL/GOOG

A statistical arbitrage strategy that identifies and trades cointegrated stock pairs. This implementation achieves a **1.99 Sharpe ratio** on GOOGL/GOOG with a 5.12% return over the 2022-2024 period.

##  Strategy Performance

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 1.99 |
| **Total Return** | 5.12% |
| **Cointegration P-Value** | 0.016 |
| **Number of Trades** | 127 |
| **Win Rate** | ~52% |
| **Max Drawdown** | -3.2% |

![Strategy Results](pairs_trading_results.png)

##  How It Works

1. **Pair Selection**: Tests for cointegration using the Engle-Granger method
2. **Hedge Ratio**: Calculated via OLS regression to create a stationary spread
3. **Entry Signals**: Enter when z-score exceeds ±2.0 (spread is 2 std dev from mean)
4. **Exit Signals**: Exit when z-score crosses zero (spread reverts to mean)
5. **Risk Management**: Position sizing based on hedge ratio, no leverage assumed

##  Technologies Used

- **Python 3.9+**
- **yfinance** – Data acquisition
- **pandas, numpy** – Data manipulation
- **statsmodels** – Cointegration testing and regression
- **matplotlib** – Performance visualization

##  Repository Structure
pairs-trading-algorithm/
├── pairs_trading.py # Main algorithm
├── README.md # This file
├── requirements.txt # Dependencies
├── pairs_trading_results.png # Strategy visualization
└── LICENSE # GPL-3.0 license

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pairs-trading-algorithm.git
   cd pairs-trading-algorithm