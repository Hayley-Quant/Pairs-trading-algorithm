**PAIRS TRADING ALGORITHM**

A statistical arbitrage strategy that identifies and trades cointegrated stock pairs.
This implementation achieves a 1.99 Sharpe ratio on GOOGL/GOOG with a 5.12% return over the 2022-2024 period.

**How it works:**

1. Pair Selection: Tests for cointegration using the Engle-Granger method
2. Hedge Ratio: Calculated via OLS regression to create a stationary spread (simple linear)
3. Entry Signals: Enter when z-score exceeds ±2.0 (spread is 2 std dev from mean)
4. Exit Signals: Exit when z-score crosses zero (spread reverts to mean)
5. Risk Management: Position sizing based on hedge ratio, assuming no leverage.


**Languages + extensions used:**
  
- Python 3.9+  
- yfinance – Data acquisition  
- pandas, numpy – Data manipulation  
- statsmodels– Cointegration testing and regression  
- matplotlib – Performance visualization  
  
  
**Example:**

---  
PAIRS TRADING ANALYSIS: GOOGL vs GOOG
===  

--- Cointegration Test ---
P-value: 0.0163  
Pairs ARE cointegrated  
Hedge ratio: 1.0048  
R-squared: 0.9994  

--- Backtest Results ---  
Total Return: 5.12%  
Sharpe Ratio: 1.99  
Number of trades: 18  
Number of days holding a position: 127  

---  
EXTRA STATISTICS  
=== 

---Trade statistics ---    
Total trades: 17  
Winning trades: 16  
Losing trades: 1  
Trade win rate: 94.1%  
Average trade return: 0.29%  
Average win: 0.31%  
Average loss: -0.02%  
Average trade duration: 7.3 days  
Profit factor: 277.27  


---  
END
===  
