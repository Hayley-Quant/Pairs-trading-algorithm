# simple pairs trading algorithm
# USE THIS TO CHECK IF TWO STOCKS ARE WORTHY
#21 feb 2026
# note: change out the ticker 1 and 2 for other stocks to test their correlation
# we want Cointegration (p < 0.05), rolling corr >0.8ish  and ratio plots close to their mean lines. 


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Choose a pair
ticker1 = "GOOGL"  # Google (voting)
ticker2 = "GOOG"   # Google (non-voting)

# Get data
start_date = "2020-01-01"
end_date = "2025-01-01"

# Download both tickers together (cleaner)
data = yf.download([ticker1, ticker2], start=start_date, end=end_date)

# Show what columns we have
print("Available columns:", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

# Try different ways to get adjusted close
if 'Adj Close' in data.columns:
    # If Adj Close is a top-level column
    prices = data['Adj Close']
elif ('Adj Close', ticker1) in data.columns:
    # If it's a MultiIndex column
    prices = pd.DataFrame({
        ticker1: data[('Adj Close', ticker1)],
        ticker2: data[('Adj Close', ticker2)]
    })
else:
    # Fall back to Close price if Adj Close not available
    print("\n'Adj Close' not found, using 'Close' instead")
    if 'Close' in data.columns:
        prices = data['Close']
    else:
        # Last resort - get the price data directly
        prices = pd.DataFrame({
            ticker1: yf.download(ticker1, start=start_date, end=end_date)['Close'],
            ticker2: yf.download(ticker2, start=start_date, end=end_date)['Close']
        })

# Drop missing values
prices = prices.dropna()

print(f"\nLoaded {len(prices)} days of data")
print("\nFirst few price rows:")
print(prices.head())

# --- Cointegration Test ---
def check_cointegration(prices, ticker1, ticker2):
    score, pvalue, crit_value = coint(prices[ticker1], prices[ticker2])
    
    print(f"\n--- Cointegration Test ---")
    print(f"P-value: {pvalue:.4f}")
    if pvalue < 0.05:
        print("✓ Pairs ARE cointegrated")
    else:
        print("✗ Pairs are NOT cointegrated")
    return pvalue

def calculate_hedge_ratio(prices, ticker1, ticker2):
    X = prices[ticker1]
    y = prices[ticker2]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params[ticker1]
    
    print(f"\nHedge ratio: {hedge_ratio:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")
    return hedge_ratio

# Run analysis
print("\n" + "="*50)
print("PAIRS TRADING ANALYSIS: "+ ticker1 + " vs " + ticker2)
print("="*50)

p_value = check_cointegration(prices, ticker1, ticker2)
hedge_ratio = calculate_hedge_ratio(prices, ticker1, ticker2)

import matplotlib.pyplot as plt

# Create a visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Normalized prices (to compare movement)
norm_prices = prices / prices.iloc[0]  # Normalize to 100 at start
axes[0].plot(norm_prices.index, norm_prices[ticker1], label=ticker1)
axes[0].plot(norm_prices.index, norm_prices[ticker2], label=ticker2)
axes[0].set_title(f'{ticker1} vs {ticker2} - Normalized Prices')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Price ratio
ratio = prices[ticker1] / prices[ticker2]
axes[1].plot(ratio.index, ratio, color='purple')
axes[1].axhline(y=ratio.mean(), color='red', linestyle='--', label=f'Mean: {ratio.mean():.2f}')
axes[1].set_title('Price Ratio (with mean)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Rolling correlation
rolling_corr = prices[ticker1].rolling(window=60).corr(prices[ticker2])
axes[2].plot(rolling_corr.index, rolling_corr, color='orange')
axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[2].set_title('60-Day Rolling Correlation')
axes[2].set_ylabel('Correlation')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some stats
print(f"\n--- Additional Statistics ---")
print(f"Mean ratio: {ratio.mean():.4f}")
print(f"Ratio std dev: {ratio.std():.4f}")
print(f"Current ratio: {ratio.iloc[-1]:.4f}")
print(f"Rolling correlation (last thing): {rolling_corr.iloc[-1]:.4f}")
