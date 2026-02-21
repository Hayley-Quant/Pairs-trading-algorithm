# pairs trading algorithm
#21 feb 2026
# SEQUAL TO OTHER FILE

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

# Choose a pair (GOOGL and GOOG)
ticker1 = "GOOGL"
ticker2 = "GOOG"

# Get data
start_date = "2022-01-01"
end_date = "2024-01-01"

# Download data
data = yf.download([ticker1, ticker2], start=start_date, end=end_date)

# Get prices
if 'Adj Close' in data.columns:
    prices = data['Adj Close']
elif ('Adj Close', ticker1) in data.columns:
    prices = pd.DataFrame({
        ticker1: data[('Adj Close', ticker1)],
        ticker2: data[('Adj Close', ticker2)]
    })
else:
    prices = data['Close']

prices = prices.dropna()

print(f"Loaded {len(prices)} days of data")

# --- PART 1: Cointegration Test ---
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
    print(f"Hedge ratio: {hedge_ratio:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")
    return hedge_ratio

print("\n" + "="*50)
print(f"PAIRS TRADING ANALYSIS: {ticker1} vs {ticker2}")
print("="*50)

p_value = check_cointegration(prices, ticker1, ticker2)
hedge_ratio = calculate_hedge_ratio(prices, ticker1, ticker2)

# --- PART 2: Calculate Spread and Z-Score ---
def calculate_spread(prices, ticker1, ticker2, hedge_ratio):
    spread = prices[ticker1] - hedge_ratio * prices[ticker2]
    return spread

def calculate_zscore(spread, window=20):
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

spread = calculate_spread(prices, ticker1, ticker2, hedge_ratio)
zscore = calculate_zscore(spread, window=20)

# --- PART 3: Generate Trading Signals ---
def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.0):
    position1 = pd.Series(index=zscore.index, data=0)
    position2 = pd.Series(index=zscore.index, data=0)
    
    in_trade = False
    current_position = 0
    
    for i in range(1, len(zscore)):
        if not in_trade:
            # Entry signals
            if zscore.iloc[i] < -entry_threshold:
                # Spread too low: Long spread (buy stock1, sell stock2)
                position1.iloc[i] = 1
                position2.iloc[i] = -1
                in_trade = True
                current_position = 1
            elif zscore.iloc[i] > entry_threshold:
                # Spread too high: Short spread (sell stock1, buy stock2)
                position1.iloc[i] = -1
                position2.iloc[i] = 1
                in_trade = True
                current_position = -1
        else:
            # Exit signals
            if current_position == 1 and zscore.iloc[i] >= exit_threshold:
                position1.iloc[i] = 0
                position2.iloc[i] = 0
                in_trade = False
                current_position = 0
            elif current_position == -1 and zscore.iloc[i] <= exit_threshold:
                position1.iloc[i] = 0
                position2.iloc[i] = 0
                in_trade = False
                current_position = 0
            else:
                position1.iloc[i] = position1.iloc[i-1]
                position2.iloc[i] = position2.iloc[i-1]
    
    return position1, position2

pos1, pos2 = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.0)

# --- PART 4: Backtest ---
def backtest(prices, positions1, positions2, ticker1, ticker2):
    returns1 = prices[ticker1].pct_change()
    returns2 = prices[ticker2].pct_change()
    
    strategy_returns = (positions1.shift(1) * returns1) + (positions2.shift(1) * returns2)
    strategy_returns = strategy_returns.dropna()
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    print(f"\n--- Backtest Results ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Number of trades: {(positions1 != 0).sum()}")
    
    return cumulative_returns, strategy_returns

cum_returns, daily_returns = backtest(prices, pos1, pos2, ticker1, ticker2)

# --- PART 5: Visualization with 5 Plots (including Drawdown) ---
fig, axes = plt.subplots(5, 1, figsize=(12, 18))

# Plot 1: Normalized prices
norm_prices = prices / prices.iloc[0]
axes[0].plot(norm_prices.index, norm_prices[ticker1], label=ticker1, linewidth=1.5)
axes[0].plot(norm_prices.index, norm_prices[ticker2], label=ticker2, linewidth=1.5)
axes[0].set_title(f'{ticker1} vs {ticker2} - Normalized Prices (Base=100)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Z-Score with signals
axes[1].plot(zscore.index, zscore, color='blue', alpha=0.7, linewidth=1)
axes[1].axhline(y=2, color='red', linestyle='--', label='Entry (+2)')
axes[1].axhline(y=-2, color='green', linestyle='--', label='Entry (-2)')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_title('Z-Score with Trading Signals')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Color the trades on z-score plot
for i in range(len(zscore)):
    if pos1.iloc[i] != 0:
        color = 'yellow' if pos1.iloc[i] == 1 else 'orange'
        axes[1].axvline(x=zscore.index[i], color=color, alpha=0.15)

# Plot 3: Spread
axes[2].plot(spread.index, spread, color='purple', linewidth=1)
axes[2].axhline(y=spread.mean(), color='red', linestyle='--', label=f'Mean: {spread.mean():.2f}')
axes[2].fill_between(spread.index, 
                     spread.mean() - spread.std(), 
                     spread.mean() + spread.std(), 
                     alpha=0.2, color='gray', label='±1 Std Dev')
axes[2].set_title('Spread with Mean and Standard Deviation')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Cumulative returns
axes[3].plot(cum_returns.index, cum_returns, color='green', linewidth=2)
axes[3].axhline(y=1, color='black', linestyle='-', alpha=0.3)
axes[3].set_title('Strategy Cumulative Returns')
axes[3].set_ylabel('Cumulative Return (x)')
axes[3].grid(True, alpha=0.3)

# Plot 5: Drawdown
rolling_max = cum_returns.expanding().max()
drawdown = (cum_returns - rolling_max) / rolling_max * 100
axes[4].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.4)
axes[4].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[4].set_title('Strategy Drawdown %')
axes[4].set_ylabel('Drawdown %')
axes[4].grid(True, alpha=0.3)

# Add max drawdown annotation
max_drawdown = drawdown.min()
max_dd_date = drawdown.idxmin()
axes[4].annotate(f'Max Drawdown: {max_drawdown:.1f}%', 
                xy=(max_dd_date, max_drawdown),
                xytext=(max_dd_date, max_drawdown - 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10)

plt.tight_layout()
plt.savefig('pairs_trading_results.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PART 6: Additional Statistics ---
ratio = prices[ticker1] / prices[ticker2]
print(f"\n--- Ratio Statistics ---")
print(f"Mean ratio: {ratio.mean():.4f}")
print(f"Ratio std dev: {ratio.std():.4f}")
print(f"Current ratio: {ratio.iloc[-1]:.4f}")

# Calculate and print max drawdown
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Max Drawdown Date: {max_dd_date.strftime('%Y-%m-%d')}")

# Win rate
winning_trades = (daily_returns > 0).sum()
total_trading_days = len(daily_returns)
win_rate = winning_trades / total_trading_days * 100
print(f"Win Rate: {win_rate:.1f}%")