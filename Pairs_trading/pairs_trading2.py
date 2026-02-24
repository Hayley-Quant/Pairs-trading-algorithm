# simple pairs trading algorithm
# 21 feb 2026
# hayley falloon
# updated version of the other file

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

# Choose a pair of stocks (note you can choose any from yfinance)
ticker1 = "GOOGL"
ticker2 = "GOOG"

# Get historicla data
start_date = "2022-01-01"
end_date = "2024-01-01"

# Download that data
data = yf.download([ticker1, ticker2], start=start_date, end=end_date)

# Get their prices
# if yfinance gives us simply structured data eg rows are (adj...)
if 'Adj Close' in data.columns:
    prices = data['Adj Close']
# if yfinance gives us multi index data eg rows are (adj..., GOOG)
elif ('Adj Close', ticker1) in data.columns:
    prices = pd.DataFrame({
        ticker1: data[('Adj Close', ticker1)],
        ticker2: data[('Adj Close', ticker2)]
    })
# else use the raw closing prices 
else:
    prices = data['Close']

# remove the rows with missing data:
prices = prices.dropna()

# printing how many days of data we have:
print(f"Loaded {len(prices)} days of data")

# cointegration Test:
# defining a function to test for cointegration (Engle-Granger method):
def check_cointegration(prices, ticker1, ticker2):
    # note score means the test statistic, 
    # note coint() is a python function, it produces the critical value.
    score, pvalue, crit_value = coint(prices[ticker1], prices[ticker2])
    print(f"\n--- Cointegration Test ---")
    # prints the pvalue to 4 decimal places: 
    print(f"P-value: {pvalue:.4f}")
    if pvalue < 0.05:
        print("Pairs ARE cointegrated")
    else:
        print("Pairs are NOT cointegrated")
    return pvalue

# defining a function to calculate the hedge ratio:
# runs a linear regression to tell us how many shares of ticker2 we need to 
# hedge 1 share of ticker1. 
def calculate_hedge_ratio(prices, ticker1, ticker2):
    # seperating the prices into two groups (their stocks) - using X as the indept var
    X = prices[ticker1]
    y = prices[ticker2]
    # adding a constant term (alpha = 1) ie the intercept of the regression line:
    X = sm.add_constant(X)
    # run the regression: (note OLS - ordinary least squares)
    # .fit() finds the line that minimises the sum of square residulas. 
    # Note: y (ticker2) goes first in (y,x) so we get the reg formula: y =a+bx+residules
    model = sm.OLS(y, X).fit()
    # extract the hedge ration (beta coefficient) it the slope. 
    hedge_ratio = model.params[ticker1]
    # print these alpha (a) and beta (b) out:
    print(f"Hedge ratio: {hedge_ratio:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")
    # return the hedge ratio to wherever the function was called, so we can use it later. 
    return hedge_ratio

# fancy fancy header:
print("\n" + "="*50)
print(f"PAIRS TRADING ANALYSIS: {ticker1} vs {ticker2}")
print("="*50)

# running the two functions we defined above: 
p_value = check_cointegration(prices, ticker1, ticker2)
hedge_ratio = calculate_hedge_ratio(prices, ticker1, ticker2)

# defining a function to calculate the spread of the two stocks:
def calculate_spread(prices, ticker1, ticker2, hedge_ratio):
    # price of 1st stock minus the price of 2nd stock scaled by hedge ratio:
    spread = prices[ticker1] - hedge_ratio * prices[ticker2]
    return spread

# defining a function to calculate the z-score, using a rolling window of 20 days
def calculate_zscore(spread, window=20):
    # rolling() creates the rolling window, and mean() averages that:
    rolling_mean = spread.rolling(window=window).mean()
    # calculates the standard deviation over each of the windows:
    rolling_std = spread.rolling(window=window).std()
    # calculate the z-score: 
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

# calling the two functions we defined above:
spread = calculate_spread(prices, ticker1, ticker2, hedge_ratio)
zscore = calculate_zscore(spread, window=20)

# defining a function to generate signals to enter & exit a trade (using the zscore as thresholds)
def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.0):
    # create an empty 1D array for each stock, with zscore as the index, and empty data points:
    position1 = pd.Series(index=zscore.index, data=0)
    position2 = pd.Series(index=zscore.index, data=0)
    # tracking the trade statusand our position:
    in_trade = False
    current_position = 0
    # loop thru each day (note starting at day 1 not 0 as we need previous days' close)
    for i in range(1, len(zscore)):
        # if we're flat and looking to enter:
        if not in_trade:
            # entry signals
            if zscore.iloc[i] < -entry_threshold:
                # when the spread is low (long spread): buy stock1, sell stock2
                position1.iloc[i] = 1
                position2.iloc[i] = -1
                in_trade = True
                # noting that were in a long spread position:
                current_position = 1
            elif zscore.iloc[i] > entry_threshold:
                # when the spread is too high (Short spread) sell stock1, buy stock2
                position1.iloc[i] = -1
                position2.iloc[i] = 1
                in_trade = True
                # note were in a short spread position:
                current_position = -1
        # now that were in a trade, we will exit given these signals:
        # note: we may still be holding yesterdy's position. 
        else:
            # exit signals
            if current_position == 1 and zscore.iloc[i] >= exit_threshold:
            # we're in a long position and the spread has reverted back to normal:
                position1.iloc[i] = 0
                position2.iloc[i] = 0
                in_trade = False
                current_position = 0
            # we're in a short position and the spread has reverted back to normal:
            elif current_position == -1 and zscore.iloc[i] <= exit_threshold:
                position1.iloc[i] = 0
                position2.iloc[i] = 0
                in_trade = False
                current_position = 0
            # if the spread has not returned to normal by the end of the day, we keep 
            # our position set as yesterdays' position:
            # ie nothing changed today so leave position same as yesterday's:
            else:
                position1.iloc[i] = position1.iloc[i-1]
                position2.iloc[i] = position2.iloc[i-1]
    
    return position1, position2

# runs the function defined above, and stores a time series of the positions of both stocks:
pos1, pos2 = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.0)

# define a function to backtest 
def backtest(prices, positions1, positions2, ticker1, ticker2):
    # calculating the percent changes between each day's price:
    returns1 = prices[ticker1].pct_change()
    returns2 = prices[ticker2].pct_change()
    
    # calculate strategy return:
    # note we are moving over by 1 each element in position 1&2 with .shift(1):
    strategy_returns = (positions1.shift(1) * returns1) + (positions2.shift(1) * returns2)
    # remove missing daTa including the first entry since we shifted everything over 1:
    strategy_returns = strategy_returns.dropna()
    
    # calculate the cumulative returns:
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # must minus 1 to convert final figure into percentage (note [-1] = last row):
    total_return = cumulative_returns.iloc[-1] - 1
    # calculate the sharpe ratio:
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    print(f"\n--- Backtest Results ---")
    # print the total return as a percent with 2dp
    print(f"Total Return: {total_return:.2%}")
    # print he sharpe with 2dp
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    # count the number of days we make have a position, and how many trades we make:
    changes = (positions1.diff() != 0).sum()
    # divide by 2 because each trade has an entry and exit - both count as a change here:
    trades = changes // 2  
    print(f"Number of trades: {trades}") 
    print(f"Number of days holding a position: {(positions1 != 0).sum()}")
    
    return cumulative_returns, strategy_returns

# run the backtest function:
cum_returns, daily_returns = backtest(prices, pos1, pos2, ticker1, ticker2)


# create 5 plots:
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

