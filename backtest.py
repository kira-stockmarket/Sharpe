import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# 1. Define Assets (Current Top 10 Nifty 50 Constituents by Weight)
tickers = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'ITC.NS', 'TCS.NS', 'LT.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'AXISBANK.NS'
]
benchmark = '^NSEI'

# Fetch Data
start_date = '2010-01-01'
print("Downloading data...")
data = yf.download(tickers + [benchmark], start=start_date)['Adj Close']
data = data.ffill().dropna()

# Separate Benchmark and Portfolio Returns
returns = data.pct_change().dropna()
bench_ret = returns[benchmark]
port_ret_data = returns[tickers]

# 2. Max Sharpe Optimization Function
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.07):
    num_assets = len(mean_returns)
    
    def neg_sharpe(weights):
        p_ret = np.sum(mean_returns * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return -(p_ret - risk_free_rate) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    optimized = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized.x

# 3. Rolling Annual Rebalancing Backtest
years = port_ret_data.index.year.unique()
strategy_returns = pd.Series(dtype=float)
weights_history = []

print("Running optimization and backtest...")
for i in range(1, len(years)):
    prev_year = years[i-1]
    curr_year = years[i]
    
    # Train on previous year
    train_data = port_ret_data[port_ret_data.index.year == prev_year]
    mean_ret = train_data.mean()
    cov_mat = train_data.cov()
    
    # Get optimal weights
    opt_weights = max_sharpe_ratio(mean_ret, cov_mat)
    weights_history.append((curr_year, dict(zip(tickers, np.round(opt_weights, 4)))))
    
    # Apply to current year (Out of Sample)
    test_data = port_ret_data[port_ret_data.index.year == curr_year]
    strat_year_ret = test_data.dot(opt_weights)
    strategy_returns = pd.concat([strategy_returns, strat_year_ret])

# Align benchmark returns to the backtested period
aligned_bench_ret = bench_ret.loc[strategy_returns.index]

# 4. Calculate Cumulative Returns
cum_strategy = (1 + strategy_returns).cumprod()
cum_benchmark = (1 + aligned_bench_ret).cumprod()

# 5. Generate Visuals and Reports
plt.figure(figsize=(12, 6))
plt.plot(cum_strategy, label='Top 10 Max Sharpe (Annual Rebalance)')
plt.plot(cum_benchmark, label='Nifty 50 (Buy & Hold)')
plt.title('Nifty 50 vs Max Sharpe Optimized Top 10')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig('results/performance_chart.png')
plt.close()

# Generate Markdown Report
total_strat_return = (cum_strategy.iloc[-1] - 1) * 100
total_bench_return = (cum_benchmark.iloc[-1] - 1) * 100

with open('results/report.md', 'w') as f:
    f.write("# Portfolio Optimization Results\n\n")
    f.write(f"**Since {years[1]}**\n\n")
    f.write(f"- **Top 10 Max Sharpe Strategy Cumulative Return:** {total_strat_return:.2f}%\n")
    f.write(f"- **Nifty 50 Buy & Hold Cumulative Return:** {total_bench_return:.2f}%\n\n")
    
    f.write("### Latest Target Weights (for current year execution)\n")
    latest_weights = weights_history[-1][1]
    for ticker, weight in latest_weights.items():
        if weight > 0.001:
            f.write(f"- **{ticker}**: {weight*100:.2f}%\n")

print("Backtest complete. Results saved to /results.")
