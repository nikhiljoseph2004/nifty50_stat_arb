# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nikhiljoseph2004/nifty50_stat_arb.git
cd nifty50_stat_arb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Strategy

### 1. Basic Run (Default Parameters)

```bash
python main.py
```

This will:
- Fetch 2 years of Nifty 50 data
- Find cointegrated pairs
- Run the strategy on top 5 pairs
- Display backtest results

### 2. Custom Parameters

```bash
python main.py --period 3y --top-pairs 10 --entry-threshold 1.8
```

### 3. Generate Visualizations

```bash
python main.py --plot
```

This creates PNG files with:
- Equity curves
- Spread charts
- Z-score plots with entry/exit thresholds

## Understanding the Output

The strategy will output:

1. **Data Fetching Status**: Shows number of stocks and date range
2. **Cointegration Results**: Lists pairs with p-values and statistics
3. **Strategy Execution**: Shows progress for each pair
4. **Performance Metrics**:
   - Total Return
   - Annualized Return
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Number of Trades

## Example Output

```
================================================================================
Nifty 50 Statistical Arbitrage Strategy
================================================================================

[Step 1/5] Fetching historical data...
Successfully fetched data: 504 days, 48 stocks

[Step 2/5] Analyzing cointegration...
Testing cointegration for 1128 pairs...
Found 25 cointegrated pairs at 0.05 significance level

[Step 3/5] Analyzing top 5 pairs...

Top Cointegrated Pairs:
--------------------------------------------------------------------------------
1. STOCK1.NS vs STOCK2.NS
   Cointegration p-value: 0.000123
   Hedge ratio: 1.2345
   Correlation: 0.89

[Step 4/5] Running pairs trading strategy...
  Running strategy for STOCK1.NS_vs_STOCK2.NS...

[Step 5/5] Backtesting results...

Backtest Results Summary:
--------------------------------------------------------------------------------

============================================================
Backtest Summary: STOCK1.NS_vs_STOCK2.NS
============================================================
Total Return:           15.23%
Annualized Return:      15.23%
Annualized Volatility:  8.45%
Sharpe Ratio:           1.80
Maximum Drawdown:       -5.67%
Calmar Ratio:           2.69

Trading Statistics:
Number of Trades:       24
Win Rate:               58.33%
Average Win:            0.0234
Average Loss:           -0.0189
============================================================
```

## Running Examples

The repository includes example scripts:

### Basic Example
```bash
cd examples
python basic_example.py
```

### Advanced Example (Multiple Strategies)
```bash
cd examples
python advanced_example.py
```

## Customizing Parameters

Key parameters you can adjust:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `--entry-threshold` | Z-score to enter trade | 1.5 - 3.0 |
| `--exit-threshold` | Z-score to exit trade | 0.3 - 1.0 |
| `--stop-loss` | Z-score for stop loss | 3.0 - 5.0 |
| `--lookback` | Rolling window size | 30 - 120 |

**Tips:**
- Lower entry threshold = More trades, potentially more noise
- Higher entry threshold = Fewer trades, potentially missing opportunities
- Shorter lookback = More responsive to recent changes
- Longer lookback = More stable but slower to adapt

## Troubleshooting

### No Cointegrated Pairs Found

- Try increasing `--significance` (e.g., 0.10)
- Use a longer `--period` (e.g., 3y or 5y)
- Check that data was fetched successfully

### Data Fetching Errors

- Check internet connection
- Yahoo Finance may have rate limits
- Some symbols may not be available

### Poor Performance

- Adjust entry/exit thresholds
- Try different lookback periods
- Consider using more conservative parameters
- Remember: past performance doesn't guarantee future results

## Next Steps

1. Review the generated plots to understand strategy behavior
2. Experiment with different parameters
3. Analyze individual pairs in detail
4. Read the full documentation in README.md
5. Study the code to understand the implementation

## Warning

This is for educational purposes only. Do not use for actual trading without:
- Proper backtesting with transaction costs
- Understanding the risks involved
- Consulting with financial advisors
- Implementing proper risk management
