# Nifty 50 Statistical Arbitrage Strategy

A pairs trading strategy that identifies cointegrated pairs in the Nifty 50 index and trades based on mean reversion principles.

## Overview

This project implements a complete statistical arbitrage (stat arb) trading strategy for the Indian stock market, specifically targeting the Nifty 50 index. The strategy:

1. **Identifies Cointegrated Pairs**: Uses statistical tests to find pairs of stocks that move together over time
2. **Calculates Hedge Ratios**: Determines the optimal ratio for pairing stocks
3. **Generates Trading Signals**: Creates entry/exit signals based on z-score deviations
4. **Backtests Performance**: Evaluates strategy performance with comprehensive metrics

## Features

- ✅ Automatic data fetching for all Nifty 50 stocks
- ✅ Cointegration testing using Engle-Granger method
- ✅ Augmented Dickey-Fuller (ADF) test for spread stationarity
- ✅ Z-score based trading signals with configurable thresholds
- ✅ Comprehensive backtesting with performance metrics
- ✅ Position sizing with hedge ratios
- ✅ Risk management with stop-loss mechanisms
- ✅ Visualization of results and equity curves

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nikhiljoseph2004/nifty50_stat_arb.git
cd nifty50_stat_arb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the strategy with default parameters:

```bash
python main.py
```

### Advanced Usage

Customize the strategy parameters:

```bash
python main.py --period 2y --top-pairs 10 --entry-threshold 2.0 --exit-threshold 0.5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--period` | Historical data period (e.g., 1y, 2y, 5y) | 2y |
| `--start-date` | Start date (YYYY-MM-DD format) | None |
| `--end-date` | End date (YYYY-MM-DD format) | None |
| `--top-pairs` | Number of top pairs to analyze | 5 |
| `--significance` | Significance level for cointegration test | 0.05 |
| `--entry-threshold` | Z-score threshold for entry | 2.0 |
| `--exit-threshold` | Z-score threshold for exit | 0.5 |
| `--stop-loss` | Z-score threshold for stop loss | 4.0 |
| `--lookback` | Lookback period for rolling statistics | 60 |
| `--plot` | Generate plots for results | False |
| `--cache-path` | Local CSV path for cached price data | data/nifty50_prices.csv |
| `--refresh-cache` | Re-download data even if a cache exists | False |

### Caching and Reloading

The `DataFetcher` now writes the downloaded price matrix to `data/nifty50_prices.csv` by default. Subsequent runs load that file straight from disk, so the CLI/strategy stack does not re-download data unless you explicitly request it. Use `--cache-path` to point to a different cache or set `--refresh-cache` to force a fresh download (for example, after you change the historical range).

### Examples

1. **Analyze 3-year data with top 10 pairs:**
```bash
python main.py --period 3y --top-pairs 10
```

2. **Custom date range:**
```bash
python main.py --start-date 2020-01-01 --end-date 2023-12-31
```

3. **Aggressive strategy (lower thresholds):**
```bash
python main.py --entry-threshold 1.5 --exit-threshold 0.3
```

4. **Generate plots:**
```bash
python main.py --plot
```

## Strategy Details

### Cointegration

Two stocks are cointegrated if their price series have a long-term equilibrium relationship. When the spread between them deviates from this equilibrium, it tends to revert to the mean, creating trading opportunities.

### Trading Logic

1. **Entry Signal**:
   - When z-score > entry_threshold: Short the spread (short stock1, long stock2)
   - When z-score < -entry_threshold: Long the spread (long stock1, short stock2)

2. **Exit Signal**:
   - Close positions when z-score returns to exit_threshold
   - Stop loss triggered if z-score exceeds stop_loss threshold

3. **Position Sizing**:
   - Positions are sized according to the hedge ratio
   - Stock2 position = -hedge_ratio × Stock1 position

### Performance Metrics

The backtester calculates:
- Total and annualized returns
- Sharpe ratio
- Maximum drawdown
- Calmar ratio
- Win rate and average win/loss
- Number of trades

## Project Structure

```
nifty50_stat_arb/
├── nifty50_stat_arb/
│   ├── __init__.py          # Package initialization
│   ├── data_fetcher.py      # Data fetching module
│   ├── cointegration.py     # Cointegration analysis
│   ├── strategy.py          # Trading strategy implementation
│   └── backtest.py          # Backtesting framework
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Module Documentation

### DataFetcher

Fetches historical price data for Nifty 50 stocks using Yahoo Finance API. The results are cached to `data/nifty50_prices.csv` by default so that repeated runs reuse the local file instead of hitting Yahoo again.

```python
from nifty50_stat_arb import DataFetcher

fetcher = DataFetcher()
prices = fetcher.fetch_data(period='2y')
```

### CointegrationAnalyzer

Analyzes pairs for cointegration relationships.

```python
from nifty50_stat_arb import CointegrationAnalyzer

analyzer = CointegrationAnalyzer(significance_level=0.05)
pairs = analyzer.test_cointegration(prices)
top_pairs = analyzer.get_top_pairs(prices, n_pairs=10)
```

### PairsTrading

Implements the pairs trading strategy.

```python
from nifty50_stat_arb import PairsTrading

strategy = PairsTrading(
    entry_threshold=2.0,
    exit_threshold=0.5,
    stop_loss=4.0
)
results = strategy.run_strategy(prices, 'STOCK1.NS', 'STOCK2.NS', hedge_ratio)
```

### Backtester

Backtests the strategy and calculates performance metrics.

```python
from nifty50_stat_arb import Backtester

backtester = Backtester(initial_capital=100000)
metrics = backtester.backtest_pair(strategy_results, 'Pair1')
backtester.print_summary('Pair1')
```

## Technical Details

### Statistical Tests

1. **Engle-Granger Cointegration Test**
   - Tests whether two non-stationary time series are cointegrated
   - Null hypothesis: No cointegration
   - Rejection of null indicates cointegration

2. **Augmented Dickey-Fuller Test**
   - Tests for stationarity of the spread
   - Stationary spread is essential for mean reversion
   - Null hypothesis: Unit root (non-stationary)

### Risk Management

- **Stop Loss**: Automatically exits positions if spread moves too far against the position
- **Position Sizing**: Uses hedge ratios to balance positions
- **Lookback Period**: Uses rolling statistics to adapt to changing market conditions

## Limitations and Considerations

- **Transaction Costs**: The backtest doesn't include brokerage, taxes, and slippage
- **Market Impact**: Assumes perfect execution at closing prices
- **Liquidity**: Doesn't account for liquidity constraints
- **Parameter Sensitivity**: Results may vary with different parameter choices
- **Forward-Looking Bias**: Uses entire dataset for cointegration testing
- **Regime Changes**: Cointegration relationships may break down over time

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Multiple time-frame analysis
- [ ] Portfolio optimization across multiple pairs
- [ ] Dynamic parameter adjustment
- [ ] Integration with broker APIs for live trading
- [ ] Machine learning for pair selection
- [ ] Transaction cost modeling
- [ ] Risk-adjusted position sizing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without understanding the risks involved. Past performance does not guarantee future results. Always do your own research and consult with financial advisors before making investment decisions.

## Author

Nikhil Joseph

## Acknowledgments

- Yahoo Finance for providing market data
- Statsmodels library for statistical tests
- The quantitative finance community for pairs trading research