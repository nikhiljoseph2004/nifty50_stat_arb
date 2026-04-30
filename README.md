# Nifty 50 Data Fetcher

Focused utility to fetch and cache Nifty 50 historical price data from Yahoo Finance.

## Overview

This repository is currently kept intentionally simple. It contains a single core component: the data fetcher.

What it does:
- Downloads adjusted close price series for Nifty 50 symbols
- Caches price data locally as CSV
- Reloads from cache to avoid repeated API calls
- Supports period-based fetches or explicit start/end date ranges
- Provides helper method for daily log-return calculation

This is the baseline state for building a new method slowly and steadily.

## Features

- Automatic data fetching for Nifty 50 symbols
- Local CSV caching and reload support
- Date range filtering on cached data
- Custom symbol universe support
- Log-return helper for quick diagnostics

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

Run the fetch utility with default parameters:

```bash
python main.py
```

### Advanced Usage

Use custom period or explicit date range:

```bash
python main.py --period 3y --show-returns
python main.py --start-date 2022-01-01 --end-date 2024-12-31
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--period` | Historical data period (e.g., 1y, 2y, 5y) | 1y |
| `--start-date` | Start date (YYYY-MM-DD format) | None |
| `--end-date` | End date (YYYY-MM-DD format) | None |
| `--cache-path` | Local CSV path for cached price data | data/nifty50_prices.csv |
| `--refresh-cache` | Re-download data even if a cache exists | False |
| `--head` | Number of rows to print from the top of the price table | 5 |
| `--show-returns` | Compute and print log-return diagnostics | False |

### Caching and Reloading

The DataFetcher writes the downloaded price matrix to `data/nifty50_prices.csv` by default. Subsequent runs load from that file directly, so repeat executions avoid unnecessary re-downloads. Use `--cache-path` for a different file or `--refresh-cache` to force a fresh pull.

### Examples

1. **Fetch 3 years and show returns:**
```bash
python main.py --period 3y --show-returns
```

2. **Custom date range:**
```bash
python main.py --start-date 2020-01-01 --end-date 2023-12-31
```

## Project Structure

```
nifty50_stat_arb/
├── nifty50_stat_arb/
│   ├── __init__.py          # Package initialization
│   ├── data_fetcher.py      # Data fetching module
├── main.py                  # Main execution script
├── examples/                # Usage examples
│   ├── basic_example.py
│   └── advanced_example.py
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Module Documentation

### DataFetcher

Fetches historical price data for Nifty 50 stocks using Yahoo Finance API. Results are cached to `data/nifty50_prices.csv` by default so repeated runs can reuse local data.

```python
from nifty50_stat_arb import DataFetcher

fetcher = DataFetcher()
prices = fetcher.fetch_data(period='2y')
returns = fetcher.get_returns(prices)
```

## Current Scope

This repository intentionally focuses on one thing right now: reliable data collection and caching. Strategy research can be added later in small, controlled steps.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Verify data quality and assumptions before using outputs in any investment workflow.

## Author

Nikhil Joseph

## Acknowledgments

- Yahoo Finance for providing market data
- Statsmodels library for statistical tests
- The quantitative finance community for pairs trading research