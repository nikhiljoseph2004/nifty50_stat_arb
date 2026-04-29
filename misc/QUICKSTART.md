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

## Running the Fetch Utility

### 1. Basic Run (Default Parameters)

```bash
python main.py
```

- Fetch the most recent 1 year of Nifty 50 data
- Cache downloaded prices to `data/nifty50_prices.csv` for reuse
- Use `--refresh-cache` to force a fresh download if you change the date range

### 2. Custom Parameters

```bash
python main.py --period 3y --show-returns
```
You can also use explicit dates:

```bash
python main.py --start-date 2022-01-01 --end-date 2024-12-31
```

## Understanding the Output

The utility will output:

1. **Data Fetching Status**: Shows number of stocks and date range
2. **Price Sample**: Head of the fetched price table
3. **Optional Return Diagnostics**: Daily log-return samples and summary stats

## Example Output

```
================================================================================
Nifty 50 Data Fetch Utility
================================================================================

Fetching historical data...
Successfully fetched data: 252 days, 50 stocks

Rows: 252
Columns (symbols): 50
Date range: 2025-05-01 to 2026-04-30

Price sample:
                  RELIANCE.NS  TCS.NS  INFY.NS
Date
2025-05-01      2865.10  3921.2   1480.4
2025-05-02      2878.35  3934.0   1492.1
```

## Running Examples

The repository includes example scripts:

### Basic Example
```bash
cd examples
python basic_example.py
```

### Advanced Example (Custom Universe + Diagnostics)
```bash
cd examples
python advanced_example.py
```

## Troubleshooting

### Data Fetching Errors

- Check internet connection
- Yahoo Finance may have rate limits
- Some symbols may not be available

## Next Steps

1. Verify cached data quality
2. Experiment with custom symbol lists in examples
3. Build your next method on top of this stable data layer
4. Read the full documentation in README.md

## Warning

This is for educational purposes only. Do not use for actual trading without:
- Verifying data quality and assumptions
