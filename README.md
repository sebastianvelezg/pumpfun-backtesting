# PumpfunBacktester

A configurable Python-based backtesting tool for analyzing Solana token performance in real-time. This tool connects to the PumpPortal API, tracks newly created tokens, and provides detailed performance analysis at specified intervals.

## Features

- **Real-time monitoring** of newly created tokens via WebSocket connection
- **Configurable filtering criteria** for token analysis
- **Pattern detection** for pump and dump behaviors
- **Multi-interval analysis** (5, 15, 30 minutes by default, customizable)
- **Performance visualization** with auto-generated charts
- **Comprehensive reporting** with CSV exports and summary statistics

## Requirements

- Python 3.7+
- Required packages:
  - asyncio
  - websockets
  - pandas
  - matplotlib
  - json
  - datetime

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install asyncio websockets pandas matplotlib
   ```

## Usage

Run the backtester with default settings:

```python
python backtester.py
```

### Configuration Options

The backtester supports extensive configuration options that can be modified in the `main()` function:

```python
config = {
    # Token filtering criteria
    "min_initial_liquidity": 100,  # Minimum $100 initial liquidity
    "min_transactions": 5,         # At least 5 transactions
    "require_raydium": True,       # Require Raydium liquidity

    # Performance thresholds
    "min_price_change_pct": -50,   # Min price change percentage
    "max_price_change_pct": 5000,  # Max price change percentage
    "min_volume": 50,              # Minimum volume in USD

    # Pattern detection
    "detect_pump_dump": True,
    "pump_threshold": 50,          # % increase to consider a pump
    "dump_threshold": -30,         # % decrease to consider a dump

    # Analysis intervals
    "extended_intervals": [60]     # Also analyze at 60 minutes
}
```

## Output

The tool generates the following outputs in the `backtest_data` directory (configurable):

1. Separate folders for each analysis interval (5min, 15min, 30min, etc.)
2. Per-token CSV files with performance metrics
3. Per-token trade history CSV files
4. Price charts for each token at each analysis interval
5. Aggregated results in CSV format
6. Summary statistics for strategy development

## How It Works

1. **Connection**: Establishes WebSocket connection to PumpPortal API
2. **Monitoring**: Subscribes to new token creation events
3. **Data Collection**: Tracks trades for each token in real-time
4. **Filtering**: Applies configured criteria to focus on tokens of interest
5. **Pattern Detection**: Identifies pump and dump patterns
6. **Analysis**: Performs detailed analysis at specified time intervals
7. **Reporting**: Generates charts, CSV files, and summary statistics

## Analysis Metrics

The tool captures and analyzes the following metrics:

- Initial and current price
- Price change percentage
- Trading volume
- Transaction count
- Highest/lowest price
- Liquidity provider information
- Trading patterns
- And more...

## Extending the Tool

You can extend the tool by:

1. Adding new pattern detection algorithms in the `detect_patterns` method
2. Creating custom filtering criteria in the `apply_token_filters` method
3. Expanding the analysis metrics in the `analyze_token_performance` method
4. Modifying the chart generation in the `generate_price_chart` method

## License

[Specify your license here]
