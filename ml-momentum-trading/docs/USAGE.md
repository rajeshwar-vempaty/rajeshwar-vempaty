# Usage Guide

Comprehensive guide for using the ML Momentum Trading System.

## Table of Contents

1. [Data Collection](#data-collection)
2. [Feature Engineering](#feature-engineering)
3. [Model Training](#model-training)
4. [Backtesting](#backtesting)
5. [Live Trading](#live-trading)
6. [Monitoring](#monitoring)

---

## Data Collection

### Fetching Historical Data

```python
from src.data_ingest.kite_data_fetcher import KiteDataFetcher
from src.data_ingest.data_cache import DataCache
from datetime import datetime, timedelta

# Initialize
api_key = "your_api_key"
access_token = "your_access_token"

fetcher = KiteDataFetcher(api_key, access_token)
cache = DataCache("data/cache")

# Get instruments
instruments = fetcher.get_instruments("NSE")
infy = instruments[instruments['tradingsymbol'] == 'INFY'].iloc[0]
instrument_token = infy['instrument_token']

# Fetch data
from_date = datetime.now() - timedelta(days=365)
to_date = datetime.now()

df = fetcher.fetch_historical(
    instrument_token=instrument_token,
    from_date=from_date,
    to_date=to_date,
    interval="15minute"
)

# Cache for reuse
cache.put(df, "INFY", "15minute", from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))

print(f"Fetched {len(df)} bars for INFY")
```

### Fetching Multiple Symbols

```python
symbols_info = [
    {'symbol': 'INFY', 'instrument_token': 408065},
    {'symbol': 'TCS', 'instrument_token': 2953217},
    # Add more...
]

data_dict = fetcher.fetch_multiple_symbols(
    symbols=symbols_info,
    from_date=from_date,
    to_date=to_date,
    interval="15minute"
)
```

---

## Feature Engineering

### Generating Features

```python
from src.features.feature_pipeline import FeaturePipeline

# Initialize pipeline
pipeline = FeaturePipeline()

# Generate features for single symbol
features = pipeline.generate_features(df)

# With index data for market features
index_df = ...  # Load NIFTY data
features = pipeline.generate_features(df, index_df)

print(f"Generated {len(features.columns)} features")
```

### Preparing Training Data

```python
# Complete pipeline: features + labels
data = pipeline.prepare_training_data(
    df=df,
    index_df=index_df,
    horizon=1,  # Predict 1 bar ahead
    threshold=0.0,  # Binary: return > 0%
    dropna=True
)

# Split features and labels
feature_cols = pipeline.get_feature_columns(data)
X = data[feature_cols]
y = data['label']

print(f"Training data: {len(X)} samples, {len(feature_cols)} features")
```

---

## Model Training

### Walk-Forward Cross-Validation

```python
from src.training.walk_forward_cv import WalkForwardCV
from src.training.model_trainer import ModelTrainer

# Initialize CV
cv = WalkForwardCV(
    train_period_days=180,
    test_period_days=30,
    embargo_days=5,
    min_train_samples=1000
)

# Walk-forward validation
results = []

for train_idx, test_idx in cv.split(data):
    # Get train/test data
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    # Train model
    trainer = ModelTrainer(model_type="lightgbm")
    metrics = trainer.train(X_train, y_train, X_test, y_test)

    # Evaluate
    results.append({
        'train_start': train_idx.min(),
        'train_end': train_idx.max(),
        'test_start': test_idx.min(),
        'test_end': test_idx.max(),
        **metrics
    })

# Aggregate results
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df[['test_start', 'val_accuracy', 'val_auc', 'val_hit_rate']])
```

### Training Final Model

```python
# Train on full dataset
trainer = ModelTrainer()
metrics = trainer.train(X, y)

# Get feature importance
top_features = trainer.get_top_features(n=20)
print(top_features)

# Save model
trainer.save("models/momentum_model.txt")
```

### Probability Calibration

```python
from src.training.calibration import ModelCalibrator

# Get predictions on validation set
y_pred_proba = trainer.predict(X_test)

# Calibrate
calibrator = ModelCalibrator(method="isotonic")
y_calibrated = calibrator.fit_transform(y_test, y_pred_proba)

# Evaluate calibration
cal_metrics = calibrator.evaluate_calibration(y_test, y_calibrated)
print(f"ECE: {cal_metrics['ece']:.4f}")
```

### Model Registry

```python
from src.model_registry.registry import ModelRegistry

registry = ModelRegistry()

# Register model
version = registry.register_model(
    model_path="models/momentum_model.txt",
    model_name="momentum_classifier",
    metrics=metrics,
    feature_schema=feature_cols.tolist(),
    hyperparameters=trainer.params,
    notes="LightGBM with 15min bars, 180-day train window"
)

print(f"Model registered: version {version}")

# Load model later
model_path = registry.get_model_path("momentum_classifier")
metadata = registry.get_metadata("momentum_classifier")
```

---

## Backtesting

### Simple Backtest

```python
from src.backtest.backtester import Backtester
from src.backtest.transaction_costs import TransactionCostModel
from src.backtest.performance_metrics import PerformanceMetrics

# Get predictions
predictions = trainer.predict(X_test)

# Initialize backtester
cost_model = TransactionCostModel()
backtester = Backtester(
    initial_capital=100000,
    cost_model=cost_model,
    confidence_threshold=0.6
)

# Run backtest
results = backtester.run(
    data=data.loc[test_idx],
    predictions=predictions
)

# Performance metrics
metrics = PerformanceMetrics.comprehensive_report(
    results['strategy_return_net'],
    results['trades'],
    periods_per_year=252 * 26  # ~26 15-min bars per day
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Total Trades: {metrics['total_trades']:.0f}")
```

### Multi-Symbol Portfolio Backtest

```python
# Generate predictions for each symbol
predictions_dict = {}
for symbol, df in data_dict.items():
    X_symbol = ...  # Generate features
    predictions_dict[symbol] = trainer.predict(X_symbol)

# Run portfolio backtest
portfolio_results = backtester.run_multiple_symbols(
    data_dict=data_dict,
    predictions_dict=predictions_dict,
    allocation="equal"
)

print(f"Portfolio Return: {portfolio_results['cumulative_return'].iloc[-1]:.2%}")
```

---

## Live Trading

### Shadow Mode (Recommended First)

Shadow mode logs all signals without placing real orders.

```python
# In .env
TRADING_MODE=SHADOW

# Run live trading script
python scripts/live_trading.py
```

This will:
- Connect to live market data
- Generate real-time signals
- Log all decisions (no orders placed)
- Track "virtual" P&L

### Going Live

**⚠️ Only after thorough testing in shadow mode!**

```python
# In .env
TRADING_MODE=LIVE

# Start with minimal capital
INITIAL_CAPITAL=10000
MAX_POSITIONS=1
MAX_DAILY_LOSS=500

# Run
python scripts/live_trading.py
```

### Manual Trading Example

```python
from src.live_trading.market_data_service.market_data import MarketDataService
from src.live_trading.signal_service.signal_generator import SignalGenerator
from src.live_trading.execution_service.executor import OrderExecutor

# Initialize
market_data = MarketDataService(api_key, access_token)
signal_gen = SignalGenerator(model_path, feature_pipeline)
executor = OrderExecutor(api_key, access_token)

# Get current data
quote = market_data.get_quote(["NSE:INFY"])
# ... fetch historical bars for features ...

# Generate signal
signal = signal_gen.generate_signal(historical_data)

if signal['signal'] == 1 and signal['confidence'] > 0.6:
    # Place order
    order = executor.place_order(
        symbol="INFY",
        exchange="NSE",
        transaction_type="BUY",
        quantity=10,
        order_type="MARKET",
        product="MIS"
    )
    print(f"Order placed: {order['order_id']}")
```

---

## Monitoring

### Real-Time Performance

```python
from src.live_trading.monitoring.monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Record signals
monitor.record_signal(signal)

# Record trades
monitor.record_trade(
    symbol="INFY",
    side="BUY",
    quantity=10,
    intended_price=1500.00,
    fill_price=1500.50,
    fill_time=datetime.now(),
    signal=signal
)

# Get daily summary
summary = monitor.get_daily_summary()
print(f"P&L: {summary['total_pnl']:.2f}")
print(f"Win Rate: {summary['win_rate']:.2%}")
print(f"Avg Slippage: {summary['avg_slippage_bps']:.2f} bps")

# Check for model drift
drift = monitor.detect_model_drift()
if drift['drift_detected']:
    print(f"⚠️ Model drift detected!")
```

### Audit Logs

```python
from src.live_trading.audit_logger.logger import AuditLogger

logger = AuditLogger()

# Log events
logger.log_signal(symbol, signal, features, market_data)
logger.log_order(symbol, "BUY", order_params, order_response, signal)
logger.log_fill(symbol, order_id, fill_price, quantity, intended_price, slippage)

# Load logs for analysis
signals = logger.load_logs(
    "signals",
    start_date=datetime.now() - timedelta(days=7)
)

print(f"Logged {len(signals)} signals in past week")
```

---

## Best Practices

### 1. Start Small
- Begin with shadow mode
- Test with 1-2 symbols
- Use minimal capital initially

### 2. Monitor Closely
- Check slippage daily
- Track model drift
- Review audit logs weekly

### 3. Risk Management
- Never exceed max_daily_loss
- Use stop-losses on all positions
- Close positions before market close (if MIS)

### 4. Model Maintenance
- Retrain monthly with new data
- Monitor feature importance drift
- Update calibration periodically

### 5. Compliance
- Maintain all audit logs
- Keep detailed records of model changes
- Document all manual interventions

---

## Next Steps

- 📊 [Performance Analysis](PERFORMANCE.md)
- 🔧 [Troubleshooting](TROUBLESHOOTING.md)
- 📈 [Advanced Strategies](ADVANCED.md)
