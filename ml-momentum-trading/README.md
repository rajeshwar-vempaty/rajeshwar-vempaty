# ML Momentum Trading System

End-to-end machine learning system for momentum-based trading with Zerodha Kite Connect integration.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This project implements a production-grade ML trading system that:

- **Predicts** next-period returns using momentum and market microstructure features
- **Backtests** strategies with realistic transaction costs and slippage
- **Executes** trades via Zerodha Kite Connect with proper risk controls
- **Monitors** performance, slippage, and model drift in real-time
- **Logs** all decisions and events for compliance and debugging

### Key Features

✅ **Walk-forward cross-validation** with purging and embargo to prevent leakage
✅ **LightGBM/XGBoost** models with probability calibration
✅ **Realistic backtesting** including brokerage, STT, slippage, and liquidity constraints
✅ **Live trading** with shadow mode for safe testing
✅ **Risk management** with stop-loss, position limits, and drawdown controls
✅ **Comprehensive logging** for audit compliance
✅ **Real-time monitoring** with drift detection and performance alerts

---

## Architecture

```
ml-momentum-trading/
├── src/
│   ├── data_ingest/          # Historical data download & caching
│   ├── features/              # Momentum, volatility, market features
│   ├── training/              # Walk-forward CV, model training, calibration
│   ├── backtest/              # Vectorized backtester with costs
│   ├── model_registry/        # Versioned model storage
│   └── live_trading/
│       ├── market_data_service/    # Real-time quotes & ticks
│       ├── signal_service/         # ML inference
│       ├── risk_manager/           # Position sizing & limits
│       ├── execution_service/      # Order placement with rate limits
│       ├── monitoring/             # Performance & drift tracking
│       └── audit_logger/           # Compliance logging
├── config/                    # YAML configuration
├── data/                      # Raw, processed, cached data
├── notebooks/                 # Research notebooks
├── scripts/                   # Training & live trading scripts
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- Zerodha Kite Connect account with API access
- Capital for trading (recommended: start with small amounts in shadow mode)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/rajeshwar-vempaty/ml-momentum-trading.git
cd ml-momentum-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# KITE_API_KEY=your_api_key
# KITE_API_SECRET=your_secret
# KITE_ACCESS_TOKEN=your_access_token

# Review and customize config/config.yaml
```

### 4. Get Zerodha Access Token

Kite Connect uses OAuth2 flow. You need to:

1. Create an app at https://developers.kite.trade/
2. Get API key and secret
3. Generate access token using login flow (see [Kite Connect docs](https://kite.trade/docs/connect/v3/))

**Important:** Access tokens expire daily and must be regenerated.

---

## Usage

### Training a Model

```bash
# Run training pipeline
python scripts/train_model.py

# This will:
# 1. Download historical data from Kite Connect
# 2. Generate momentum/volatility features
# 3. Train LightGBM model with walk-forward CV
# 4. Calibrate probabilities
# 5. Register model in model registry
```

### Backtesting

```python
from src.backtest.backtester import Backtester
from src.backtest.transaction_costs import TransactionCostModel

# Initialize backtester
cost_model = TransactionCostModel()
backtester = Backtester(
    initial_capital=100000,
    cost_model=cost_model,
    confidence_threshold=0.6
)

# Run backtest
results = backtester.run(data, predictions)

# Analyze performance
from src.backtest.performance_metrics import PerformanceMetrics
metrics = PerformanceMetrics.comprehensive_report(results['strategy_return_net'])

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"CAGR: {metrics['cagr']:.2%}")
```

### Live Trading

```bash
# Shadow mode (no real orders - recommended for testing)
# Set TRADING_MODE=SHADOW in .env
python scripts/live_trading.py

# Live mode (real orders - USE WITH CAUTION)
# Set TRADING_MODE=LIVE in .env
python scripts/live_trading.py
```

**Safety Features:**
- Shadow mode logs all signals without placing orders
- Position limits and daily loss caps
- Stop-loss and take-profit automation
- Emergency shutdown with `Ctrl+C`

---

## Trading Strategy

### Signal Generation

The system uses a **binary classification** approach:

1. **Target:** Predict if next 15-min return > 0%
2. **Features:** 80+ momentum, volatility, and market features
3. **Model:** LightGBM with isotonic calibration
4. **Threshold:** Only trade when P(up) > 0.6

### Risk Management

- **Position sizing:** Scaled by signal confidence
- **Stop-loss:** 2% from entry
- **Take-profit:** 3% from entry
- **Max positions:** 3 concurrent
- **Max daily loss:** ₹5,000
- **Max drawdown:** 10% from peak

---

## Features

### Momentum Features
- Log returns (1, 3, 5, 10, 20 periods)
- Moving average crossovers (5, 10, 20, 50)
- Breakout indicators (Donchian channels)
- Volume trends and OBV

### Volatility Features
- Rolling standard deviation
- Parkinson & Garman-Klass estimators
- ATR (Average True Range)
- RSI and ADX-style directional indicators

### Market Context
- NIFTY index returns and volatility
- Rolling beta vs. market
- Correlation with index

---

## Configuration

Edit `config/config.yaml` to customize:

- **Universe:** List of trading symbols
- **Data:** Bar interval (5min, 15min, daily)
- **Features:** Feature engineering parameters
- **Training:** Model hyperparameters, CV settings
- **Risk:** Position limits, stop-loss/take-profit
- **Execution:** Order types, rate limits

---

## Transaction Costs

The backtest includes realistic costs for Indian equities:

| Cost Component | Rate |
|----------------|------|
| Brokerage | 0.03% (or ₹20, whichever lower) |
| STT | 0.025% (sell side) |
| Exchange charges | ~0.00325% |
| GST | 18% on brokerage + exchange |
| Stamp duty | 0.015% (buy side) |
| Slippage | 5 bps + volatility adjustment |

---

## Monitoring & Alerts

The system tracks:

- **Real-time P&L** and hit rate
- **Slippage** (intended vs. actual fill prices)
- **Model drift** (signal distribution changes)
- **Fill rate** (signals executed / total signals)

Alerts are triggered for:
- Daily loss exceeding threshold
- High slippage (> 20 bps)
- Model drift detection
- Risk limit breaches

---

## Compliance & Auditing

All events are logged in JSONL format:

- `logs/audit/signals.jsonl` - Every signal generated
- `logs/audit/orders.jsonl` - Every order placed and filled
- `logs/audit/positions.jsonl` - Position updates
- `logs/audit/risk_events.jsonl` - Risk management actions
- `logs/audit/errors.jsonl` - System errors

This meets Zerodha's requirement for maintaining accurate logs subject to audit.

---

## Testing

```bash
# Run unit tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

---

## MVP Scope (Recommended Starting Point)

For a working MVP that demonstrates all components:

- **Universe:** 10-20 liquid NSE stocks
- **Horizon:** 15-minute bars
- **Model:** LightGBM classifier
- **Strategy:** Trade when P(up) > 0.60 and trend filter positive
- **Risk:** 1 position at a time, fixed stop-loss, max ₹2,000 loss/day
- **Deployment:** Shadow mode initially, then micro-sizing (₹10,000 capital)

---

## Roadmap

- [ ] Add support for F&O (futures & options)
- [ ] Multi-symbol portfolio optimization
- [ ] Advanced features (sentiment, order book)
- [ ] Reinforcement learning for position sizing
- [ ] Web dashboard for monitoring
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)

---

## Disclaimer

⚠️ **IMPORTANT RISK DISCLOSURE**

This software is provided for educational and research purposes only.

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Use at your own risk - no warranties provided
- Start with shadow mode and paper trading
- Only trade with capital you can afford to lose
- Consult a financial advisor before live trading

The authors are not responsible for any financial losses incurred using this system.

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Author

**Rajeshwar Vempaty (Anurag)**
- 📧 Email: vrsanurag@gmail.com
- 🔗 LinkedIn: [rajeshwar-vempaty](https://www.linkedin.com/in/rajeshwar-vempaty-5b9a1734/)
- 🎓 MS Data Science, Fordham University

---

## Acknowledgments

- Zerodha for Kite Connect API
- LightGBM and scikit-learn communities
- Quantitative finance research community

---

## Support

For questions and issues:
- 📖 Read the [documentation](docs/)
- 🐛 Report bugs via [GitHub Issues](https://github.com/rajeshwar-vempaty/ml-momentum-trading/issues)
- 💬 Discussions welcome

---

**Built with ❤️ for the quant community**
