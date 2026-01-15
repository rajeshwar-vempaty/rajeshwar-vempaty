# Installation Guide

Detailed installation instructions for the ML Momentum Trading System.

## System Requirements

### Hardware
- **CPU:** Multi-core processor (4+ cores recommended for training)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 10 GB free space (for data caching)

### Software
- **OS:** Linux, macOS, or Windows
- **Python:** 3.9, 3.10, or 3.11
- **Git:** For version control

---

## Step-by-Step Installation

### 1. Install Python

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

**macOS:**
```bash
brew install python@3.9
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and install.

### 2. Clone Repository

```bash
git clone https://github.com/rajeshwar-vempaty/ml-momentum-trading.git
cd ml-momentum-trading
```

### 3. Create Virtual Environment

```bash
# Create venv
python3.9 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 5. Verify Installation

```bash
python -c "import lightgbm; import pandas; import kiteconnect; print('All imports successful!')"
```

---

## Zerodha Kite Connect Setup

### 1. Create Kite Connect App

1. Go to [Kite Developers Portal](https://developers.kite.trade/)
2. Sign up / Log in with your Zerodha account
3. Create a new app:
   - **App name:** ML Momentum Trading
   - **App type:** Connect
   - **Redirect URL:** `http://127.0.0.1` (for local testing)

4. Note your **API Key** and **API Secret**

### 2. Generate Access Token

Kite Connect uses OAuth2 flow. Access tokens expire daily and must be regenerated.

**Method 1: Manual Login Flow (Recommended for Testing)**

```python
from kiteconnect import KiteConnect

api_key = "your_api_key"
api_secret = "your_api_secret"

kite = KiteConnect(api_key=api_key)

# Get login URL
login_url = kite.login_url()
print(f"Login URL: {login_url}")

# Open URL in browser, login, copy request_token from redirect URL
request_token = "paste_request_token_here"

# Generate access token
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

print(f"Access Token: {access_token}")
```

**Method 2: Automated (for Production)**

For production, you'll need to automate the login flow using Selenium or similar.

### 3. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env  # or use your preferred editor
```

Add your credentials:
```
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=your_access_token_here
```

---

## Configuration

### 1. Review Config File

```bash
nano config/config.yaml
```

Key settings to verify:

```yaml
universe:
  symbols:  # List your trading symbols
    - "INFY"
    - "TCS"
    # Add more...

data:
  interval: "15minute"  # 5minute, 15minute, day

risk:
  max_daily_loss: 5000  # Adjust to your risk tolerance
  max_positions: 3
```

### 2. Create Data Directories

```bash
mkdir -p data/raw data/processed data/cache
mkdir -p logs/audit
mkdir -p model_registry
```

---

## Testing Installation

### 1. Test Data Fetching

```python
# test_connection.py
from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

load_dotenv()

kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

# Test quote fetching
quote = kite.quote(["NSE:INFY"])
print(f"INFY LTP: {quote['NSE:INFY']['last_price']}")

# Test instruments
instruments = kite.instruments("NSE")
print(f"Found {len(instruments)} NSE instruments")
```

### 2. Run Unit Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

### Issue: ImportError for kiteconnect

**Solution:**
```bash
pip install --upgrade kiteconnect
```

### Issue: Access token expired

**Solution:**
Access tokens expire daily. Regenerate using the login flow.

### Issue: Rate limit errors

**Solution:**
- Check `rate_limit_delay` in config
- Ensure you're not exceeding 3 requests/second for data

### Issue: Missing LightGBM

**Solution (Linux):**
```bash
sudo apt install cmake
pip install lightgbm --install-option=--mpi
```

**Solution (Windows):**
Install Visual C++ Build Tools, then:
```bash
pip install lightgbm
```

---

## Next Steps

1. ✅ Installation complete
2. 📚 Read [Usage Guide](USAGE.md)
3. 🎯 Run training script: `python scripts/train_model.py`
4. 🧪 Test in shadow mode: `python scripts/live_trading.py`

---

## Getting Help

- 📖 Documentation: [docs/](../docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/rajeshwar-vempaty/ml-momentum-trading/issues)
- 📧 Email: vrsanurag@gmail.com
