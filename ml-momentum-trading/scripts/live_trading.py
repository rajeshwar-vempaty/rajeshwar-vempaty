"""
Live Trading Script
Runs live trading system with real-time data and execution
"""

import sys
import yaml
import logging
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live_trading.market_data_service.market_data import MarketDataService
from live_trading.signal_service.signal_generator import SignalGenerator
from live_trading.risk_manager.risk_manager import RiskManager
from live_trading.execution_service.executor import OrderExecutor
from live_trading.monitoring.monitor import PerformanceMonitor
from live_trading.audit_logger.logger import AuditLogger
from features.feature_pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTradingSystem:
    """Main live trading system orchestrator"""

    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load credentials
        from dotenv import load_dotenv
        import os
        load_dotenv()

        api_key = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        model_path = os.getenv("MODEL_PATH")

        # Initialize components
        logger.info("Initializing live trading system")

        self.market_data = MarketDataService(api_key, access_token)
        self.feature_pipeline = FeaturePipeline(self.config['features'])

        # Load model and create signal generator
        self.signal_generator = SignalGenerator(
            model_path=model_path,
            feature_pipeline=self.feature_pipeline,
            confidence_threshold=self.config['live_trading']['signal_confidence_threshold']
        )

        self.risk_manager = RiskManager(
            max_position_size=self.config['risk']['max_position_size'],
            max_positions=self.config['risk']['max_positions'],
            max_loss_per_trade=self.config['risk']['max_loss_per_trade'],
            max_daily_loss=self.config['risk']['max_daily_loss'],
            max_drawdown_pct=self.config['risk']['max_drawdown_pct'],
            stop_loss_pct=self.config['risk']['stop_loss_pct'],
            take_profit_pct=self.config['risk']['take_profit_pct']
        )

        self.executor = OrderExecutor(
            api_key=api_key,
            access_token=access_token,
            rate_limit_delay=self.config['live_trading']['rate_limit_delay'],
            max_retries=self.config['live_trading']['max_retries']
        )

        self.monitor = PerformanceMonitor(
            alert_threshold_pnl=self.config['monitoring']['alert_threshold_pnl'],
            alert_threshold_slippage_bps=self.config['monitoring']['alert_threshold_slippage_bps'],
            drift_window=self.config['monitoring']['drift_window']
        )

        self.audit_logger = AuditLogger(log_dir=self.config['logging']['log_dir'])

        self.trading_mode = self.config['live_trading']['mode']
        self.running = False

        logger.info(f"System initialized in {self.trading_mode} mode")

    def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting live trading system")

        try:
            while self.running:
                # Reset daily P&L at start of new day
                self.risk_manager.reset_daily_pnl()

                # Main trading logic
                self._trading_iteration()

                # Sleep for next iteration (adjust based on bar frequency)
                time.sleep(900)  # 15 minutes for 15-min bars

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.shutdown()

        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            self.audit_logger.log_error(
                component="LiveTradingSystem",
                error_type="FATAL",
                error_message=str(e),
                context={}
            )
            self.shutdown()

    def _trading_iteration(self):
        """Single iteration of trading logic"""
        logger.info("=== Trading Iteration ===")

        symbols = self.config['universe']['symbols']

        for symbol in symbols:
            try:
                # 1. Get market data
                quote = self.market_data.get_quote([f"NSE:{symbol}"])

                if not quote:
                    logger.warning(f"No quote data for {symbol}")
                    continue

                # 2. Get historical data for features (simplified - you'd maintain a rolling buffer)
                # For demo purposes, we'll skip this

                # 3. Generate signal
                # signal = self.signal_generator.generate_signal(historical_data)

                # 4. Check risk constraints
                # allowed, reason = self.risk_manager.check_trade_allowed(symbol, signal, current_price)

                # 5. Execute trade (if allowed and in LIVE mode)
                # if allowed and self.trading_mode == "LIVE":
                #     self._execute_trade(symbol, signal)

                # 6. Update positions
                # self._update_positions(symbol)

                logger.info(f"Processed {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.audit_logger.log_error(
                    component="TradingIteration",
                    error_type="PROCESSING_ERROR",
                    error_message=str(e),
                    context={"symbol": symbol}
                )

        # 7. Check model drift
        drift_analysis = self.monitor.detect_model_drift()
        if drift_analysis.get('drift_detected'):
            logger.warning("Model drift detected - consider retraining")

        # 8. Generate daily summary
        summary = self.monitor.get_daily_summary()
        logger.info(f"Daily Summary: {summary}")

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading system")
        self.running = False

        # Close all positions if in LIVE mode
        if self.trading_mode == "LIVE":
            logger.warning("Closing all open positions")
            self.executor.close_all_positions()

        logger.info("Shutdown complete")


def main():
    system = LiveTradingSystem()
    system.run()


if __name__ == "__main__":
    main()
