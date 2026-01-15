"""
Performance Monitor
Tracks live trading metrics, slippage, and model drift
"""

import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors live trading performance and detects issues
    """

    def __init__(
        self,
        alert_threshold_pnl: float = -5000,
        alert_threshold_slippage_bps: float = 20,
        drift_window: int = 100
    ):
        """
        Initialize performance monitor

        Args:
            alert_threshold_pnl: Alert if daily P&L drops below this
            alert_threshold_slippage_bps: Alert if slippage exceeds this (bps)
            drift_window: Window size for drift detection
        """
        self.alert_threshold_pnl = alert_threshold_pnl
        self.alert_threshold_slippage_bps = alert_threshold_slippage_bps
        self.drift_window = drift_window

        self.trades = []
        self.signals = []
        self.fill_records = []

    def record_signal(self, signal: Dict):
        """
        Record generated signal

        Args:
            signal: Signal dict from SignalGenerator
        """
        self.signals.append({
            'timestamp': signal.get('timestamp', datetime.now()),
            'signal': signal['signal'],
            'probability': signal['probability'],
            'confidence': signal['confidence'],
            'price': signal.get('price')
        })

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        intended_price: float,
        fill_price: float,
        fill_time: datetime,
        signal: Dict
    ):
        """
        Record executed trade

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Trade quantity
            intended_price: Expected price (from signal)
            fill_price: Actual fill price
            fill_time: Fill timestamp
            signal: Original signal
        """
        slippage_bps = abs((fill_price - intended_price) / intended_price) * 10000

        trade_record = {
            'timestamp': fill_time,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'intended_price': intended_price,
            'fill_price': fill_price,
            'slippage_bps': slippage_bps,
            'signal_probability': signal.get('probability'),
            'signal_confidence': signal.get('confidence')
        }

        self.trades.append(trade_record)
        self.fill_records.append(trade_record)

        # Check slippage alert
        if slippage_bps > self.alert_threshold_slippage_bps:
            logger.warning(
                f"HIGH SLIPPAGE ALERT: {symbol} - {slippage_bps:.2f} bps "
                f"(intended: {intended_price:.2f}, fill: {fill_price:.2f})"
            )

    def record_pnl(self, pnl: float, symbol: str):
        """
        Record trade P&L

        Args:
            pnl: Realized P&L
            symbol: Trading symbol
        """
        if len(self.trades) > 0:
            # Attach to most recent trade for this symbol
            for trade in reversed(self.trades):
                if trade['symbol'] == symbol and 'pnl' not in trade:
                    trade['pnl'] = pnl
                    break

    def get_daily_summary(self) -> Dict:
        """
        Get daily performance summary

        Returns:
            Summary dict with key metrics
        """
        today = datetime.now().date()
        today_trades = [
            t for t in self.trades
            if t['timestamp'].date() == today
        ]

        if not today_trades:
            return {
                'date': today,
                'num_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_slippage_bps': 0
            }

        df = pd.DataFrame(today_trades)

        total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
        win_rate = (df['pnl'] > 0).mean() if 'pnl' in df.columns else 0
        avg_slippage = df['slippage_bps'].mean()

        summary = {
            'date': today,
            'num_trades': len(today_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_slippage_bps': avg_slippage,
            'max_slippage_bps': df['slippage_bps'].max(),
            'trades': today_trades
        }

        # Check alerts
        if total_pnl < self.alert_threshold_pnl:
            logger.warning(f"DAILY P&L ALERT: {total_pnl:.2f} < {self.alert_threshold_pnl:.2f}")

        return summary

    def detect_model_drift(self) -> Dict:
        """
        Detect potential model drift by comparing recent performance

        Returns:
            Drift analysis dict
        """
        if len(self.signals) < self.drift_window * 2:
            return {'drift_detected': False, 'message': 'Insufficient data'}

        recent_signals = self.signals[-self.drift_window:]
        historical_signals = self.signals[-self.drift_window*2:-self.drift_window]

        df_recent = pd.DataFrame(recent_signals)
        df_historical = pd.DataFrame(historical_signals)

        # Compare signal distributions
        recent_mean_prob = df_recent['probability'].mean()
        historical_mean_prob = df_historical['probability'].mean()

        recent_signal_rate = df_recent['signal'].mean()
        historical_signal_rate = df_historical['signal'].mean()

        # Simple drift detection: significant shift in mean probability or signal rate
        prob_shift = abs(recent_mean_prob - historical_mean_prob)
        signal_rate_shift = abs(recent_signal_rate - historical_signal_rate)

        drift_detected = prob_shift > 0.1 or signal_rate_shift > 0.2

        result = {
            'drift_detected': drift_detected,
            'recent_mean_prob': recent_mean_prob,
            'historical_mean_prob': historical_mean_prob,
            'prob_shift': prob_shift,
            'recent_signal_rate': recent_signal_rate,
            'historical_signal_rate': historical_signal_rate,
            'signal_rate_shift': signal_rate_shift
        }

        if drift_detected:
            logger.warning(f"MODEL DRIFT DETECTED: {result}")

        return result

    def get_slippage_analysis(self) -> Dict:
        """
        Analyze slippage patterns

        Returns:
            Slippage analysis dict
        """
        if not self.fill_records:
            return {}

        df = pd.DataFrame(self.fill_records)

        return {
            'mean_slippage_bps': df['slippage_bps'].mean(),
            'median_slippage_bps': df['slippage_bps'].median(),
            'max_slippage_bps': df['slippage_bps'].max(),
            'std_slippage_bps': df['slippage_bps'].std(),
            'total_fills': len(self.fill_records)
        }

    def get_fill_rate(self) -> float:
        """
        Calculate fill rate (fills / signals)

        Returns:
            Fill rate (0-1)
        """
        num_signals = sum(1 for s in self.signals if s['signal'] == 1)
        num_fills = len(self.fill_records)

        return num_fills / num_signals if num_signals > 0 else 0.0
