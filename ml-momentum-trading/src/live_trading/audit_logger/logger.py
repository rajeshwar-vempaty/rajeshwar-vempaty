"""
Audit Logger
Structured logging for compliance and debugging
Maintains detailed records as required by Kite Connect terms
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Maintains comprehensive audit trail for trading activity
    """

    def __init__(self, log_dir: str = "logs/audit"):
        """
        Initialize audit logger

        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Separate log files for different event types
        self.signal_log = self.log_dir / "signals.jsonl"
        self.order_log = self.log_dir / "orders.jsonl"
        self.position_log = self.log_dir / "positions.jsonl"
        self.risk_log = self.log_dir / "risk_events.jsonl"
        self.error_log = self.log_dir / "errors.jsonl"

    def _write_log(self, log_file: Path, event: Dict[str, Any]):
        """
        Write event to log file (JSONL format)

        Args:
            log_file: Path to log file
            event: Event dict
        """
        event['timestamp'] = event.get('timestamp', datetime.now().isoformat())

        with open(log_file, 'a') as f:
            f.write(json.dumps(event, default=str) + '\n')

    def log_signal(
        self,
        symbol: str,
        signal: Dict,
        features: Dict,
        market_data: Dict
    ):
        """
        Log signal generation event

        Args:
            symbol: Trading symbol
            signal: Signal dict from SignalGenerator
            features: Feature values used
            market_data: Current market data snapshot
        """
        event = {
            'event_type': 'signal',
            'symbol': symbol,
            'signal': signal['signal'],
            'probability': signal['probability'],
            'confidence': signal['confidence'],
            'price': signal.get('price'),
            'features': features,
            'market_data': market_data
        }

        self._write_log(self.signal_log, event)
        logger.debug(f"Logged signal for {symbol}")

    def log_order(
        self,
        symbol: str,
        order_type: str,
        order_params: Dict,
        order_response: Dict,
        signal: Dict
    ):
        """
        Log order placement

        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            order_params: Order parameters
            order_response: Response from broker API
            signal: Associated signal
        """
        event = {
            'event_type': 'order',
            'symbol': symbol,
            'order_type': order_type,
            'order_params': order_params,
            'order_response': order_response,
            'signal_probability': signal.get('probability'),
            'signal_confidence': signal.get('confidence')
        }

        self._write_log(self.order_log, event)
        logger.info(f"Logged order for {symbol}: {order_type}")

    def log_fill(
        self,
        symbol: str,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
        intended_price: float,
        slippage_bps: float
    ):
        """
        Log order fill

        Args:
            symbol: Trading symbol
            order_id: Order ID
            fill_price: Actual fill price
            fill_quantity: Filled quantity
            intended_price: Expected price
            slippage_bps: Slippage in bps
        """
        event = {
            'event_type': 'fill',
            'symbol': symbol,
            'order_id': order_id,
            'fill_price': fill_price,
            'fill_quantity': fill_quantity,
            'intended_price': intended_price,
            'slippage_bps': slippage_bps
        }

        self._write_log(self.order_log, event)
        logger.info(f"Logged fill for {symbol}: {fill_quantity} @ {fill_price}")

    def log_position_update(
        self,
        symbol: str,
        action: str,
        position_data: Dict
    ):
        """
        Log position update

        Args:
            symbol: Trading symbol
            action: "OPEN", "UPDATE", or "CLOSE"
            position_data: Position details
        """
        event = {
            'event_type': 'position',
            'symbol': symbol,
            'action': action,
            'position_data': position_data
        }

        self._write_log(self.position_log, event)
        logger.debug(f"Logged position update for {symbol}: {action}")

    def log_risk_event(
        self,
        event_type: str,
        description: str,
        data: Dict
    ):
        """
        Log risk management event

        Args:
            event_type: "STOP_LOSS", "TAKE_PROFIT", "MAX_LOSS", "EMERGENCY_EXIT", etc.
            description: Human-readable description
            data: Event data
        """
        event = {
            'event_type': 'risk',
            'risk_event_type': event_type,
            'description': description,
            'data': data
        }

        self._write_log(self.risk_log, event)
        logger.warning(f"Logged risk event: {event_type} - {description}")

    def log_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        context: Dict
    ):
        """
        Log error event

        Args:
            component: Component where error occurred
            error_type: Error type/category
            error_message: Error message
            context: Additional context
        """
        event = {
            'event_type': 'error',
            'component': component,
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }

        self._write_log(self.error_log, event)
        logger.error(f"Logged error: {component} - {error_type}")

    def load_logs(self, log_type: str, start_date: datetime = None, end_date: datetime = None) -> list:
        """
        Load and filter logs

        Args:
            log_type: "signals", "orders", "positions", "risk", or "errors"
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of log events
        """
        log_file_map = {
            'signals': self.signal_log,
            'orders': self.order_log,
            'positions': self.position_log,
            'risk': self.risk_log,
            'errors': self.error_log
        }

        log_file = log_file_map.get(log_type)
        if not log_file or not log_file.exists():
            return []

        events = []
        with open(log_file, 'r') as f:
            for line in f:
                event = json.loads(line)
                event_time = datetime.fromisoformat(event['timestamp'])

                # Filter by date range
                if start_date and event_time < start_date:
                    continue
                if end_date and event_time > end_date:
                    continue

                events.append(event)

        return events
