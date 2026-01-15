"""
Risk Manager
Position sizing, stop-loss, max drawdown, and exposure limits
"""

import logging
from typing import Dict, Optional
from datetime import datetime, date

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces risk limits for live trading
    """

    def __init__(
        self,
        max_position_size: float = 100000,
        max_positions: int = 3,
        max_loss_per_trade: float = 2000,
        max_daily_loss: float = 5000,
        max_drawdown_pct: float = 0.10,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03
    ):
        """
        Initialize risk manager

        Args:
            max_position_size: Maximum position value
            max_positions: Maximum concurrent positions
            max_loss_per_trade: Maximum loss per trade
            max_daily_loss: Maximum loss per day
            max_drawdown_pct: Maximum drawdown from peak (as fraction)
            stop_loss_pct: Stop-loss percentage
            take_profit_pct: Take-profit percentage
        """
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.max_loss_per_trade = max_loss_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # State tracking
        self.current_positions = {}  # symbol -> position dict
        self.daily_pnl = 0.0
        self.peak_capital = 0.0
        self.current_capital = 0.0
        self.today = date.today()

        logger.info(f"Risk Manager initialized: max_positions={max_positions}, "
                   f"max_daily_loss={max_daily_loss}")

    def check_trade_allowed(
        self,
        symbol: str,
        signal: Dict,
        current_price: float
    ) -> tuple[bool, str]:
        """
        Check if trade is allowed under risk constraints

        Args:
            symbol: Trading symbol
            signal: Signal dict from SignalGenerator
            current_price: Current market price

        Returns:
            (allowed: bool, reason: str)
        """
        # Check if already in position
        if symbol in self.current_positions:
            return False, f"Already in position for {symbol}"

        # Check max positions
        if len(self.current_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit ({self.max_daily_loss}) reached"

        # Check max drawdown
        if self.current_capital > 0 and self.peak_capital > 0:
            drawdown_pct = (self.peak_capital - self.current_capital) / self.peak_capital
            if drawdown_pct > self.max_drawdown_pct:
                return False, f"Max drawdown ({self.max_drawdown_pct*100}%) exceeded"

        # Check signal confidence
        if signal['signal'] == 0:
            return False, "Signal is 0 (no trade)"

        # All checks passed
        return True, "Trade allowed"

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        confidence: float,
        available_capital: float
    ) -> int:
        """
        Calculate position size based on risk parameters

        Args:
            symbol: Trading symbol
            price: Current price
            confidence: Signal confidence (0-1)
            available_capital: Available capital

        Returns:
            Position size (number of shares)
        """
        # Base position value (limited by max_position_size)
        base_position_value = min(self.max_position_size, available_capital * 0.3)

        # Scale by confidence
        position_value = base_position_value * confidence

        # Calculate quantity
        quantity = int(position_value / price)

        # Ensure at least 1 share (if affordable)
        if quantity == 0 and price < available_capital:
            quantity = 1

        logger.info(f"Position size for {symbol}: {quantity} shares @ {price} "
                   f"(confidence: {confidence:.2f})")

        return quantity

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        signal: Dict
    ):
        """
        Register new position

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position size
            signal: Signal dict
        """
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)

        self.current_positions[symbol] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'signal': signal,
            'unrealized_pnl': 0.0
        }

        logger.info(f"Added position: {symbol} - {quantity} @ {entry_price:.2f}, "
                   f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

    def update_position(self, symbol: str, current_price: float):
        """
        Update position with current price and check exit conditions

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            None or exit action dict if position should be closed
        """
        if symbol not in self.current_positions:
            return None

        position = self.current_positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']

        # Calculate unrealized P&L
        unrealized_pnl = (current_price - entry_price) * quantity
        position['unrealized_pnl'] = unrealized_pnl

        # Check stop-loss
        if current_price <= position['stop_loss']:
            logger.warning(f"Stop-loss hit for {symbol}: {current_price:.2f} <= {position['stop_loss']:.2f}")
            return {'action': 'exit', 'reason': 'stop_loss', 'price': current_price}

        # Check take-profit
        if current_price >= position['take_profit']:
            logger.info(f"Take-profit hit for {symbol}: {current_price:.2f} >= {position['take_profit']:.2f}")
            return {'action': 'exit', 'reason': 'take_profit', 'price': current_price}

        return None

    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close position and calculate realized P&L

        Args:
            symbol: Trading symbol
            exit_price: Exit price

        Returns:
            Realized P&L
        """
        if symbol not in self.current_positions:
            logger.warning(f"No position to close for {symbol}")
            return 0.0

        position = self.current_positions[symbol]
        pnl = (exit_price - position['entry_price']) * position['quantity']

        # Update daily P&L
        self.daily_pnl += pnl

        # Update capital
        self.current_capital += pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        logger.info(f"Closed position: {symbol} - P&L: {pnl:.2f}, Daily P&L: {self.daily_pnl:.2f}")

        # Remove position
        del self.current_positions[symbol]

        return pnl

    def reset_daily_pnl(self):
        """Reset daily P&L counter (call at start of new trading day)"""
        current_date = date.today()
        if current_date != self.today:
            logger.info(f"New trading day - resetting daily P&L (was: {self.daily_pnl:.2f})")
            self.daily_pnl = 0.0
            self.today = current_date

    def get_risk_summary(self) -> Dict:
        """Get current risk metrics"""
        return {
            'num_positions': len(self.current_positions),
            'daily_pnl': self.daily_pnl,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown_pct': (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0,
            'positions': self.current_positions
        }
