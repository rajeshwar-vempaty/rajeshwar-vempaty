"""
Order Executor
Places orders via Kite Connect with rate limit handling and retry logic
"""

import time
import logging
from typing import Dict, Optional, List
from datetime import datetime
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles order placement and monitoring with Zerodha Kite Connect
    Implements rate limiting (10 orders/sec, 200 orders/min, 3000 orders/day)
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        rate_limit_delay: float = 0.15,  # ~6-7 orders/sec (conservative)
        max_retries: int = 3
    ):
        """
        Initialize order executor

        Args:
            api_key: Zerodha API key
            access_token: Valid access token
            rate_limit_delay: Delay between orders in seconds
            max_retries: Maximum retry attempts for failed orders
        """
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        self.last_order_time = 0
        self.orders_today = 0
        self.order_history = []

        logger.info("OrderExecutor initialized")

    def _rate_limit(self):
        """Enforce rate limits"""
        elapsed = time.time() - self.last_order_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_order_time = time.time()

    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MARKET",
        product: str = "MIS",  # MIS (intraday) or CNC (delivery)
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        tag: str = "ml_momentum"
    ) -> Optional[Dict]:
        """
        Place order with retry logic

        Args:
            symbol: Trading symbol (e.g., "INFY")
            exchange: Exchange (e.g., "NSE")
            transaction_type: "BUY" or "SELL"
            quantity: Order quantity
            order_type: "MARKET", "LIMIT", "SL", "SL-M"
            product: "MIS" (intraday) or "CNC" (delivery)
            price: Limit price (for LIMIT orders)
            trigger_price: Trigger price (for SL/SL-M orders)
            tag: Order tag for tracking

        Returns:
            Order response dict or None if failed
        """
        self._rate_limit()

        order_params = {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "product": product,
            "tag": tag
        }

        if price:
            order_params["price"] = price

        if trigger_price:
            order_params["trigger_price"] = trigger_price

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Placing order (attempt {attempt + 1}/{self.max_retries}): "
                    f"{transaction_type} {quantity} {symbol} @ {order_type}"
                )

                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    **order_params
                )

                # Log order
                order_record = {
                    'order_id': order_id,
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'transaction_type': transaction_type,
                    'quantity': quantity,
                    'order_type': order_type,
                    'status': 'PLACED'
                }
                self.order_history.append(order_record)
                self.orders_today += 1

                logger.info(f"Order placed successfully: ID={order_id}")

                return {
                    'order_id': order_id,
                    'status': 'PLACED',
                    **order_params
                }

            except Exception as e:
                logger.error(f"Order placement failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(f"Order failed after {self.max_retries} attempts")
                    return None

        return None

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get order status

        Args:
            order_id: Order ID

        Returns:
            Order status dict
        """
        try:
            orders = self.kite.orders()
            for order in orders:
                if order['order_id'] == order_id:
                    return order
            return None
        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None

    def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        """
        Cancel pending order

        Args:
            order_id: Order ID
            variety: Order variety

        Returns:
            True if successful
        """
        try:
            self.kite.cancel_order(variety=variety, order_id=order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_positions(self) -> Dict:
        """
        Get current positions

        Returns:
            Positions dict
        """
        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {"net": [], "day": []}

    def close_all_positions(self, product: str = "MIS") -> List[Dict]:
        """
        Close all open positions (emergency exit)

        Args:
            product: "MIS" or "CNC"

        Returns:
            List of order responses
        """
        logger.warning("Closing all positions (emergency exit)")

        positions = self.get_positions()
        net_positions = positions.get("net", [])

        results = []

        for position in net_positions:
            if position['quantity'] == 0:
                continue

            symbol = position['tradingsymbol']
            exchange = position['exchange']
            quantity = abs(position['quantity'])

            # Determine transaction type (opposite of current position)
            transaction_type = "SELL" if position['quantity'] > 0 else "BUY"

            result = self.place_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type="MARKET",
                product=product,
                tag="emergency_exit"
            )

            results.append(result)

        return results

    def get_order_history(self) -> List[Dict]:
        """Get local order history"""
        return self.order_history

    def get_trades_today(self) -> int:
        """Get number of orders placed today"""
        return self.orders_today
