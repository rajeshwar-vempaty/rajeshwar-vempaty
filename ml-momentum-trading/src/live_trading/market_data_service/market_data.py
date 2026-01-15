"""
Market Data Service
Real-time quotes and streaming ticks from Kite Connect
"""

import time
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
import pandas as pd
from kiteconnect import KiteConnect, KiteTicker

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Manages real-time market data streaming from Zerodha
    """

    def __init__(self, api_key: str, access_token: str):
        """
        Initialize market data service

        Args:
            api_key: Zerodha API key
            access_token: Valid access token
        """
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

        self.kws = KiteTicker(api_key, access_token)
        self.subscribed_tokens = []
        self.latest_ticks = {}
        self.tick_callbacks = []

    def get_quote(self, instruments: List[str]) -> Dict:
        """
        Get current quotes (LTP, OHLC, volume)

        Args:
            instruments: List of instrument symbols (e.g., ["NSE:INFY", "NSE:TCS"])

        Returns:
            Dict with quote data
        """
        try:
            quotes = self.kite.quote(instruments)
            logger.debug(f"Fetched quotes for {len(instruments)} instruments")
            return quotes
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            return {}

    def get_ltp(self, instruments: List[str]) -> Dict[str, float]:
        """
        Get last traded price for instruments

        Args:
            instruments: List of instrument symbols

        Returns:
            Dict mapping symbol -> LTP
        """
        try:
            ltp_data = self.kite.ltp(instruments)
            return {k: v['last_price'] for k, v in ltp_data.items()}
        except Exception as e:
            logger.error(f"Error fetching LTP: {e}")
            return {}

    def subscribe_ticks(self, instrument_tokens: List[int], callback: Optional[Callable] = None):
        """
        Subscribe to live tick stream

        Args:
            instrument_tokens: List of instrument tokens
            callback: Optional callback function(ticks)
        """
        self.subscribed_tokens = instrument_tokens

        if callback:
            self.tick_callbacks.append(callback)

        # Set up WebSocket callbacks
        def on_ticks(ws, ticks):
            """Handle incoming ticks"""
            for tick in ticks:
                self.latest_ticks[tick['instrument_token']] = tick

            # Call registered callbacks
            for cb in self.tick_callbacks:
                try:
                    cb(ticks)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")

        def on_connect(ws, response):
            """On WebSocket connect"""
            logger.info("WebSocket connected")
            ws.subscribe(instrument_tokens)
            ws.set_mode(ws.MODE_FULL, instrument_tokens)  # Full mode for OHLC

        def on_close(ws, code, reason):
            """On WebSocket close"""
            logger.warning(f"WebSocket closed: {code} - {reason}")

        def on_error(ws, code, reason):
            """On WebSocket error"""
            logger.error(f"WebSocket error: {code} - {reason}")

        # Assign callbacks
        self.kws.on_ticks = on_ticks
        self.kws.on_connect = on_connect
        self.kws.on_close = on_close
        self.kws.on_error = on_error

    def start_streaming(self):
        """Start WebSocket tick stream (blocking)"""
        logger.info("Starting tick stream...")
        self.kws.connect(threaded=False)

    def start_streaming_threaded(self):
        """Start WebSocket tick stream (non-blocking)"""
        logger.info("Starting tick stream in background thread...")
        self.kws.connect(threaded=True)

    def stop_streaming(self):
        """Stop WebSocket stream"""
        logger.info("Stopping tick stream...")
        self.kws.close()

    def get_latest_tick(self, instrument_token: int) -> Optional[Dict]:
        """
        Get latest tick for an instrument

        Args:
            instrument_token: Instrument token

        Returns:
            Latest tick dict or None
        """
        return self.latest_ticks.get(instrument_token)

    def build_bar(self, instrument_token: int, interval_seconds: int = 900) -> Optional[Dict]:
        """
        Build OHLCV bar from ticks (simple aggregation)

        Args:
            instrument_token: Instrument token
            interval_seconds: Bar interval in seconds (default 900 = 15min)

        Returns:
            OHLCV dict or None
        """
        # This is a simplified version
        # In production, you'd maintain a proper bar builder with tick aggregation
        tick = self.get_latest_tick(instrument_token)

        if tick is None:
            return None

        # Use OHLC from tick (Kite provides OHLC in full mode)
        return {
            'open': tick.get('ohlc', {}).get('open'),
            'high': tick.get('ohlc', {}).get('high'),
            'low': tick.get('ohlc', {}).get('low'),
            'close': tick.get('last_price'),
            'volume': tick.get('volume'),
            'timestamp': tick.get('timestamp')
        }
