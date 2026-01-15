"""
Kite Connect Data Fetcher
Downloads historical OHLCV data from Zerodha with rate limit handling
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)


class KiteDataFetcher:
    """
    Fetches historical candle data from Zerodha Kite Connect
    Implements rate limiting (3 req/sec guideline) and retry logic
    """

    def __init__(self, api_key: str, access_token: str, rate_limit_delay: float = 0.35):
        """
        Initialize Kite Connect client

        Args:
            api_key: Zerodha API key
            access_token: Valid access token from auth flow
            rate_limit_delay: Delay between requests in seconds (default 0.35s = ~3 req/sec)
        """
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def fetch_historical(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "15minute",
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch historical candles for a single instrument

        Args:
            instrument_token: Instrument token from Kite
            from_date: Start date
            to_date: End date
            interval: Candle interval (minute, 3minute, 5minute, 15minute, day, etc.)
            max_retries: Maximum retry attempts

        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()

        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {interval} data for {instrument_token} from {from_date} to {to_date}")
                data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval
                )

                df = pd.DataFrame(data)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

                logger.info(f"Fetched {len(df)} candles")
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(f"Failed to fetch data after {max_retries} attempts")
                    raise

        return pd.DataFrame()

    def fetch_multiple_symbols(
        self,
        symbols: List[Dict[str, any]],
        from_date: datetime,
        to_date: datetime,
        interval: str = "15minute"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols with rate limiting

        Args:
            symbols: List of dicts with 'symbol' and 'instrument_token'
            from_date: Start date
            to_date: End date
            interval: Candle interval

        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}

        for symbol_info in symbols:
            symbol = symbol_info['symbol']
            token = symbol_info['instrument_token']

            try:
                df = self.fetch_historical(token, from_date, to_date, interval)
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """
        Get list of available instruments

        Args:
            exchange: Exchange name (NSE, BSE, NFO, etc.)

        Returns:
            DataFrame of instruments
        """
        self._rate_limit()
        instruments = self.kite.instruments(exchange)
        return pd.DataFrame(instruments)
