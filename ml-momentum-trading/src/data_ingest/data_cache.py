"""
Data Cache Manager
Handles caching of downloaded historical data to avoid redundant API calls
"""

import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    Manages local cache of historical market data
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, interval: str, from_date: str, to_date: str) -> Path:
        """Generate cache file path"""
        filename = f"{symbol}_{interval}_{from_date}_{to_date}.parquet"
        return self.cache_dir / filename

    def get(
        self,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if available

        Args:
            symbol: Trading symbol
            interval: Candle interval
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            Cached DataFrame or None
        """
        cache_path = self._get_cache_path(symbol, interval, from_date, to_date)

        if cache_path.exists():
            logger.info(f"Cache hit: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Cache miss: {cache_path}")
        return None

    def put(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str
    ):
        """
        Store data in cache

        Args:
            df: DataFrame to cache
            symbol: Trading symbol
            interval: Candle interval
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        """
        if df.empty:
            logger.warning(f"Skipping cache for empty DataFrame: {symbol}")
            return

        cache_path = self._get_cache_path(symbol, interval, from_date, to_date)
        df.to_parquet(cache_path, compression='gzip')
        logger.info(f"Cached {len(df)} rows to {cache_path}")

    def clear(self):
        """Clear all cached data"""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
        logger.info(f"Cleared cache directory: {self.cache_dir}")
