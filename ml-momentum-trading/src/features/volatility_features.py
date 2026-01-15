"""
Volatility and range-based features
ATR, Parkinson/Garman-Klass estimators, RSI, ADX-style signals
"""

import numpy as np
import pandas as pd
from typing import List


class VolatilityFeatures:
    """
    Generates volatility and range-based features
    """

    @staticmethod
    def rolling_volatility(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Rolling standard deviation of returns

        Args:
            df: DataFrame with 'close' column
            windows: List of window sizes

        Returns:
            DataFrame with volatility features
        """
        features = df.copy()
        returns = df['close'].pct_change()

        for window in windows:
            features[f'volatility_{window}'] = (
                returns.rolling(window).std() * np.sqrt(252)  # Annualized
            )

        return features

    @staticmethod
    def parkinson_volatility(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Parkinson volatility estimator (uses high-low range)
        More efficient than close-to-close volatility

        Args:
            df: DataFrame with 'high' and 'low' columns
            windows: List of window sizes

        Returns:
            DataFrame with Parkinson volatility features
        """
        features = df.copy()

        # Parkinson estimator: sqrt(1/(4*ln(2)) * (ln(H/L))^2)
        hl_ratio = np.log(df['high'] / df['low'])

        for window in windows:
            features[f'parkinson_vol_{window}'] = (
                np.sqrt(
                    (1 / (4 * np.log(2))) *
                    (hl_ratio ** 2).rolling(window).mean()
                ) * np.sqrt(252)
            )

        return features

    @staticmethod
    def garman_klass_volatility(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Garman-Klass volatility estimator (uses OHLC)

        Args:
            df: DataFrame with OHLC data
            windows: List of window sizes

        Returns:
            DataFrame with Garman-Klass volatility features
        """
        features = df.copy()

        # GK estimator components
        hl_ratio = np.log(df['high'] / df['low'])
        co_ratio = np.log(df['close'] / df['open'])

        gk_component = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)

        for window in windows:
            features[f'gk_vol_{window}'] = (
                np.sqrt(gk_component.rolling(window).mean()) * np.sqrt(252)
            )

        return features

    @staticmethod
    def atr(df: pd.DataFrame, windows: List[int] = [14, 20]) -> pd.DataFrame:
        """
        Average True Range (ATR)

        Args:
            df: DataFrame with OHLC data
            windows: List of window sizes

        Returns:
            DataFrame with ATR features
        """
        features = df.copy()

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift(1))
        lc = abs(df['low'] - df['close'].shift(1))

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        for window in windows:
            features[f'atr_{window}'] = tr.rolling(window).mean()

            # Normalized ATR (as % of price)
            features[f'atr_pct_{window}'] = (
                features[f'atr_{window}'] / df['close']
            )

        return features

    @staticmethod
    def rsi(df: pd.DataFrame, periods: List[int] = [14, 20]) -> pd.DataFrame:
        """
        Relative Strength Index (RSI)

        Args:
            df: DataFrame with 'close' column
            periods: List of RSI periods

        Returns:
            DataFrame with RSI features
        """
        features = df.copy()
        delta = df['close'].diff()

        for period in periods:
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            rs = gain / (loss + 1e-8)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return features

    @staticmethod
    def adx_style_features(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        ADX-style directional indicators

        Args:
            df: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with directional features
        """
        features = df.copy()

        # Directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # True Range
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        # Directional Indicators
        atr = tr.rolling(period).mean()
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)

        features[f'pos_di_{period}'] = pos_di
        features[f'neg_di_{period}'] = neg_di
        features[f'di_diff_{period}'] = pos_di - neg_di

        return features
