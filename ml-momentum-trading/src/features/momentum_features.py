"""
Momentum-based features
Returns, trends, breakouts, and moving average signals
"""

import numpy as np
import pandas as pd
from typing import List


class MomentumFeatures:
    """
    Generates momentum and trend-based features
    All features are causal (use only past information)
    """

    @staticmethod
    def log_returns(df: pd.DataFrame, periods: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate log returns over multiple periods

        Args:
            df: DataFrame with 'close' column
            periods: List of lookback periods

        Returns:
            DataFrame with return features
        """
        features = df.copy()

        for period in periods:
            features[f'return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        return features

    @staticmethod
    def rolling_cumulative_return(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Rolling cumulative returns

        Args:
            df: DataFrame with 'close' column
            windows: List of window sizes

        Returns:
            DataFrame with cumulative return features
        """
        features = df.copy()

        for window in windows:
            features[f'cum_return_{window}'] = (
                df['close'].pct_change().rolling(window).sum()
            )

        return features

    @staticmethod
    def moving_average_features(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Moving average crossovers and distance features

        Args:
            df: DataFrame with 'close' column
            windows: List of MA window sizes

        Returns:
            DataFrame with MA features
        """
        features = df.copy()

        # Calculate MAs
        for window in windows:
            features[f'ma_{window}'] = df['close'].rolling(window).mean()

        # Distance from MAs (normalized by price)
        for window in windows:
            features[f'dist_ma_{window}'] = (
                (df['close'] - features[f'ma_{window}']) / df['close']
            )

        # MA slopes (regression on MA)
        for window in windows:
            features[f'ma_slope_{window}'] = (
                features[f'ma_{window}'].diff(5) / 5
            )

        # Crossover signals (fast MA - slow MA)
        if len(windows) >= 2:
            for i in range(len(windows) - 1):
                fast = windows[i]
                slow = windows[i + 1]
                features[f'ma_cross_{fast}_{slow}'] = (
                    features[f'ma_{fast}'] - features[f'ma_{slow}']
                ) / df['close']

        return features

    @staticmethod
    def breakout_features(
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Breakout and Donchian channel features

        Args:
            df: DataFrame with OHLC data
            windows: List of lookback windows

        Returns:
            DataFrame with breakout features
        """
        features = df.copy()

        for window in windows:
            # Distance from rolling high/low
            rolling_high = df['high'].rolling(window).max()
            rolling_low = df['low'].rolling(window).min()

            features[f'dist_high_{window}'] = (
                (df['close'] - rolling_high) / df['close']
            )
            features[f'dist_low_{window}'] = (
                (df['close'] - rolling_low) / df['close']
            )

            # Donchian channel position (0 = at low, 1 = at high)
            features[f'donchian_pos_{window}'] = (
                (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
            )

        return features

    @staticmethod
    def volume_features(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Volume-based features

        Args:
            df: DataFrame with 'volume' column
            windows: List of window sizes

        Returns:
            DataFrame with volume features
        """
        features = df.copy()

        for window in windows:
            # Volume z-score
            vol_mean = df['volume'].rolling(window).mean()
            vol_std = df['volume'].rolling(window).std()
            features[f'volume_zscore_{window}'] = (
                (df['volume'] - vol_mean) / (vol_std + 1e-8)
            )

            # Volume trend
            features[f'volume_trend_{window}'] = (
                df['volume'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
                )
            )

        # On-Balance Volume (OBV)
        features['obv'] = (
            np.sign(df['close'].diff()) * df['volume']
        ).cumsum()

        # Normalized OBV
        features['obv_norm'] = (
            features['obv'] / features['obv'].rolling(20).mean()
        )

        return features
