"""
Market context features
Index returns, sector proxies, market regime indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class MarketFeatures:
    """
    Generates market-wide context features
    """

    @staticmethod
    def index_features(
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        windows: list = [1, 5, 10]
    ) -> pd.DataFrame:
        """
        Add index (e.g., NIFTY) return features

        Args:
            stock_df: Stock price DataFrame
            index_df: Index price DataFrame
            windows: List of return windows

        Returns:
            DataFrame with index features
        """
        features = stock_df.copy()

        # Align index data with stock data
        index_aligned = index_df.reindex(stock_df.index, method='ffill')

        for window in windows:
            features[f'index_return_{window}'] = (
                index_aligned['close'].pct_change(window)
            )

        # Index volatility
        features['index_volatility_10'] = (
            index_aligned['close'].pct_change().rolling(10).std() * np.sqrt(252)
        )

        return features

    @staticmethod
    def beta_features(
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        windows: list = [20, 60]
    ) -> pd.DataFrame:
        """
        Calculate rolling beta (stock vs index)

        Args:
            stock_df: Stock price DataFrame
            index_df: Index price DataFrame
            windows: List of rolling windows

        Returns:
            DataFrame with beta features
        """
        features = stock_df.copy()

        # Calculate returns
        stock_returns = stock_df['close'].pct_change()
        index_returns = index_df.reindex(stock_df.index, method='ffill')['close'].pct_change()

        for window in windows:
            # Rolling covariance and variance
            cov = stock_returns.rolling(window).cov(index_returns)
            var = index_returns.rolling(window).var()

            features[f'beta_{window}'] = cov / (var + 1e-8)

        return features

    @staticmethod
    def correlation_features(
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        windows: list = [20, 60]
    ) -> pd.DataFrame:
        """
        Rolling correlation with market index

        Args:
            stock_df: Stock price DataFrame
            index_df: Index price DataFrame
            windows: List of rolling windows

        Returns:
            DataFrame with correlation features
        """
        features = stock_df.copy()

        stock_returns = stock_df['close'].pct_change()
        index_returns = index_df.reindex(stock_df.index, method='ffill')['close'].pct_change()

        for window in windows:
            features[f'corr_index_{window}'] = (
                stock_returns.rolling(window).corr(index_returns)
            )

        return features

    @staticmethod
    def market_regime_features(
        index_df: pd.DataFrame,
        volatility_threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Simple market regime indicators

        Args:
            index_df: Index price DataFrame
            volatility_threshold: Threshold for high volatility regime

        Returns:
            DataFrame with regime features
        """
        features = index_df.copy()

        # Trend regime (above/below MA)
        features['trend_regime'] = (
            (index_df['close'] > index_df['close'].rolling(50).mean()).astype(int)
        )

        # Volatility regime
        vol = index_df['close'].pct_change().rolling(20).std()
        features['volatility_regime'] = (vol > volatility_threshold).astype(int)

        # Drawdown from peak
        peak = index_df['close'].expanding().max()
        features['drawdown'] = (index_df['close'] - peak) / peak

        return features
