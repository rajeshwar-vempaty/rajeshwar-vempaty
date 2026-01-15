"""
Feature Pipeline
Orchestrates feature generation from raw OHLCV data
"""

import pandas as pd
import logging
from typing import Optional, Dict
from .momentum_features import MomentumFeatures
from .volatility_features import VolatilityFeatures
from .market_features import MarketFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Unified pipeline for feature generation
    Ensures causal features and handles missing data
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature pipeline

        Args:
            config: Configuration dict with feature parameters
        """
        self.config = config or self._default_config()
        self.momentum = MomentumFeatures()
        self.volatility = VolatilityFeatures()
        self.market = MarketFeatures()

    @staticmethod
    def _default_config() -> Dict:
        """Default feature configuration"""
        return {
            'return_periods': [1, 3, 5, 10, 20],
            'ma_windows': [5, 10, 20, 50],
            'volatility_windows': [5, 10, 20],
            'atr_windows': [14, 20],
            'rsi_periods': [14, 20],
            'breakout_windows': [10, 20, 50],
            'volume_windows': [5, 10, 20]
        }

    def generate_features(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all features for a stock

        Args:
            df: OHLCV DataFrame for stock
            index_df: Optional index DataFrame for market features

        Returns:
            DataFrame with all features
        """
        logger.info(f"Generating features for {len(df)} bars")

        # Start with raw data
        features = df.copy()

        # Momentum features
        features = self.momentum.log_returns(
            features,
            periods=self.config['return_periods']
        )
        features = self.momentum.rolling_cumulative_return(features)
        features = self.momentum.moving_average_features(
            features,
            windows=self.config['ma_windows']
        )
        features = self.momentum.breakout_features(
            features,
            windows=self.config['breakout_windows']
        )
        features = self.momentum.volume_features(
            features,
            windows=self.config['volume_windows']
        )

        # Volatility features
        features = self.volatility.rolling_volatility(
            features,
            windows=self.config['volatility_windows']
        )
        features = self.volatility.parkinson_volatility(
            features,
            windows=self.config['volatility_windows']
        )
        features = self.volatility.garman_klass_volatility(
            features,
            windows=self.config['volatility_windows']
        )
        features = self.volatility.atr(
            features,
            windows=self.config['atr_windows']
        )
        features = self.volatility.rsi(
            features,
            periods=self.config['rsi_periods']
        )
        features = self.volatility.adx_style_features(features)

        # Market features (if index data provided)
        if index_df is not None:
            features = self.market.index_features(features, index_df)
            features = self.market.beta_features(features, index_df)
            features = self.market.correlation_features(features, index_df)

        logger.info(f"Generated {len(features.columns)} features")

        return features

    def generate_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate classification labels for next-period returns

        Args:
            df: DataFrame with 'close' column
            horizon: Periods ahead to predict
            threshold: Return threshold for positive class

        Returns:
            DataFrame with label column
        """
        features = df.copy()

        # Forward return
        features['future_return'] = (
            df['close'].shift(-horizon).pct_change(horizon)
        )

        # Binary label
        features['label'] = (features['future_return'] > threshold).astype(int)

        return features

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        horizon: int = 1,
        threshold: float = 0.0,
        dropna: bool = True
    ) -> pd.DataFrame:
        """
        Complete pipeline: features + labels

        Args:
            df: OHLCV DataFrame
            index_df: Optional index DataFrame
            horizon: Prediction horizon
            threshold: Label threshold
            dropna: Whether to drop rows with missing values

        Returns:
            Complete DataFrame ready for ML
        """
        # Generate features
        data = self.generate_features(df, index_df)

        # Generate labels
        data = self.generate_labels(data, horizon, threshold)

        # Drop NaN values (from rolling windows)
        if dropna:
            original_len = len(data)
            data = data.dropna()
            logger.info(f"Dropped {original_len - len(data)} rows with NaN values")

        return data

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of feature column names (exclude OHLCV, label, etc.)

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude = ['open', 'high', 'low', 'close', 'volume', 'label', 'future_return']
        return [col for col in df.columns if col not in exclude]
