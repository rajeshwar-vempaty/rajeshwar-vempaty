"""
Signal Generator
Loads model and generates trading signals from live features
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional
import logging
from ...features.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates trading signals using trained ML model
    """

    def __init__(
        self,
        model_path: str,
        feature_pipeline: FeaturePipeline,
        confidence_threshold: float = 0.6,
        feature_schema: Optional[list] = None
    ):
        """
        Initialize signal generator

        Args:
            model_path: Path to trained model file
            feature_pipeline: Feature pipeline instance
            confidence_threshold: Minimum probability to generate signal
            feature_schema: Expected feature column names
        """
        self.model = lgb.Booster(model_file=model_path)
        self.feature_pipeline = feature_pipeline
        self.confidence_threshold = confidence_threshold
        self.feature_schema = feature_schema

        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Confidence threshold: {confidence_threshold}")

    def generate_signal(
        self,
        current_data: pd.DataFrame,
        index_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate trading signal from current market data

        Args:
            current_data: Recent OHLCV data (must have sufficient history for features)
            index_data: Optional index data for market features

        Returns:
            Dict with signal, probability, and metadata
        """
        try:
            # Generate features
            features = self.feature_pipeline.generate_features(current_data, index_data)

            # Get feature columns
            feature_cols = self.feature_pipeline.get_feature_columns(features)

            # Use schema if provided
            if self.feature_schema is not None:
                feature_cols = self.feature_schema

            # Get latest feature vector (most recent bar)
            X = features[feature_cols].iloc[[-1]]

            # Check for NaN
            if X.isnull().any().any():
                logger.warning("NaN values in features, returning no signal")
                return {
                    'signal': 0,
                    'probability': 0.5,
                    'confidence': 0.0,
                    'timestamp': pd.Timestamp.now(),
                    'error': 'NaN in features'
                }

            # Predict
            probability = self.model.predict(X)[0]

            # Generate signal
            signal = 1 if probability > self.confidence_threshold else 0
            confidence = abs(probability - 0.5) * 2  # Scale to [0, 1]

            result = {
                'signal': signal,
                'probability': probability,
                'confidence': confidence,
                'timestamp': pd.Timestamp.now(),
                'price': current_data['close'].iloc[-1],
                'features': X.to_dict('records')[0]
            }

            logger.debug(
                f"Signal: {signal}, Probability: {probability:.3f}, "
                f"Confidence: {confidence:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                'signal': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'timestamp': pd.Timestamp.now(),
                'error': str(e)
            }

    def batch_generate_signals(
        self,
        data_dict: Dict[str, pd.DataFrame],
        index_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """
        Generate signals for multiple symbols

        Args:
            data_dict: Dict mapping symbol -> OHLCV DataFrame
            index_data: Optional index data

        Returns:
            Dict mapping symbol -> signal dict
        """
        signals = {}

        for symbol, data in data_dict.items():
            signals[symbol] = self.generate_signal(data, index_data)

        return signals

    def validate_features(self, features: pd.DataFrame) -> bool:
        """
        Validate that features match expected schema

        Args:
            features: Feature DataFrame

        Returns:
            True if valid
        """
        if self.feature_schema is None:
            return True

        missing = set(self.feature_schema) - set(features.columns)
        if missing:
            logger.error(f"Missing features: {missing}")
            return False

        return True
