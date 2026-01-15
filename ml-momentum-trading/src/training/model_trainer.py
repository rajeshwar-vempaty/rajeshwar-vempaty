"""
Model Trainer
Handles training of LightGBM/XGBoost models with proper validation
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates gradient boosting models
    """

    def __init__(self, model_type: str = "lightgbm", params: Optional[Dict] = None):
        """
        Initialize trainer

        Args:
            model_type: "lightgbm" or "xgboost"
            params: Model hyperparameters
        """
        self.model_type = model_type
        self.params = params or self._default_params()
        self.model = None
        self.feature_importance = None

    def _default_params(self) -> Dict:
        """Default LightGBM parameters"""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ) -> Dict:
        """
        Train LightGBM model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_boost_round: Maximum boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Training metrics dict
        """
        logger.info(f"Training {self.model_type} model")
        logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val) if X_val is not None else 0} samples")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val) if X_val is not None else None

        # Train
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds)
        ] if valid_data else [lgb.log_evaluation(period=100)]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data] if valid_data else [train_data],
            valid_names=['train', 'valid'] if valid_data else ['train'],
            callbacks=callbacks
        )

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        # Metrics
        metrics = self._evaluate(X_train, y_train, prefix='train')

        if X_val is not None:
            val_metrics = self._evaluate(X_val, y_val, prefix='val')
            metrics.update(val_metrics)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Features

        Returns:
            Predicted probabilities (class 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels

        Args:
            X: Features
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        probs = self.predict(X)
        return (probs > threshold).astype(int)

    def _evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Features
            y: True labels
            prefix: Metric name prefix

        Returns:
            Metrics dict
        """
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            f'{prefix}_accuracy': accuracy_score(y, y_pred),
            f'{prefix}_auc': roc_auc_score(y, y_pred_proba),
            f'{prefix}_logloss': log_loss(y, y_pred_proba)
        }

        # Hit rate (for positive predictions only)
        pos_mask = y_pred == 1
        if pos_mask.sum() > 0:
            metrics[f'{prefix}_hit_rate'] = (y[pos_mask] == y_pred[pos_mask]).mean()
        else:
            metrics[f'{prefix}_hit_rate'] = 0.0

        return metrics

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features

        Args:
            n: Number of features

        Returns:
            DataFrame with top features
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")

        return self.feature_importance.head(n)

    def save(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")
