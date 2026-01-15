"""
Probability Calibration
Isotonic and Platt scaling for better probability estimates
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    Calibrates model probabilities using isotonic regression or Platt scaling
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator

        Args:
            method: "isotonic" or "sigmoid" (Platt scaling)
        """
        self.method = method
        self.calibrator = None

    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Fit calibration model

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities (uncalibrated)
        """
        logger.info(f"Fitting {self.method} calibration model")

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
        elif self.method == "sigmoid":
            # Platt scaling: fit logistic regression on predictions
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Transform predictions using fitted calibrator

        Args:
            y_pred_proba: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted yet")

        if self.method == "isotonic":
            return self.calibrator.transform(y_pred_proba)
        elif self.method == "sigmoid":
            return self.calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]

    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, y_pred_proba)
        return self.transform(y_pred_proba)

    @staticmethod
    def evaluate_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10):
        """
        Evaluate calibration quality using calibration curve

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dict with calibration metrics
        """
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=n_bins,
            strategy='uniform'
        )

        # Expected Calibration Error (ECE)
        bin_totals = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))[0]
        ece = np.sum(bin_totals * np.abs(prob_true - prob_pred)) / len(y_true)

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(prob_true - prob_pred))

        return {
            'ece': ece,
            'mce': mce,
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
