"""
Walk-Forward Cross-Validation
Time-series aware validation with purging and embargo
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class WalkForwardCV:
    """
    Walk-forward cross-validation for time-series data
    Implements purging and embargo to prevent leakage
    """

    def __init__(
        self,
        train_period_days: int = 180,
        test_period_days: int = 30,
        embargo_days: int = 5,
        min_train_samples: int = 1000
    ):
        """
        Initialize walk-forward CV

        Args:
            train_period_days: Training window size in days
            test_period_days: Test window size in days
            embargo_days: Embargo period after test set (prevents leakage)
            min_train_samples: Minimum samples required for training
        """
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.embargo = timedelta(days=embargo_days)
        self.min_train_samples = min_train_samples

    def split(
        self,
        df: pd.DataFrame,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None
    ) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test splits

        Args:
            df: DataFrame with datetime index
            start_date: Optional start date for validation
            end_date: Optional end date for validation

        Yields:
            Tuples of (train_index, test_index)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        dates = df.index.sort_values()
        start = start_date or dates[0]
        end = end_date or dates[-1]

        current_test_start = start + self.train_period

        while current_test_start < end:
            # Training period
            train_start = current_test_start - self.train_period
            train_end = current_test_start

            # Test period
            test_end = min(current_test_start + self.test_period, end)

            # Get indices
            train_idx = (dates >= train_start) & (dates < train_end)
            test_idx = (dates >= current_test_start) & (dates < test_end)

            # Check minimum samples
            if train_idx.sum() < self.min_train_samples:
                logger.warning(
                    f"Skipping split: only {train_idx.sum()} training samples "
                    f"(minimum {self.min_train_samples})"
                )
                current_test_start += self.test_period
                continue

            logger.info(
                f"Split: Train={train_idx.sum()} samples ({train_start.date()} to {train_end.date()}), "
                f"Test={test_idx.sum()} samples ({current_test_start.date()} to {test_end.date()})"
            )

            yield dates[train_idx], dates[test_idx]

            # Move to next period (with embargo)
            current_test_start = test_end + self.embargo

    def get_n_splits(self, df: pd.DataFrame) -> int:
        """
        Get number of splits

        Args:
            df: DataFrame with datetime index

        Returns:
            Number of splits
        """
        return sum(1 for _ in self.split(df))

    @staticmethod
    def purge_overlapping_labels(
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
        horizon: int
    ) -> pd.DatetimeIndex:
        """
        Purge training samples whose labels overlap with test period

        Args:
            train_idx: Training set index
            test_idx: Test set index
            horizon: Label horizon (in bars)

        Returns:
            Purged training index
        """
        if len(test_idx) == 0:
            return train_idx

        test_start = test_idx.min()

        # Remove training samples within 'horizon' bars of test start
        # This prevents labels from leaking into test period
        purge_cutoff = test_start - pd.Timedelta(hours=horizon)  # Adjust based on bar frequency

        purged_idx = train_idx[train_idx < purge_cutoff]

        logger.info(f"Purged {len(train_idx) - len(purged_idx)} overlapping samples")

        return purged_idx
