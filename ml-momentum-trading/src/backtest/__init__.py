"""
Backtesting Module
Vectorized backtester with realistic costs and slippage
"""

from .backtester import Backtester
from .transaction_costs import TransactionCostModel
from .performance_metrics import PerformanceMetrics

__all__ = ["Backtester", "TransactionCostModel", "PerformanceMetrics"]
