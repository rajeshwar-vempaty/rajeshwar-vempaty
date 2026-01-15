"""
Vectorized Backtester
Fast backtesting with realistic transaction costs
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from .transaction_costs import TransactionCostModel

logger = logging.getLogger(__name__)


class Backtester:
    """
    Vectorized backtester for ML trading strategies
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        cost_model: Optional[TransactionCostModel] = None,
        max_position_size: float = 1.0,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital
            cost_model: Transaction cost model
            max_position_size: Maximum position size as fraction of capital
            confidence_threshold: Minimum probability to enter trade
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.max_position_size = max_position_size
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        prices: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Run backtest

        Args:
            data: DataFrame with OHLCV data
            predictions: Predicted probabilities (class 1)
            prices: Optional price series (default: use 'close')

        Returns:
            DataFrame with backtest results
        """
        if prices is None:
            prices = data['close']

        logger.info(f"Running backtest on {len(data)} bars")

        # Generate signals
        signals = self._generate_signals(predictions)

        # Calculate positions
        positions = signals.copy()  # For simplicity: +1 (long), 0 (flat)

        # Calculate returns
        returns = prices.pct_change()

        # Strategy returns (before costs)
        strategy_returns_gross = positions.shift(1) * returns

        # Transaction costs
        trades = positions.diff().abs()
        costs = self.cost_model.calculate_costs(prices, trades)
        cost_returns = -costs / self.initial_capital

        # Net returns
        strategy_returns_net = strategy_returns_gross + cost_returns

        # Build results DataFrame
        results = pd.DataFrame({
            'price': prices,
            'signal': signals,
            'position': positions,
            'prediction': predictions,
            'return': returns,
            'strategy_return_gross': strategy_returns_gross,
            'strategy_return_net': strategy_returns_net,
            'costs': costs,
            'trades': trades
        }, index=data.index)

        # Cumulative metrics
        results['cumulative_return_gross'] = (1 + results['strategy_return_gross']).cumprod() - 1
        results['cumulative_return_net'] = (1 + results['strategy_return_net']).cumprod() - 1
        results['cumulative_costs'] = results['costs'].cumsum()

        # Portfolio value
        results['portfolio_value'] = self.initial_capital * (1 + results['cumulative_return_net'])

        logger.info(f"Backtest complete: {trades.sum()} trades executed")

        return results

    def _generate_signals(self, predictions: np.ndarray) -> pd.Series:
        """
        Convert predictions to trading signals

        Args:
            predictions: Predicted probabilities

        Returns:
            Signal series: 1 (long), 0 (flat)
        """
        signals = pd.Series(0, index=range(len(predictions)))

        # Long when confidence > threshold
        signals[predictions > self.confidence_threshold] = 1

        return signals

    def run_multiple_symbols(
        self,
        data_dict: Dict[str, pd.DataFrame],
        predictions_dict: Dict[str, np.ndarray],
        allocation: str = "equal"
    ) -> pd.DataFrame:
        """
        Run backtest on multiple symbols with portfolio allocation

        Args:
            data_dict: Dict mapping symbol -> OHLCV DataFrame
            predictions_dict: Dict mapping symbol -> predictions
            allocation: "equal" or "weighted" (by confidence)

        Returns:
            Portfolio-level backtest results
        """
        results = {}

        # Run backtest for each symbol
        for symbol in data_dict.keys():
            if symbol not in predictions_dict:
                logger.warning(f"No predictions for {symbol}, skipping")
                continue

            results[symbol] = self.run(
                data_dict[symbol],
                predictions_dict[symbol]
            )

        # Aggregate to portfolio level
        # Simple equal weight for now
        portfolio_returns = pd.DataFrame({
            symbol: res['strategy_return_net']
            for symbol, res in results.items()
        })

        portfolio_returns_agg = portfolio_returns.mean(axis=1)

        # Portfolio metrics
        portfolio_results = pd.DataFrame({
            'portfolio_return': portfolio_returns_agg,
            'cumulative_return': (1 + portfolio_returns_agg).cumprod() - 1,
            'portfolio_value': self.initial_capital * (1 + (1 + portfolio_returns_agg).cumprod() - 1)
        })

        return portfolio_results
