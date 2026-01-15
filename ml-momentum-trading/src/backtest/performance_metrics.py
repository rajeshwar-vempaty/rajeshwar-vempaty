"""
Performance Metrics
Sharpe, Sortino, drawdown, hit rate, and other trading metrics
"""

import numpy as np
import pandas as pd
from typing import Dict


class PerformanceMetrics:
    """
    Calculate trading strategy performance metrics
    """

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        """Total cumulative return"""
        return (1 + returns).prod() - 1

    @staticmethod
    def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Compound Annual Growth Rate

        Args:
            returns: Return series
            periods_per_year: Trading periods per year (252 for daily, 252*78 for 5-min)

        Returns:
            Annualized return
        """
        n_periods = len(returns)
        total_return = (1 + returns).prod()
        years = n_periods / periods_per_year
        return (total_return ** (1 / years)) - 1 if years > 0 else 0.0

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06, periods_per_year: int = 252) -> float:
        """
        Sharpe Ratio

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / (excess_returns.std() + 1e-8)

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06, periods_per_year: int = 252) -> float:
        """
        Sortino Ratio (penalizes only downside volatility)

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()

        return np.sqrt(periods_per_year) * excess_returns.mean() / (downside_std + 1e-8)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        Maximum drawdown

        Args:
            returns: Return series

        Returns:
            Maximum drawdown (negative value)
        """
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def hit_rate(returns: pd.Series) -> float:
        """
        Hit rate (fraction of profitable periods)

        Args:
            returns: Return series

        Returns:
            Hit rate (0 to 1)
        """
        return (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Profit factor (gross profit / gross loss)

        Args:
            returns: Return series

        Returns:
            Profit factor
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / gross_loss if gross_loss > 0 else np.inf

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calmar Ratio (CAGR / Max Drawdown)

        Args:
            returns: Return series
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        cagr_val = PerformanceMetrics.cagr(returns, periods_per_year)
        max_dd = abs(PerformanceMetrics.max_drawdown(returns))
        return cagr_val / max_dd if max_dd > 0 else 0.0

    @staticmethod
    def comprehensive_report(
        returns: pd.Series,
        trades: pd.Series = None,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.06
    ) -> Dict[str, float]:
        """
        Generate comprehensive performance report

        Args:
            returns: Return series
            trades: Optional trade signal series
            periods_per_year: Trading periods per year
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict with all performance metrics
        """
        metrics = {
            'total_return': PerformanceMetrics.total_return(returns),
            'cagr': PerformanceMetrics.cagr(returns, periods_per_year),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year),
            'max_drawdown': PerformanceMetrics.max_drawdown(returns),
            'hit_rate': PerformanceMetrics.hit_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, periods_per_year),
            'volatility': returns.std() * np.sqrt(periods_per_year),
            'mean_return': returns.mean() * periods_per_year,
            'median_return': returns.median() * periods_per_year,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }

        if trades is not None:
            metrics['total_trades'] = trades.sum()
            metrics['avg_trades_per_period'] = trades.mean()

        return metrics
