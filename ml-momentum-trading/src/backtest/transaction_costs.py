"""
Transaction Cost Model
Realistic modeling of brokerage, exchange fees, STT, slippage
"""

import numpy as np
import pandas as pd


class TransactionCostModel:
    """
    Models transaction costs for Indian equity markets (Zerodha)
    """

    def __init__(
        self,
        brokerage_pct: float = 0.0003,  # 0.03% or Rs 20 per trade (whichever lower)
        stt_pct: float = 0.00025,        # 0.025% on sell side (equity delivery)
        exchange_charges_pct: float = 0.0000325,  # ~0.00325%
        gst_pct: float = 0.18,           # 18% GST on brokerage + exchange
        sebi_charges_pct: float = 0.0000001,  # Negligible
        stamp_duty_pct: float = 0.00015,  # 0.015% on buy side
        slippage_bps: float = 5.0,        # 5 bps base slippage
        volatility_multiplier: float = 2.0  # Slippage scales with volatility
    ):
        """
        Initialize cost model with Zerodha-like charges

        Args:
            brokerage_pct: Brokerage as % of trade value
            stt_pct: Securities Transaction Tax %
            exchange_charges_pct: NSE charges %
            gst_pct: GST on brokerage + exchange
            sebi_charges_pct: SEBI turnover charges
            stamp_duty_pct: Stamp duty on buy side
            slippage_bps: Base slippage in basis points
            volatility_multiplier: Slippage volatility scaling
        """
        self.brokerage_pct = brokerage_pct
        self.stt_pct = stt_pct
        self.exchange_charges_pct = exchange_charges_pct
        self.gst_pct = gst_pct
        self.sebi_charges_pct = sebi_charges_pct
        self.stamp_duty_pct = stamp_duty_pct
        self.slippage_bps = slippage_bps
        self.volatility_multiplier = volatility_multiplier

    def calculate_costs(
        self,
        prices: pd.Series,
        trade_signals: pd.Series,
        position_size: float = 1.0
    ) -> pd.Series:
        """
        Calculate total transaction costs per trade

        Args:
            prices: Price series
            trade_signals: Binary trade signals (1 = trade, 0 = no trade)
            position_size: Position size as fraction of capital

        Returns:
            Series of transaction costs
        """
        trade_value = prices * trade_signals * position_size

        # Brokerage (capped at Rs 20 per trade - not modeling cap for simplicity)
        brokerage = trade_value * self.brokerage_pct

        # STT (only on sell; for simplicity, apply on all trades)
        stt = trade_value * self.stt_pct

        # Exchange charges
        exchange_charges = trade_value * self.exchange_charges_pct

        # GST on brokerage + exchange
        gst = (brokerage + exchange_charges) * self.gst_pct

        # SEBI charges
        sebi_charges = trade_value * self.sebi_charges_pct

        # Stamp duty (buy side; apply on all)
        stamp_duty = trade_value * self.stamp_duty_pct

        # Slippage (volatility-adjusted)
        volatility = prices.pct_change().rolling(20).std().fillna(0.02)
        slippage_pct = (self.slippage_bps / 10000) * (1 + self.volatility_multiplier * volatility)
        slippage = trade_value * slippage_pct

        # Total costs
        total_costs = (
            brokerage + stt + exchange_charges + gst +
            sebi_charges + stamp_duty + slippage
        )

        return total_costs

    def get_cost_breakdown(
        self,
        price: float,
        trade_size: float = 1.0
    ) -> dict:
        """
        Get detailed cost breakdown for a single trade

        Args:
            price: Trade price
            trade_size: Trade size (quantity)

        Returns:
            Dict with cost components
        """
        trade_value = price * trade_size

        brokerage = trade_value * self.brokerage_pct
        stt = trade_value * self.stt_pct
        exchange_charges = trade_value * self.exchange_charges_pct
        gst = (brokerage + exchange_charges) * self.gst_pct
        sebi_charges = trade_value * self.sebi_charges_pct
        stamp_duty = trade_value * self.stamp_duty_pct
        slippage = trade_value * (self.slippage_bps / 10000)

        total = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty + slippage

        return {
            'trade_value': trade_value,
            'brokerage': brokerage,
            'stt': stt,
            'exchange_charges': exchange_charges,
            'gst': gst,
            'sebi_charges': sebi_charges,
            'stamp_duty': stamp_duty,
            'slippage': slippage,
            'total_cost': total,
            'cost_pct': (total / trade_value) * 100
        }
