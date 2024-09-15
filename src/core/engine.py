"""Module for hosting the exchange logic."""

import os
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import pandas as pd
from ccxt import Exchange
from pydantic import BaseModel, field_validator, model_validator

from src.common_packages import create_logger

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class TradeSignal(BaseModel):
    """Trade signal for a single symbol."""

    time: pd.Timestamp
    symbol: str
    position_side: str  # buy, sell, no_trade
    order_type: str  # limit, market
    limit_price: Optional[float]  # optional provide if order_type = "limit"
    stop_loss_price: Optional[float]  # optionally provide that
    take_profit_price: Optional[float]  # optionally provide that
    order_amount: Optional[float]

    @field_validator("position_side")
    def validate_position_type(self, value):
        """validate the position type"""

        if value not in {"buy", "sell", "no_trade"}:
            raise ValueError('position_side must be "buy", "sell", or "no_trade"')
        return value

    @field_validator("order_type")
    def validate_order_type(self, value):
        """check if order_type is set correctly"""

        if value not in {"limit", "market"}:
            raise ValueError('order_type must be "limit" or "market"')
        return value

    @model_validator(mode="before")
    def check_limit_price_if_limit_order(self, values):
        """check if limit price is set if limit order is selected."""

        order_type = values.get("order_type")
        limit_price = values.get("limit_price")
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price must be provided when order_type is 'limit'")
        return values


class EngineSettings(BaseModel):
    """Initialize the Exchange object with a specific exchange ID.

    Args:
        exchange (ccxt.Exchange): provide a ccxt exchange
        symbols (List[str]): symbols to trade.
    """

    exchange: Exchange
    symbols: List[str]


class TradingEngine(ABC):
    """Abstract trading engine"""

    def __init__(self, config: EngineSettings):
        self.config = config
        self.order_details = {}
        self.set_trading_settings()

    @abstractmethod
    def set_trading_settings(self) -> None:
        """Set the trading settings like leverage, precision, etc."""

        pass

    @abstractmethod
    def log_open_positions(self) -> None:
        """Log all open positions."""

        pass

    @abstractmethod
    def determine_order_size(
        self,
        trade_signal: TradeSignal,
        method: Literal["free_balance", "total_balance", "risk_based"] = "risk_based",
        risk_pct: float = 0.01,
    ) -> Union[float, None]:
        """Determine the position size based on the chosen method."""

        pass

    @abstractmethod
    def execute_orders(self, trade_signals: List[TradeSignal]):
        """Execute trades based on trade signals."""

        pass
