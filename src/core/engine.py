"""Module for hosting the exchange logic."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
from ccxt import Exchange
from pydantic import BaseModel, ConfigDict, Field  # , field_validator, model_validator

from src.common_packages import create_logger

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class TradeSignal(BaseModel):
    """Represents a single trade."""

    time: pd.Timestamp = Field(..., description="Timestamp of the trade")
    symbol: str = Field(..., description="Trading symbol, e.g., 'BTC/USDT'")
    order_type: str = Field(..., description="Type of order: either 'limit' or 'market'")

    # those are values which are assigned based on the prediction
    position_side: str = Field(..., description="Type of position: either 'buy', 'sell'")
    limit_price: Optional[float] = Field(None, description="Limit price for the trade (optional for market orders)")
    stop_loss_price: Optional[float] = Field(None, description="Stop loss price (optional)")
    take_profit_price: Optional[float] = Field(None, description="Take profit price (optional)")
    order_amount: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # @model_validator("position_side")
    # def validate_position_type(self, value):
    #     """validate the position type"""

    #     if value not in {"buy", "sell", "no_trade"}:
    #         raise ValueError('position_side must be "buy", "sell", or "no_trade"')
    #     return value

    # @model_validator("order_type")
    # def validate_order_type(self, value):
    #     """check if order_type is set correctly"""

    #     if value not in {"limit", "market"}:
    #         raise ValueError('order_type must be "limit" or "market"')
    #     return value

    # @model_validator(mode="before")
    # def check_limit_price_if_limit_order(self, values):
    #     """check if limit price is set if limit order is selected."""

    #     order_type = values.get("order_type")
    #     limit_price = values.get("limit_price")
    #     if order_type == "limit" and limit_price is None:
    #         raise ValueError("limit_price must be provided when order_type is 'limit'")
    #     return values


class EngineSettings(BaseModel):
    """Initialize the Exchange object with a specific exchange ID.

    Args:
        exchange (ccxt.Exchange): provide a ccxt exchange
        symbols (List[str]): symbols to trade.
    """

    exchange: Exchange
    symbols: List[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TradingEngine(ABC):
    """Abstract trading engine"""

    def __init__(self, config: EngineSettings):
        self.config = config
        self.order_details = {}
        self.symbols_with_open_positions = []
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
    def execute_orders(self, trade_signals: List[TradeSignal]):
        """Execute trades based on trade signals."""

        pass
