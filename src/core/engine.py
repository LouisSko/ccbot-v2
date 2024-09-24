"""Module for hosting the exchange logic."""

import os
from abc import ABC, abstractmethod
from typing import List

from ccxt import Exchange
from pydantic import BaseModel, ConfigDict

from src.common_packages import create_logger
from src.core.generator import TradeSignal

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


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
