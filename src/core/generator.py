"""This Module hosts the logic for the trade signal generator."""

import os
from abc import abstractmethod
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.model import Prediction

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


class GeneratorSettings(BaseModel):
    """Parameters for initializing the generator."""

    object_id: ObjectId
    depends_on: ObjectId  # ID of the processor this model depends on.
    order_type: Literal["limit", "market"]


class GeneratorConfiguration(BaseConfiguration):
    """Configuration specific to a generator."""

    component_type: str = Field(
        default="generator",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: GeneratorSettings


class TradeSignalGenerator(BasePipelineComponent):
    """Abstract base class for machine learning models."""

    def __init__(self, config: GeneratorSettings) -> None:
        """Initialize the BaseModel with configuration settings."""

        self.config = config

    @abstractmethod
    def generate_trade_signals(self, predictions: List[Prediction]) -> List[TradeSignal]:
        """generate trade signals based on predictions."""

    def create_configuration(self) -> GeneratorConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{GeneratorConfiguration.__module__}.{GeneratorConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return GeneratorConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )
