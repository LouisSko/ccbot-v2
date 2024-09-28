"""Adapter for trade signal generator."""

import os
from typing import List, Optional, Tuple

from src.common_packages import create_logger
from src.core.generator import GeneratorSettings, TradeSignal, TradeSignalGenerator
from src.core.model import Prediction

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


# TODO: maybe add this to the engine. Problem here is that we might not have the newest current price data
class SignalGenerator(TradeSignalGenerator):
    """Implementation of the trade signal generator."""

    def __init__(self, config: GeneratorSettings):
        super().__init__(config)

        self.atr_stop_loss_multiplier = 1.5
        self.atr_take_profit_multiplier = 1.5
        self.percent_stop_loss = 0.01
        self.percent_take_profit = 0.01
        self.exit_type = "atr"

    def generate_trade_signals(self, predictions: List[Prediction]) -> List[TradeSignal]:
        """generate trade signals based on predictions."""

        trade_signals = []

        for prediction in predictions:

            if prediction.prediction == 1:
                position_side = "buy"
                limit_price = prediction.close - (prediction.atr * 0.1)
                # limit_price = close * (1 - 0.001)

            elif prediction.prediction == -1:
                position_side = "sell"
                limit_price = prediction.close + (prediction.atr * 0.1)
                # limit_price = close * (1 + 0.001)

            else:
                continue

            stop_loss_price, _ = self.calculate_stops(prediction.close, prediction.atr, position_side)

            trade_signal = TradeSignal(
                time=prediction.time,
                symbol=prediction.symbol,
                order_type=self.config.order_type,
                position_side=position_side,
                limit_price=limit_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=None,
            )

            trade_signals.append(trade_signal)

        return trade_signals

    def calculate_stops(
        self,
        price: float,
        atr: Optional[float],
        position_side: str,
    ) -> Tuple[float, float]:
        """Calculates stop loss and take profit levels based on the given parameters and position type.

        Args:
            price (float): The entry price.
            atr (Optional[float]): The Average True Range value (None if not used).
            position_side (str): Type of position ('long' or 'short').

        Returns:
            Tuple[float, float]: A tuple containing stop loss and take profit prices.
        """

        if self.exit_type == "atr":
            if position_side == "buy":
                stop_loss = price - (atr * self.atr_stop_loss_multiplier)
                take_profit = price + (atr * self.atr_take_profit_multiplier)
            elif position_side == "sell":
                stop_loss = price + (atr * self.atr_stop_loss_multiplier)
                take_profit = price - (atr * self.atr_take_profit_multiplier)

        elif self.exit_type == "percent":
            if position_side == "buy":
                stop_loss = price * (1 - self.percent_stop_loss)
                take_profit = price * (1 + self.percent_take_profit)
            elif position_side == "sell":
                stop_loss = price * (1 + self.percent_stop_loss)
                take_profit = price * (1 - self.percent_take_profit)

        return stop_loss, take_profit
