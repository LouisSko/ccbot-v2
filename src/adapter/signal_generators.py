"""Adapter for trade signal generator."""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

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

    def extract_pred_types(self, predictions: List[Prediction]) -> Tuple[
        Dict[Tuple[str, pd.Timestamp], Prediction],
        Dict[Tuple[str, pd.Timestamp], Prediction],
        Dict[Tuple[str, pd.Timestamp], Prediction],
    ]:
        """extract different prediction types."""

        prediction_dir, prediction_vola, prediction_reg = {}, {}, {}
        for prediction in predictions:
            if prediction.prediction_type == "direction":
                prediction_dir[(prediction.symbol, prediction.time)] = prediction
            elif prediction.prediction_type == "volatility":
                prediction_vola[(prediction.symbol, prediction.time)] = prediction
            elif prediction.prediction_type == "regression":
                prediction_reg[(prediction.symbol, prediction.time)] = prediction

        return prediction_dir, prediction_vola, prediction_reg

    # or i separate the trades myself
    def generate_trade_signals(
        self,
        predictions: List[Prediction],
    ) -> List[TradeSignal]:
        """generate trade signals based on direction predictions and volatility predictions.

        The directional predictions are needed, the vola predictions are optional and can be used for additional confirmation.
        """

        trade_signals = []

        # filter preds. Currently we do not make use of regression predictions
        dir_preds, vola_preds, reg_preds = self.extract_pred_types(predictions)

        for key, pred in dir_preds.items():
            try:
                # if vola prediction is 0 (low), don't trade
                if vola_preds and vola_preds[key].prediction == 0:
                    continue

                if pred.prediction == 1:
                    position_side = "buy"
                    limit_price = pred.execution_price or (pred.close - (pred.atr * 0.1))
                    # limit_price = close * (1 - 0.001)

                elif pred.prediction == -1:
                    position_side = "sell"
                    limit_price = pred.execution_price or (pred.close + (pred.atr * 0.1))
                    # limit_price = close * (1 + 0.001)

                else:
                    continue
            except Exception as e:
                logger.info(50 * "-")
                logger.info(pred)
                raise ValueError(f"error: {e}") from e

            stop_loss_price, _ = self.calculate_stops(limit_price, pred.atr, position_side)

            trade_signal = TradeSignal(
                time=pred.time,
                symbol=pred.symbol,
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
        stop_loss = None
        take_profit = None

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
        else:
            raise ValueError("exit_type needs to be specified.")

        return stop_loss, take_profit


# TODO: maybe add this to the engine. Problem here is that we might not have the newest current price data
class SignalGeneratorSimple(TradeSignalGenerator):
    """Implementation of the trade signal generator."""

    def __init__(self, config: GeneratorSettings):
        super().__init__(config)

        self.atr_stop_loss_multiplier = 1.5
        self.atr_take_profit_multiplier = 1.5
        self.percent_stop_loss = 0.01
        self.percent_take_profit = 0.01
        self.exit_type = "atr"

    def extract_pred_types(self, predictions: List[Prediction]) -> Tuple[
        Dict[Tuple[str, pd.Timestamp], Prediction],
        Dict[Tuple[str, pd.Timestamp], Prediction],
        Dict[Tuple[str, pd.Timestamp], Prediction],
    ]:
        """extract different prediction types."""

        prediction_dir, prediction_vola, prediction_reg = {}, {}, {}
        for prediction in predictions:
            if prediction.prediction_type == "direction":
                prediction_dir[(prediction.symbol, prediction.time)] = prediction
            elif prediction.prediction_type == "volatility":
                prediction_vola[(prediction.symbol, prediction.time)] = prediction
            elif prediction.prediction_type == "regression":
                prediction_reg[(prediction.symbol, prediction.time)] = prediction

        return prediction_dir, prediction_vola, prediction_reg

    def generate_trade_signals(
        self,
        predictions: List[Prediction],
    ) -> List[TradeSignal]:
        """generate trade signals based on direction predictions and volatility predictions.

        The directional predictions are needed, the vola predictions are optional and can be used for additional confirmation.
        """

        trade_signals = []

        for pred in predictions:

            try:
                if pred.prediction == 1:
                    position_side = "buy"
                    limit_price = pred.execution_price or (pred.close - (pred.atr * 0.1))
                    # limit_price = close * (1 - 0.001)

                elif pred.prediction == -1:
                    position_side = "sell"
                    limit_price = pred.execution_price or (pred.close + (pred.atr * 0.1))
                    # limit_price = close * (1 + 0.001)

                else:
                    continue

            except Exception as e:
                logger.info(50 * "-")
                logger.info(pred)
                raise ValueError(f"error: {e}") from e

            stop_loss_price, _ = self.calculate_stops(limit_price, pred.atr, position_side)

            trade_signal = TradeSignal(
                time=pred.time,
                symbol=pred.symbol,
                order_type=self.config.order_type,
                position_side=position_side,
                limit_price=limit_price,
                stop_loss_price=None,
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
        stop_loss = None
        take_profit = None

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
        else:
            raise ValueError("exit_type needs to be specified.")

        return stop_loss, take_profit
