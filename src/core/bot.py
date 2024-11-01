"""Module for defining the trading bot class."""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.common_packages import create_logger
from src.core.engine import TradingEngine
from src.core.pipeline import Pipeline

logger = create_logger(log_level=os.getenv("LOGGING_LEVEL", "INFO"), logger_name=__name__)


class TradingBotSettings(BaseModel):
    """Defines the settings for the trading bot"""

    trading_engine: TradingEngine
    pipeline: Pipeline
    simulation: bool  # whether to run simulation or not

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TradingBot:
    """Trading bot logic"""

    def __init__(self, config: TradingBotSettings):
        self.config = config

        if self.config.simulation:
            logger.info("Use Trading Bot in Simulation Mode.")

            self.config.pipeline.train()

            # log the current date of the pipeline
            self.pipeline_current_date = self.config.pipeline.get_current_date()

            logger.info("Pipeline current date: %s", self.pipeline_current_date)
            # time.sleep(5)
        self.timeframe = self.config.pipeline.config.timeframe
        self.tf_in_mins = self.timeframe.total_seconds() / 60

        logger.info("Trading Bot set execution timeframe: %s", self.timeframe)

    def run_bot(self) -> None:
        """Run the trading bot to process data and execute trades.

        Can be run either in simulation mode for backtesting using a mock exchange or realtime.
        """

        # real trading
        if self.config.simulation is False:

            current_date = pd.Timestamp.now(tz="UTC")
            predictions = self.config.pipeline.trigger()
            self.config.trading_engine.execute_orders(predictions)

        # simulated trading
        if self.config.simulation:

            # get current date fo the mock exchange
            current_date = self.config.trading_engine.config.exchange.current_date

            self.config.trading_engine.config.exchange.check_limit_orders()
            self.config.trading_engine.config.exchange.check_stop_losses()
            self.config.trading_engine.config.exchange.check_take_profit()

            # exexute pipeline if the two dates match
            if self.pipeline_current_date == current_date:
                logger.info("\n\n Simulation current timestep: %s \n", current_date)
                trading_signals = self.config.pipeline.trigger()
                self.config.trading_engine.execute_orders(trading_signals)
                self.pipeline_current_date = self.config.pipeline.get_current_date()

            # train the pipeline every month.
            if current_date >= self.config.pipeline.last_training_date + pd.DateOffset(months=2):
                self.config.pipeline.train()

            # get the next date
            self.config.trading_engine.config.exchange.next_step()

    def get_next_execution_time(self) -> int:
        """Calculate the time until the next execution.

        Returns:
            int: Duration in seconds until the next execution.
        """
        # TODO: are you sure the scheduling is correct because of the different tzs?

        now = datetime.now()
        next_execution = now.replace(second=0, microsecond=0)

        # Adjust next execution time to align with the execution interval
        minutes_to_add = self.tf_in_mins - (now.minute % self.tf_in_mins)
        next_execution += timedelta(minutes=minutes_to_add, seconds=1)

        time_to_sleep = (next_execution - datetime.now()).total_seconds()
        logger.info("Next execution scheduled at %s UTC, sleeping for %s seconds.", next_execution, time_to_sleep)

        return max(time_to_sleep, 0)

    def schedule_bot(self) -> None:
        """Schedule and continuously run the bot at specified intervals."""

        while True:

            # end the simulation in case there is no data left
            if self.config.simulation and self.pipeline_current_date is None:
                logger.info("End of simulation. No more data available.")
                path = os.path.join(self.config.pipeline.config.save_dir, "trading_results.json")
                self.config.trading_engine.config.exchange.save_trade_history(path)                
                break

            # real time - set bot to sleep
            if not self.config.simulation:
                time_to_sleep = self.get_next_execution_time()
                time.sleep(time_to_sleep)

            self.run_bot()
