"""Module for running the bot in simulation mode."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.adapter.mock_exchange import MockExchange, load_mock_exchange
from src.adapter.trading_engine import CCXTFuturesTradingEngine
from src.common_packages import create_logger
from src.core.bot import TradingBot, TradingBotSettings
from src.core.engine import EngineSettings
from src.core.pipeline import Pipeline, load_pipeline

logger = create_logger(log_level=os.getenv("LOGGING_LEVEL", "INFO"), logger_name=__name__)

load_dotenv()

if __name__ == "__main__":

    pipeline: Pipeline = load_pipeline(os.getenv("PIPELINE_DIR"))
    pipeline.activate_simulation_mode()

    mock_exchange: MockExchange = load_mock_exchange(save_directory=os.getenv("MOCK_EX_DIR"))

    settings = EngineSettings(exchange=mock_exchange, symbols=mock_exchange.config.symbols)
    trading_engine = CCXTFuturesTradingEngine(config=settings)

    settings = TradingBotSettings(
        trading_engine=trading_engine,
        pipeline=pipeline,
        simulation=True,
    )

    trading_bot = TradingBot(config=settings)

    trading_bot.schedule_bot()
