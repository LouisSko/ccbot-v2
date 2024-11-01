"""Module for running the bot in real time."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ccxt
from dotenv import load_dotenv

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


    exchange_class = getattr(ccxt, "bitget")
    exchange = exchange_class(
        {
            "apiKey": os.getenv("EX_API_KEY"),
            "secret": os.getenv("SECRET_KEY"),
            "password": os.getenv("PASSWORD"),
            "options": {"defaultType": "swap"},
            "timeout": 30000,
            "enableRateLimit": True,
        }
    )

    settings = EngineSettings(exchange=exchange, symbols=pipeline.get_symbols())
    trading_engine = CCXTFuturesTradingEngine(config=settings)

    settings = TradingBotSettings(
        trading_engine=trading_engine,
        pipeline=pipeline,
        simulation=False,
    )

    trading_bot = TradingBot(config=settings)

    trading_bot.schedule_bot()
