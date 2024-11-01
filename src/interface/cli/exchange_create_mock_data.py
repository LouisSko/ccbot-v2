"""creates and saves the mock exchange data."""

import os
import sys

# TODO fix this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd
from dotenv import load_dotenv

from src.adapter.mock_exchange import MockExchange, MockExchangeSettings
from src.core.base import ObjectId
from src.core.pipeline import load_pipeline

load_dotenv()

pipeline_dir = os.getenv("PIPELINE_DIR")
mock_ex_dir = os.getenv("MOCK_EX_DIR")

pipeline = load_pipeline(pipeline_dir)
symbols = pipeline.get_symbols()


# First step: create a mock exchange which we then use to test the system.
config = MockExchangeSettings(
    object_id=ObjectId(value="mock_ccxt_exchange"),
    symbols=symbols,
    data_directory=mock_ex_dir,
    scrape_start_date=pd.Timestamp("2020-12-01T00:00:00+00:00"),
    scrape_end_date=None,
    simulation_start_date=pd.Timestamp("2021-01-01T00:00:00+00:00"),
    simulation_end_date=pd.Timestamp("2024-10-01T00:00:00+00:00"),
    timeframe=pd.Timedelta("1h"),
    exchange_config={
        "options": {"defaultType": "swap"},
        "timeout": 30000,
        "enableRateLimit": True,
    },
)

mock_exchange = MockExchange(config)


mock_exchange.create_and_save_mock_data()
