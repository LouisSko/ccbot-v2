"""Configures a Pipeline and loads mock data."""

import os
import sys

# TODO fix this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import pandas as pd
from dotenv import load_dotenv

from src.adapter.datasources import (
    ExchangeDatasource,
    ExchangeDatasourceSettings,
    FearGreedDataSource,
    FearGreedDatasourceSettings,
)
from src.adapter.models import LgbmDartClf, LgbmGbrtClf, ModelSettings, RfClf
from src.adapter.processors import (
    DataProcessorSettings,
    FeaturesExchange,
    FeaturesFearGreed,
    TargetSettings,
    TargetUpDownNo,
    TargetVolatility,
)
from src.adapter.signal_generators import SignalGenerator
from src.core.base import ObjectId
from src.core.ensemble import EnsembleModel, EnsembleSettings
from src.core.generator import GeneratorSettings
from src.core.pipeline import Pipeline, PipelineSettings
from src.core.processor import DataMerger, DataMergerSettings

load_dotenv()

pipeline_dir = os.getenv("PIPELINE_DIR")
mock_ex_dir = os.getenv("MOCK_EX_DIR")
timeframe = pd.Timedelta("4h")
test_interval_length = pd.Timedelta("60d")

scrape_start_date = pd.Timestamp("2020-01-01T00:00:00+00:00")
simulation_start_date = pd.Timestamp("2022-01-01T00:00:00+00:00")
simulation_end_date = pd.Timestamp("2024-10-01T00:00:00+00:00")
# symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
symbols = None # this determines the symbols automatically


def main():
    """defines a pipeline, exports it and creates mock data."""

    #### Datasources ####

    config = ExchangeDatasourceSettings(
        object_id=ObjectId(value="exchange_datasource"),
        exchange_id="bitget",
        symbols=symbols,
        data_directory=pipeline_dir,
        scrape_start_date=scrape_start_date,
        scrape_end_date=None,
        simulation_start_date=simulation_start_date,
        simulation_end_date=simulation_end_date,
        timeframe=timeframe,
    )

    exchange_datasource = ExchangeDatasource(config=config)

    config = FearGreedDatasourceSettings(
        object_id=ObjectId(value="fear_greed_datasource"),
        data_directory=pipeline_dir,
        scrape_start_date=scrape_start_date,
        scrape_end_date=None,
        simulation_start_date=simulation_start_date,
        simulation_end_date=simulation_end_date,
        timeframe=timeframe,
    )

    fear_greed_datasource = FearGreedDataSource(config=config)

    #### Processors ####

    config = DataProcessorSettings(
        object_id=ObjectId(value="feature_processor"),
        depends_on=ObjectId(value="exchange_datasource"),
        timeframe=timeframe,
    )

    feature_processor = FeaturesExchange(config=config)

    config = DataProcessorSettings(
        object_id=ObjectId(value="fear_greed_processor"),
        depends_on=ObjectId(value="fear_greed_datasource"),
        timeframe=timeframe,
    )

    fear_greed_processor = FeaturesFearGreed(config=config)

    config = TargetSettings(
        object_id=ObjectId(value="target_processor_dir"),
        depends_on=ObjectId(value="exchange_datasource"),
        timeframe=timeframe,
        target_value=0.002,
    )

    target_processor_dir = TargetUpDownNo(config=config)

    config = TargetSettings(
        object_id=ObjectId(value="target_processor_vola"),
        depends_on=ObjectId(value="exchange_datasource"),
        timeframe=timeframe,
        target_value=0.006,
    )

    target_processor_vola = TargetVolatility(config=config)

    #### Data Merger ####

    config = DataMergerSettings(
        object_id=ObjectId(value="merger_dir"),
        depends_on=[
            ObjectId(value="feature_processor"),
            ObjectId(value="fear_greed_processor"),
            ObjectId(value="target_processor_dir"),
        ],
    )

    merger_dir = DataMerger(config=config)

    config = DataMergerSettings(
        object_id=ObjectId(value="merger_vola"),
        depends_on=[
            ObjectId(value="feature_processor"),
            ObjectId(value="fear_greed_processor"),
            ObjectId(value="target_processor_vola"),
        ],
    )

    merger_vola = DataMerger(config=config)

    #### Models ####

    config = ModelSettings(
        object_id=ObjectId(value="Model_DART_dir"),
        depends_on=ObjectId(value="merger_dir"),
        data_directory=pipeline_dir,
        prediction_type="direction",
    )

    dart_model_dir = LgbmDartClf(config=config)

    config = ModelSettings(
        object_id=ObjectId(value="Model_GBRT_dir"),
        depends_on=ObjectId(value="merger_dir"),
        data_directory=pipeline_dir,
        prediction_type="direction",
    )

    gbrt_model_dir = LgbmGbrtClf(config=config)

    config = ModelSettings(
        object_id=ObjectId(value="Model_RF_dir"),
        depends_on=ObjectId(value="merger_dir"),
        data_directory=pipeline_dir,
        prediction_type="direction",
    )

    rf_model_dir = RfClf(config=config)

    config = ModelSettings(
        object_id=ObjectId(value="Model_DART_vola"),
        depends_on=ObjectId(value="merger_vola"),
        data_directory=pipeline_dir,
        prediction_type="volatility",
    )

    dart_model_vola = LgbmDartClf(config=config)

    config = ModelSettings(
        object_id=ObjectId(value="Model_GBRT_vola"),
        depends_on=ObjectId(value="merger_vola"),
        data_directory=pipeline_dir,
        prediction_type="volatility",
    )

    gbrt_model_vola = LgbmGbrtClf(config=config)

    config = ModelSettings(
        object_id=ObjectId(value="Model_RF_vola"),
        depends_on=ObjectId(value="merger_vola"),
        data_directory=pipeline_dir,
        prediction_type="volatility",
    )

    rf_model_vola = RfClf(config=config)

    #### Ensemble ####

    config = EnsembleSettings(
        object_id=ObjectId(value="ensemble_dir"),
        depends_on=[
            ObjectId(value="Model_RF_dir"),
            ObjectId(value="Model_GBRT_dir"),
            ObjectId(value="Model_DART_dir"),
        ],
        ground_truth_object_ref=ObjectId(value="Model_RF_dir"),
        prediction_type="direction",
        agreement_type_clf="voting",
        data_directory=pipeline_dir,
    )

    ensemble_model_dir = EnsembleModel(config=config)

    config = EnsembleSettings(
        object_id=ObjectId(value="ensemble_vola"),
        depends_on=[
            ObjectId(value="Model_RF_vola"),
            ObjectId(value="Model_GBRT_vola"),
            ObjectId(value="Model_DART_vola"),
        ],
        ground_truth_object_ref=ObjectId(value="Model_RF_vola"),
        prediction_type="volatility",
        agreement_type_clf="voting",
        data_directory=pipeline_dir,
    )

    ensemble_model_vola = EnsembleModel(config=config)

    #### Trade Signal Generator ####

    config = GeneratorSettings(
        object_id=ObjectId(value="signal_generator"),
        depends_on=[ObjectId(value="ensemble_dir"), ObjectId(value="ensemble_vola")],
        order_type="limit",
    )

    generator = SignalGenerator(config=config)

    # create pipeline
    pipeline = Pipeline(
        PipelineSettings(
            components=[
                exchange_datasource,
                fear_greed_datasource,
                feature_processor,
                fear_greed_processor,
                target_processor_dir,
                merger_dir,
                dart_model_dir,
                gbrt_model_dir,
                rf_model_dir,
                ensemble_model_dir,
                target_processor_vola,
                merger_vola,
                dart_model_vola,
                gbrt_model_vola,
                rf_model_vola,
                ensemble_model_vola,
                generator,
            ],
            save_dir=pipeline_dir,
            timeframe=timeframe,
            test_interval_length=test_interval_length,
        )
    )

    pipeline.export_pipeline()
    pipeline.create_and_save_mock_data()
    # pipeline.activate_simulation_mode()
    # pipeline.train()


if __name__ == "__main__":

    main()
