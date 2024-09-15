"""Module which creates the pipeline."""

import json
import os
from abc import abstractmethod
from importlib import import_module
from typing import List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.common_packages import CustomJSONEncoder, create_logger, timestamp_decoder
from src.core.base import BaseConfiguration, BasePipelineComponent

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class PipelineConfig(BaseModel):
    """Create a Pipeline based on a list of BaseConfiguration"""

    pipeline: Union[List[BaseConfiguration], List[BasePipelineComponent]]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Trade(BaseModel):
    """Represents a single trade."""

    time: pd.Timestamp = Field(..., description="Timestamp of the trade")
    symbol: str = Field(..., description="Trading symbol, e.g., 'BTC/USDT'")
    position_side: str = Field(..., description="Type of position: either 'buy' or 'sell'")
    order_type: str = Field(..., description="Type of order: either 'limit' or 'market'")
    limit_price: Optional[float] = Field(None, description="Limit price for the trade (optional for market orders)")
    stop_loss_price: Optional[float] = Field(None, description="Stop loss price (optional)")
    take_profit_price: Optional[float] = Field(None, description="Take profit price (optional)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Trades(BaseModel):
    """Create Trades"""

    data: List[Trade]


class Pipeline:
    """Pipeline class"""

    def __init__(self, config: PipelineConfig):

        self.config = config
        self.components = self._initialize_components(self.config)

    @abstractmethod
    def train(self):
        """Train all models in the pipeline."""

        pass

    @abstractmethod
    def trigger(self) -> Trades:
        """Triggers the pipeline."""

        pass

    # TODO: needs to get implemented
    def create_and_save_mock_data(self):
        """Function to create mock data and to store it in the config directory based on datasources."""

        # logger.info("create mock data for all datasources.")

        pass

    def export_pipeline(self, save_directory: str, json_file_name: str) -> None:
        """Export Pipeline component configurations to a json file.

        Args:
            save_directory (str): The path of the directory.
            json_file_name (str): The path of the json file.
        """

        file_path = os.path.join(save_directory, json_file_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        data = []

        # add pipeline information
        pipeline_cls = f"{self.__module__}.{self.__class__.__name__}"
        data.append({"pipeline_path": pipeline_cls})

        for component in self.config.pipeline:

            if isinstance(component, BasePipelineComponent):
                data.append(component.create_configuration().model_dump())

            elif isinstance(component, BaseConfiguration):
                data.append(component.model_dump())

        with open(file_path, "w", encoding="utf8") as file:
            json.dump(data, file, indent=4, cls=CustomJSONEncoder)

    def _initialize_components(self, config: PipelineConfig):
        """Initialize all pipeline components."""

        components = {}
        for component in config.pipeline:

            if isinstance(component, BasePipelineComponent):
                components[component.config.object_id.value] = {
                    "instance": component,
                    "config": component.create_configuration(),
                }

            elif isinstance(component, BaseConfiguration):
                components[component.object_id.value] = {
                    "instance": _load_component_from_configuration(component),
                    "config": component,
                }

            else:
                raise ValueError("component needs to be of type 'BasePipelineComponent' or 'BaseConfiguration'")

        logger.info("Pipeline initalized with the following components: %s", components)

        return components


def load_pipeline(save_directory: str, json_file_name: str) -> Pipeline:
    """Create a pipeline based on a stored pipeline component configuration json
    Args:
        save_directory (str): The path of the directory.
        json_file_name (str): The path of the json file.
    """

    file_path = os.path.join(save_directory, json_file_name)

    with open(file_path, "r", encoding="utf8") as f:
        config = json.load(f, object_hook=timestamp_decoder)

    # first entry of the config is general information regarding the pipeline
    module_path, class_name = config.pop(0)["pipeline_path"].rsplit(".", 1)
    module = import_module(module_path)
    pipeline: Pipeline = getattr(module, class_name)

    config = PipelineConfig(pipeline=[_configuration_factory(config_dict) for config_dict in config])

    return pipeline(config=config)


def _configuration_factory(config_dict: dict) -> BaseConfiguration:
    """Factory method to create the appropriate configuration object.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        BaseConfiguration: The appropriate configuration object.
    """

    # create settings object
    module_path, class_name = config_dict.get("settings_path").rsplit(".", 1)
    module = import_module(module_path)
    component_settings_cls = getattr(module, class_name)
    config_dict["settings"] = TypeAdapter(component_settings_cls).validate_python(config_dict.get("settings"))

    # create config object
    module_path, class_name = config_dict.get("config_path").rsplit(".", 1)
    module = import_module(module_path)
    component_config_cls: Type[BaseConfiguration] = getattr(module, class_name)

    return TypeAdapter(component_config_cls).validate_python(config_dict)


def _load_component_from_configuration(config: BaseConfiguration) -> BasePipelineComponent:
    """Create an instance of a PipelineComponent based on the provided configuration.

    Args:
        config (ModelConfiguration): The configuration object to create the model instance.
    """

    # load the correct class based on the config, such as a datasource or a processor
    module_path, class_name = config.resource_path.rsplit(".", 1)
    module = import_module(module_path)
    pipeline_component_class: Type[BasePipelineComponent] = getattr(module, class_name)

    # create an instance of that class
    instance = pipeline_component_class(config.settings)

    return instance
