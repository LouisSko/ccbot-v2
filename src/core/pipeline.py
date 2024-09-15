"""Module which creates the pipeline."""

import json
from abc import abstractmethod
from importlib import import_module
from typing import List, Optional, Type

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.common_packages import CustomJSONEncoder
from src.core.base import BaseConfiguration, BasePipelineComponent
from src.core.model import Model, ModelConfiguration
from src.core.processor import DataProcessorConfiguration, MergerConfiguration


class PipelineConfig(BaseModel):
    """Create a Pipeline based on a list of BaseConfiguration"""

    pipeline: List[BaseConfiguration]


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

    def _initialize_components(self, config: PipelineConfig):
        """Initialize all pipeline components."""

        components = {}
        for component in config.pipeline:
            components[component.object_id.value] = {
                "instance": load_component_from_configuration(component),
                "config": component,
            }

        return components

    def export_component_configs(self, file_path: str) -> None:
        """Export Pipeline component configurations to a json file.

        Args:
            pipeline (Pipeline): The Pipeline object to export.
            file_path (str): The path to the YAML file.
        """

        with open(file_path, "w", encoding="utf8") as file:
            # Convert list of BaseConfiguration objects to a list of dictionaries
            data = [config.model_dump() for config in self.config.pipeline]

            # yaml.dump(data, file, default_flow_style=False)
            json.dump(data, file, indent=4, cls=CustomJSONEncoder)

    def load_component_configs(self, file_path: str) -> PipelineConfig:
        """load Pipeline configu based on a stored pipeline component configuration json"""

        with open(file_path, "r", encoding="utf8") as f:
            config = json.load(f, object_hook=timestamp_decoder)

        config = [configuration_factory(config_dict) for config_dict in config]

        return config

    @abstractmethod
    def train(self):
        """Train all models in the pipeline."""

        pass

    @abstractmethod
    def trigger(self) -> Trades:
        """Triggers the pipeline."""

        pass


def load_pipeline(file_path: str) -> Pipeline:
    """Create a pipeline based on a stored pipeline component configuration json
    Args:
        file_path (str): File directory of the configuration.
    """

    with open(file_path, "r", encoding="utf8") as f:
        config = json.load(f, object_hook=timestamp_decoder)

    config = PipelineConfig(pipeline=[configuration_factory(config_dict) for config_dict in config])

    return Pipeline(config=config)


def timestamp_decoder(obj: dict):
    """Convert strings back to pd.Timestamp during JSON decoding."""

    for key, value in obj.items():
        if isinstance(value, str):
            try:
                # Try to parse the string as a timestamp
                obj[key] = pd.Timestamp(value)
            except ValueError:
                # If it fails, leave it as a string
                pass
    return obj


def configuration_factory(config_dict: dict) -> BaseConfiguration:
    """Factory method to create the appropriate configuration object.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        BaseConfiguration: The appropriate configuration object.
    """
    
    config_type = config_dict.get("config_type")

    if config_type == "model":
        return TypeAdapter(ModelConfiguration).validate_python(config_dict)
    if config_type == "processor":
        return TypeAdapter(DataProcessorConfiguration).validate_python(config_dict)
    if config_type == "merger":
        return TypeAdapter(MergerConfiguration).validate_python(config_dict)

    raise ValueError(f"Unknown config_type: {config_type}")


def load_component_from_configuration(config: BaseConfiguration) -> BasePipelineComponent:
    """Create an instance of a PipelineComponent based on the provided configuration.

    Args:
        config (ModelConfiguration): The configuration object to create the model instance.
    """

    module_path, class_name = config.resource_path.rsplit(".", 1)
    module = import_module(module_path)
    model_class: Type[BasePipelineComponent] = getattr(module, class_name)

    if issubclass(model_class, Model):
        # Initialize the model with parameters from configuration
        model_instance = model_class(
            object_id=config.object_id,
            depends_on=config.depends_on,
            training_information=config.training_information,
        )

    else:
        return None

    return model_instance
