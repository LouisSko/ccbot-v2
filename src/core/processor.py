"""DataProcessor module"""

import os
from abc import abstractmethod
from typing import List, Type, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.datasource import Data

# instantiate logger
logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class DataProcessorSettings(BaseModel):
    """Configuration settings for the processor."""

    object_id: ObjectId
    depends_on: ObjectId = Field(description="ID(s) of the datasource or processors this processor depends on.")
    timeframe: pd.Timedelta

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataProcessorConfiguration(BaseConfiguration):
    """Configuration for processors that are classifiers with init parameters."""

    component_type: str = Field(
        default="processor",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )
    settings: Union[DataProcessorSettings, Type[DataProcessorSettings]]


class DataMergerSettings(BaseModel):
    """Configuration settings for the data merger."""

    object_id: ObjectId
    depends_on: List[ObjectId]


class MergerConfiguration(BaseConfiguration):
    """Configuration for a merger that combines the output of multiple processors."""

    component_type: str = Field(
        default="merger",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: DataMergerSettings


class TargetGenerator(BasePipelineComponent):
    """Abstract class for generating target variable.

    The name of the target variable needs to be 'target'
    """

    def __init__(self, config: DataProcessorSettings):
        self.config = config

    @abstractmethod
    def create_target(self, data: Data) -> Data:
        """Abstract method to create the target variable."""

        pass

    def process(self, data: Data) -> Data:
        """Processes the data."""

        return self.create_target(data)

    def create_configuration(self) -> DataProcessorConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{DataProcessorConfiguration.__module__}.{DataProcessorConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return DataProcessorConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )


class FeatureGenerator(BasePipelineComponent):
    """Abstract class for generating features."""

    def __init__(self, config: DataProcessorSettings):
        self.config = config

    @abstractmethod
    def create_features(self, data: Data) -> Data:
        """Calculates and adds features to the dataframes in the input dictionary. Optionally filters the output based on given timestamps."""

        pass

    def process(self, data: Data) -> Data:
        """Allows the instance to be called as a function."""

        return self.create_features(data)

    def create_configuration(self) -> DataProcessorConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{DataProcessorConfiguration.__module__}.{DataProcessorConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return DataProcessorConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )


class DataMerger(BasePipelineComponent):
    """Class for merging data from different processors."""

    def __init__(self, config: DataMergerSettings):
        self.config = config

    def merge_data(self, data_collection=List[Data]) -> Data:
        """Function to combine the data from different sources."""

        merged_data = {}

        # Collect all unique symbols from all processors
        all_symbols = set()
        all_df_list = []

        for data in data_collection:
            symbols = data.data.keys()
            all_symbols.update(symbol for symbol in symbols if symbol != "all")

            # Collect 'all' entries if they exist
            all_df = data.data.get("all")
            if all_df is not None:
                all_df_list.append(all_df)

        # Loop through each symbol and merge data
        for symbol in all_symbols:
            df_list = []
            for data in data_collection:
                if (data.object_ref in self.config.depends_on) and (symbol in data.data.keys()):
                    df_list.append(data.data[symbol])

            # If 'all' dataframes exist, append them as well
            df_list.extend(all_df_list)

            # Merge the dataframes for the symbol along the columns (axis=1)
            result_df = pd.concat(df_list, join="inner", axis=1)
            # Get rid of duplicate columns
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]

            if not result_df.empty and "close" in result_df.columns:
                merged_data[symbol] = result_df

                logger.info(
                    "Processed Symbol: %s. Number of features: %s. Dataset entries: %s",
                    symbol,
                    len(result_df.columns),
                    len(result_df),
                )

        logger.info("Data combined successfully.")

        return Data(object_ref=self.config.object_id, data=merged_data)

    def create_configuration(self) -> MergerConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{MergerConfiguration.__module__}.{MergerConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return MergerConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )
