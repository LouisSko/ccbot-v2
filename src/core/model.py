"""Core module to define model logic."""

import os
from abc import abstractmethod
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.datasource import Data


class TrainingInformation(BaseModel):
    """Symbols on which the model got trained."""

    symbols: List[str]
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelSettings(BaseModel):
    """Parameters for initializing the model."""

    object_id: ObjectId
    depends_on: ObjectId  # ID of the processor this model depends on.
    data_directory: str  # directory where the data is getting saved
    file_name_model: str  # file name of the model
    training_information: Optional[TrainingInformation] = None


class ModelConfiguration(BaseConfiguration):
    """Configuration specific to a model."""

    component_type: str = Field(
        default="model",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: ModelSettings


class Prediction(BaseModel):
    """Result of the model predictions.

    For each symbol and timestamp only a single prediction can be made
    """

    symbol: str
    time: List[pd.Timestamp]
    prediction: List[int]
    object_ref: ObjectId  # object id of the model on which the prediction is based

    model_config = ConfigDict(arbitrary_types_allowed=True)


logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


# TODO: maybe also add settings here in the same way as for the datasource
class Model(BasePipelineComponent):
    """Abstract base class for machine learning models."""

    def __init__(self, config: ModelSettings) -> None:
        """Initialize the BaseModel with configuration settings."""

        self.config = config
        if os.path.exists(os.path.join(self.config.data_directory, self.config.file_name_model)):
            self.load()

    @abstractmethod
    def load(self) -> None:
        """Load the model from disk."""

    @abstractmethod
    def save(self) -> None:
        """Save the trained model to disk."""

    @abstractmethod
    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions."""

        pass

    @abstractmethod
    def _train(self, training_data: Data) -> TrainingInformation:
        """Train the model.

        This method must be implemented by subclasses to define the training process.
        """

        pass

    def predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions."""

        # if prediction_data.object_ref != self.config.depends_on:
        #    raise ValueError("Object reference is not correct. Set 'depends_on' in the 'ModelSettings' accordingly.")

        return self._predict(prediction_data)

    def train(self, training_data: Data) -> None:
        """Train and save the model"""

        logger.info("Start training process...")

        if training_data.object_ref != self.config.depends_on:
            raise ValueError(
                f"Object reference is not correct. depends_on: '{self.config.depends_on}', '{training_data.object_ref}' got selected. but Set 'depends_on' in the 'ModelSettings' accordingly."
            )

        self.config.training_information = self._train(training_data)

        logger.info("Training process completed.")

    def create_configuration(self) -> ModelConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{ModelConfiguration.__module__}.{ModelConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return ModelConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )
