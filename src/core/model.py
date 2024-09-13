from abc import abstractmethod
from importlib import import_module
from typing import List, Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId


class TrainingInformation(BaseModel):
    """Symbols on which the model got trained."""

    symbols: List[str]
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    save_path: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseModelConfiguration(BaseConfiguration):
    """Configuration specific to a model."""

    config_type: str = Field(
        default="model",
        Literal=True,
        description="Type of the configuration (e.g., model, preprocessor). Do not change this value",
    )
    depends_on: ObjectId = Field(description="ID of the preprocessor this model depends on.")
    training_information: Optional[TrainingInformation]


class TrainingDataSymbol(BaseModel):
    """Training data of a single symbol."""

    symbol: str
    features: pd.DataFrame
    target: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PredictionDataSymbol(BaseModel):
    """Data used for making predictions"""

    symbol: str
    features: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainingData(BaseModel):
    """Training Data for a model"""

    data: List[TrainingDataSymbol]


class PredictionData(BaseModel):
    """Data used for making predictions"""

    data: List[PredictionDataSymbol]


class PredictionResultSymbol(BaseModel):
    """Result of the prediction"""

    symbol: str
    predictions: np.array

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PredictionResult(BaseModel):
    """Result of the prediction"""

    data: List[PredictionResultSymbol]


class Model(BasePipelineComponent):
    """Abstract base class for machine learning models."""

    def __init__(
        self, object_id: ObjectId, depends_on: ObjectId, training_information: Optional[TrainingInformation] = None
    ) -> None:
        """Initialize the BaseModel with configuration settings."""

        self.object_id = object_id
        self.depends_on = depends_on
        self.training_information = training_information

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model from disk."""

    @abstractmethod
    def _save_model(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path (Optional[str]): Optionally provide a full path to save the model.
        """

    @abstractmethod
    def _train(self, training_data: TrainingData) -> TrainingInformation:
        """Train the model.

        This method must be implemented by subclasses to define the training process.
        """

        pass

    @abstractmethod
    def predict(self, prediction_data: PredictionData) -> PredictionResult:
        """Function to make predictions."""

        pass

    @classmethod
    def load_from_configuration(cls, config: BaseModelConfiguration) -> "Model":
        """Create an instance of the model based on the provided configuration.

        Args:
            config (BaseModelConfiguration): The configuration object to create the model instance.

        Returns:
            Model: An instance of the model class.

        Usage:
            config = BaseModelConfiguration(...)  # Assume this is loaded from JSON or another source
            model_instance = Model.from_configuration(config)

        """
        module_path, class_name = config.resource_path.rsplit(".", 1)
        module = import_module(module_path)
        model_class: Type[Model] = getattr(module, class_name)

        # Initialize the model with parameters from configuration
        model_instance = model_class(
            object_id=config.object_id,
            depends_on=config.depends_on,
            training_information=config.training_information,
        )

        return model_instance

    def create_configuration(self) -> BaseModelConfiguration:
        """Returns the configuration of the class.

        Args:
            training_information (Optional[TrainingInformation]): The training information.

        Returns:
            BaseModelConfiguration: The configuration of the model.
        """

        full_path = f"{self.__module__}.{self.__class__.__name__}"

        return BaseModelConfiguration(
            object_id=self.object_id,
            resource_path=full_path,
            depends_on=self.depends_on,
            training_information=self.training_information,
        )

    def train_and_save(self, training_data: TrainingData, path=str) -> None:
        """Train and save the model"""

        self.training_information = self._train(training_data)

        self._save_model(path)
