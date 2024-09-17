"""Core module to define model logic."""

import json
import os
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.datasource import Data
from src.core.evaluation import calculate_classification_metrics


class TrainingInformation(BaseModel):
    """Symbols on which the model got trained."""

    symbols: List[str]
    file_path_model: str  # file name of the model
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelSettings(BaseModel):
    """Parameters for initializing the model."""

    object_id: ObjectId
    depends_on: ObjectId  # ID of the processor this model depends on.
    data_directory: str  # directory where the data is getting saved
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
    prediction: List[int]  # TODO: might need to change this in the future, because we also want to do regression
    ground_truth: Optional[List[int]] = None
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
        if self.config.training_information is not None:
            logger.info("Found training information. Attempt to load model.")
            if os.path.exists(self.config.training_information.file_path_model):
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

    def evaluate(self, evaluation_data: Data) -> None:
        """Train the model.

        This method must be implemented by subclasses to define the training process.
        """

        logger.info("Start evaluation...")

        all_preds = []
        all_targets = []

        # 1. get predictions
        predictions = self.predict(evaluation_data)

        # 2. extract the predictions in a suitable format
        for prediction in predictions:
            all_targets.extend(prediction.ground_truth)
            all_preds.extend(prediction.prediction)

        # 3. calculate metrics
        metrics = calculate_classification_metrics(np.array(all_targets), np.array(all_preds))

        # 4. store the metrics
        metrics_dir = os.path.join(self.config.data_directory, "metrics")

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            logger.info("Created a new directory for storing evaluation results: %s", metrics_dir)

        file_path = os.path.join(metrics_dir, self.config.object_id.value + ".json")
        with open(file_path, "w", encoding="utf8") as file:
            json.dump(metrics.model_dump(), file, indent=4)

        logger.info("Evaluation results saved to %s", file_path)

    def predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions."""

        # if prediction_data.object_ref != self.config.depends_on:
        #    raise ValueError("Object reference is not correct. Set 'depends_on' in the 'ModelSettings' accordingly.")

        return self._predict(prediction_data)

    def train_and_evaluate(self, data: Data, split_date: pd.Timestamp) -> None:
        """Function to train and test a model.

        Args:
            data (Data): the data for training and evaluation
            split_date (pd.Timestamp): the date at which the data is split into train and test

        Note:
            It might seem a bit cumbersome to provide the split as a timestamp and not like a fraction (e.g. 0.8) or pd.Timedelta (e.g. pd.Timedelta("30d")).
            Hower it helps to ensure that the same data is used for training different models in case we have multiple datasources.
            Those different datasources might have different data resolution which could be problematic.
        """

        # 1. Split data in train and test
        training_data, testing_data = self.train_test_splitter(data=data, split_date=split_date)

        # 2. Train the model
        self.train(training_data)

        # 3. evaluate the model
        self.evaluate(testing_data)

    def train(self, training_data: Data) -> None:
        """Train and save the model"""

        logger.info("Start training process...")

        # check if references are correct
        if training_data.object_ref != self.config.depends_on:
            raise ValueError(
                f"Object reference is not correct. depends_on: '{self.config.depends_on}', '{training_data.object_ref}' got selected. but Set 'depends_on' in the 'ModelSettings' accordingly."
            )

        self.config.training_information = self._train(training_data)

        logger.info("Training process completed.")

    def train_test_splitter(self, data: Data, split_date: pd.Timestamp) -> Tuple[Data, Data]:
        """Splits the data into training and test set."""

        data_train, data_test = {}, {}

        for key, df in data.data.items():
            data_train[key] = df[df.index <= split_date].copy()
            data_test[key] = df[df.index > split_date].copy()

        return Data(object_ref=data.object_ref, data=data_train), Data(object_ref=data.object_ref, data=data_test)

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
