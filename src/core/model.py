"""Core module to define model logic."""

import os
from abc import abstractmethod
from typing import List, Literal, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from pydantic.config import ConfigDict

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.datasource import Data


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
    timeframe: Optional[pd.Timedelta] = None
    prediction_type: Literal["regression", "direction", "volatility"]
    training_information: Optional[TrainingInformation] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelConfiguration(BaseConfiguration):
    """Configuration specific to a model."""

    component_type: str = Field(
        default="model",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: Union[ModelSettings, Type[ModelSettings]]


class Prediction(BaseModel):
    """
    Base class for predictions, holding common information about the model output.

    Attributes:
        symbol (str): The symbol or asset for which the prediction is made.
        object_ref (ObjectId): The object ID of the model used for the prediction.
        time (pd.Timestamp): The timestamp of the prediction.
        close (Optional[float]): The current price of the asset, used for evaluation or display.
        atr (Optional[float]): The Average True Range (ATR), used for stop loss/take profit calculation.
        confidence (Optional[float]): Confidence score of the prediction (if applicable).
        prediction_type (Literal["regression", "direction", "volatility"]): Specifies the type of prediction:

        - **"regression"**:
          - The prediction is a continuous float value, representing a numerical outcome such as the predicted return.
        - **"direction"**:
          - The prediction is an integer value from {-1, 0, 1}, representing the directional action:
            - -1: Sell
            - 0: Hold (do nothing)
            - 1: Buy
        - **"volatility"**:
          - The prediction is a binary integer (0 or 1), representing volatility classification:
            - 0: Low volatility
            - 1: High volatility

        prediction (Union[int, float]): The prediction value, which depends on the prediction type:
        - For **regression**, it is a float value.
        - For **direction**, it is an integer {-1, 0, 1}.
        - For **volatility**, it is a binary integer {0, 1}.

        ground_truth (Optional[Union[int, float]]): The actual observed value for the prediction type (optional).
        - For **regression**, it's a float representing the true return.
        - For **direction**, it's an integer {-1, 0, 1}.
        - For **volatility**, it's a binary integer {0, 1}.

    Raises:
        ValueError: If the prediction value does not match the constraints based on the prediction type.
    """

    symbol: str
    object_ref: ObjectId  # object id of the model on which the prediction is based
    time: pd.Timestamp
    close: Optional[float] = None  # current price
    atr: Optional[float] = None  # current ATR value, used for stop loss/take profit
    confidence: Optional[float] = None  # Optional confidence score
    prediction_type: Literal["regression", "direction", "volatility"]
    prediction: Union[int, float]
    ground_truth: Optional[Union[int, float]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_ground_truth_in_depends_on(cls, values):  # pylint: disable=no-self-argument
        """Validate the predictions and ground truths based on prediction_type."""

        if values.prediction_type == "regression":
            if not isinstance(values.prediction, float):
                raise ValueError("For regression, prediction must be a float.")
            if values.ground_truth is not None and not isinstance(values.ground_truth, float):
                raise ValueError("For regression, ground_truth must be a float.")

        elif values.prediction_type == "direction":
            if values.prediction not in [-1, 0, 1]:
                raise ValueError("For direction, prediction must be -1, 0, or 1.")
            if values.ground_truth is not None and values.ground_truth not in [-1, 0, 1]:
                raise ValueError("For direction, ground_truth must be -1, 0, or 1.")

        elif values.prediction_type == "volatility":
            if values.prediction not in [0, 1]:
                raise ValueError("For volatility, prediction must be 0 or 1.")
            if values.ground_truth is not None and values.ground_truth not in [-1, 0, 1]:
                raise ValueError("For volatility, ground_truth must be 0 or 1.")
        return values


logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


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
    def load(self, model_path: Optional[str] = None) -> None:
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

        # check if references are correct
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
