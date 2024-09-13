"""Concrete implementation of models."""

import os
from typing import Optional

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from src.common_packages import create_logger
from src.core.base import ObjectId
from src.core.model import (
    Model,
    PredictionData,
    PredictionResult,
    PredictionResultSymbol,
    TrainingData,
    TrainingInformation,
)

# instantiate logger
logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class LgbmModelClf(Model):
    """LGBM Regression Model

    with:
        - hyperparameter tuning using RandomizedSearchCV
        - blockingtimeseries splits
    """

    def __init__(
        self, object_id: ObjectId, depends_on: ObjectId, training_information: Optional[TrainingInformation] = None
    ) -> None:
        super().__init__(object_id, depends_on, training_information)
        self.model = LGBMClassifier(boosting_type="gbdt", n_jobs=-1, verbose=-1)

    def load_model(self, path: str) -> None:
        """Load the model from disk."""

        if not os.path.exists(path):
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

        self.model = joblib.load(path)
        logger.info("Model successfully loaded from %s", path)

    def _save_model(self, path: str):
        """Here the model gest loaded"""

        if not path:
            raise ValueError("Model path must be specified to save the model.")

        # Get the directory from the model path
        model_dir = os.path.dirname(path)

        # Create the directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def _train(self, training_data: TrainingData) -> TrainingInformation:
        """Train the model with hyperparameter tuning."""

        logger.info("Start hyperparameter tuning with RandomizedSearchCV...")

        # TODO: need to sort them in the right order
        # bring the data in the right shape
        x = pd.concat([data.features for data in training_data])
        y = pd.concat([data.target for data in training_data])

        # Fit the model with hyperparameter tuning
        self.model.fit(x, y)

        logger.info("Model training complete.")

        # get the training information
        symbols = [data.symbols for data in training_data]
        train_start_date = min(x.index)
        train_end_date = max(x.index)

        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
        )

    def predict(self, prediction_data: PredictionData) -> PredictionResult:
        """Function to make predictions."""

        result = []

        for prediction_data_symbol in prediction_data:

            y = self.model.predict(prediction_data_symbol.features)
            y = y.reshape(-1, 1)

            result.append(PredictionResultSymbol(symbol=prediction_data_symbol.symbol, predictions=y))

        return PredictionData(data=result)
