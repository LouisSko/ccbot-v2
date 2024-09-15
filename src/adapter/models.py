"""Concrete implementation of models."""

import os
from typing import List

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from src.common_packages import create_logger
from src.core.datasource import Data
from src.core.model import Model, ModelSettings, Prediction, TrainingInformation

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

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)
        self.model = LGBMClassifier(boosting_type="gbdt", n_jobs=-1, verbose=-1)

    def load(self) -> None:
        """Load the model from disk."""

        path = self.config.save_path_model

        if not os.path.exists(path):
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

        self.model = joblib.load(path)
        logger.info("Model successfully loaded from %s", path)

    def save(self):
        """Here the model gest loaded"""

        path = self.config.save_path_model
        # Get the directory from the model path
        model_dir = os.path.dirname(path)

        # Create the directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def _train(self, training_data: Data) -> TrainingInformation:
        """Train the model with hyperparameter tuning."""

        logger.info("Start training the model.")

        data = pd.concat([data for data in training_data.data.values()]).sort_index()
        x = data.drop(columns="target")
        y = data["target"]

        # Fit the model with hyperparameter tuning
        self.model.fit(x, y)

        logger.info("Model training complete.")

        # get the training information
        symbols = list(training_data.data.keys())
        train_start_date = min(x.index)
        train_end_date = max(x.index)

        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions."""

        result = []

        for symbol, df in prediction_data.data.items():

            if "target" in df.columns:
                df = df.drop(columns="target")

            y_pred = self.model.predict(df)
            # y_pred = y_pred.reshape(-1, 1)

            # TODO: maybe add a date infromation
            result.append(Prediction(symbol=symbol, predictions=y_pred))

        return List[Prediction](data=result)
