"""Concrete implementation of models."""

import os
from typing import List

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from src.common_packages import create_logger
from src.core.datasource import Data
from src.core.model import Model, ModelSettings, Prediction, TrainingInformation

# instantiate logger
logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class LgbmClf(Model):
    """LGBM Classification Model"""

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)

        # create model if it does not exist
        if not hasattr(self, "model") or self.model is None:
            self.model = LGBMClassifier(boosting_type="gbdt", n_jobs=-1, verbose=-1)

    def load(self) -> None:
        """Load the model from disk."""

        path = self.config.training_information.file_path_model

        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info("Model successfully loaded from %s", path)
        else:
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

    def save(self):
        """Save the trained model to disk."""

        # Create the directory if it doesn't exist
        if not os.path.exists(self.config.data_directory):
            os.makedirs(self.config.data_directory)

        full_model_path = os.path.join(self.config.data_directory, self.config.object_id.value + ".joblib")

        joblib.dump(self.model, full_model_path)
        logger.info("Model saved to %s", full_model_path)

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

        full_model_path = os.path.join(self.config.data_directory, self.config.object_id.value + ".joblib")

        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions."""

        predictions = []
        ground_truth = None

        for symbol, df in prediction_data.data.items():

            if "target" in df.columns:
                ground_truth = df["target"].to_list()
                df = df.drop(columns="target")

            y_pred = self.model.predict(df)

            time = list(df.index)

            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=list(y_pred),
                    ground_truth=ground_truth,
                    time=time,
                )
            )

        return predictions


class RfClf(Model):
    """Random Forest Classifier"""

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)

        # create model if it does not exist
        if not hasattr(self, "model") or self.model is None:
            self.model = RandomForestClassifier()

    def load(self) -> None:
        """Load the model from disk."""

        path = self.config.training_information.file_path_model

        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info("Model successfully loaded from %s", path)
        else:
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

    def save(self):
        """Save the trained model to disk."""

        # Create the directory if it doesn't exist
        if not os.path.exists(self.config.data_directory):
            os.makedirs(self.config.data_directory)

        full_model_path = os.path.join(self.config.data_directory, self.config.object_id.value + ".joblib")

        joblib.dump(self.model, full_model_path)
        logger.info("Model saved to %s", full_model_path)

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

        full_model_path = os.path.join(self.config.data_directory, self.config.object_id.value + ".joblib")

        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions."""

        predictions = []
        ground_truth = None

        for symbol, df in prediction_data.data.items():

            if "target" in df.columns:
                ground_truth = df["target"].to_list()
                df = df.drop(columns="target")

            y_pred = self.model.predict(df)

            time = list(df.index)

            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=list(y_pred),
                    ground_truth=ground_truth,
                    time=time,
                )
            )

        return predictions
