"""Concrete implementation of models."""

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import ta
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

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

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model from disk."""

        path = model_path or self.config.training_information.file_path_model

        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info("Model successfully loaded from %s", path)
        else:
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

    def save(self):
        """Save the trained model to disk."""

        model_dir = os.path.join(self.config.data_directory, "models")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self.config.training_information is None:
            raise ValueError("No training information available. Seems like the model has not been trained yet.")

        joblib.dump(self.model, self.config.training_information.file_path_model)
        logger.info("Model saved to %s", self.config.training_information.file_path_model)

    def _train(self, training_data: Data) -> TrainingInformation:
        """Train the model with hyperparameter tuning."""

        logger.info("Start training the model.")

        data = pd.concat([data for data in training_data.data.values()]).sort_index()
        x = data.drop(columns="target")
        y = data["target"]

        # get the training information
        symbols = list(training_data.data.keys())
        train_start_date: pd.Timestamp = min(x.index)
        train_end_date: pd.Timestamp = max(x.index)

        # check whether a trained model already exists for the corresponding train period
        full_model_path = os.path.join(
            self.config.data_directory,
            "models",
            self.config.object_id.value
            + "_"
            + train_start_date.isoformat()
            + "_"
            + train_end_date.isoformat()
            + ".joblib",
        )

        if os.path.exists(full_model_path):
            logger.info("Found an already saved model. Training gets skipped.")
            self.load(full_model_path)
        else:
            # Fit the model with hyperparameter tuning
            self.model.fit(x, y)

            logger.info("Model training complete.")

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

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model from disk."""

        path = model_path or self.config.training_information.file_path_model

        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info("Model successfully loaded from %s", path)
        else:
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

    def save(self):
        """Save the trained model to disk."""

        model_dir = os.path.join(self.config.data_directory, "models")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self.config.training_information is None:
            raise ValueError("No training information available. Seems like the model has not been trained yet.")

        joblib.dump(self.model, self.config.training_information.file_path_model)
        logger.info("Model saved to %s", self.config.training_information.file_path_model)

    def _train(self, training_data: Data) -> TrainingInformation:
        """Train the model with hyperparameter tuning."""

        logger.info("Start training the model.")

        data = pd.concat([data for data in training_data.data.values()]).sort_index()
        x = data.drop(columns="target")
        y = data["target"]

        # get the training information
        symbols = list(training_data.data.keys())
        train_start_date: pd.Timestamp = min(x.index)
        train_end_date: pd.Timestamp = max(x.index)

        # check whether a trained model already exists for the corresponding train period
        full_model_path = os.path.join(
            self.config.data_directory,
            "models",
            self.config.object_id.value
            + "_"
            + train_start_date.isoformat()
            + "_"
            + train_end_date.isoformat()
            + ".joblib",
        )

        if os.path.exists(full_model_path):
            logger.info("Found an already saved model. Training gets skipped.")
            self.load(full_model_path)
        else:
            # Fit the model with hyperparameter tuning
            self.model.fit(x, y)

            logger.info("Model training complete.")

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


class PSARModel(Model):
    """Random Forest Classifier"""

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)

        # in model we store the step information for each symbol.
        self.model = {}

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model from disk."""

        path = model_path or self.config.training_information.file_path_model

        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info("Model successfully loaded from %s", path)
        else:
            logger.error("Model file does not exist at %s", path)
            raise ValueError("Model path does not exist.")

    def save(self):
        """Save the trained model to disk."""

        model_dir = os.path.join(self.config.data_directory, "models")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self.config.training_information is None:
            raise ValueError("No training information available. Seems like the model has not been trained yet.")

        joblib.dump(self.model, self.config.training_information.file_path_model)
        logger.info("Model saved to %s", self.config.training_information.file_path_model)

    def _train(self, training_data: Data) -> TrainingInformation:
        """Train the model with hyperparameter tuning."""

        logger.info("Start training the model.")

        # get the training information
        symbols = list(training_data.data.keys())
        data = pd.concat([data for data in training_data.data.values()]).sort_index()
        train_start_date: pd.Timestamp = min(data.index)
        train_end_date: pd.Timestamp = max(data.index)

        # check whether a trained model already exists for the corresponding train period
        full_model_path = os.path.join(
            self.config.data_directory,
            "models",
            self.config.object_id.value
            + "_"
            + train_start_date.isoformat()
            + "_"
            + train_end_date.isoformat()
            + ".joblib",
        )

        if os.path.exists(full_model_path):
            logger.info("Found an already saved model. Training gets skipped.")
            self.load(full_model_path)
        else:
            for symbol, df in training_data.data.items():

                df = df.copy()

                result = self.find_best_step(df)

                self.model[symbol] = result

            logger.info("Model training complete.")

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

            psar = ta.trend.PSARIndicator(
                high=df["high"], low=df["low"], close=df["close"], step=self.model[symbol]["step"]
            )

            y_pred = psar.psar_up() > 0

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

    def find_best_step(self, df: pd.DataFrame) -> Dict[str, float]:
        """Function to determine the best step value for psar indicator."""

        best_sr = -np.inf
        best_sr_return = -np.inf
        best_step = -np.inf

        for step in tqdm(np.linspace(0.001, 1, 1000)):

            step = step.item()

            psar = ta.trend.PSARIndicator(high=df["high"], low=df["low"], close=df["close"], step=step)

            # psar_up = psar.psar_up_indicator().astype(bool)
            psar_up = psar.psar_up() > 0
            ret = ((df["close"].shift(-1, freq="4h") - df["close"]) / df["close"]).rename("return")
            res_df = pd.concat([df["close"], ret, psar_up], axis=1)

            res_df = res_df.dropna()

            return_list = []
            buy_price = None

            for i in range(len(res_df)):

                current_price, ret, action = res_df.iloc[i]

                # get in a new trade
                if action:
                    if not buy_price:
                        buy_price = current_price
                # otherwise close the trade
                else:
                    # get out of trade if action says no trade
                    if buy_price:
                        return_list.append(((current_price - buy_price) / buy_price).item())
                        buy_price = None

            if len(return_list) != 0:
                return_array = np.array(return_list)

                ret = (1 + return_array).cumprod()[-1]
                sr = self.calculate_sharpe_ratio(return_array)

                if sr > best_sr:
                    best_sr = sr
                    best_step = step
                    best_sr_return = ret

        return {"step": best_step, "sharpe_ratio": sr, "return": best_sr_return}

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0) -> float:
        """
        Calculate the Sharpe Ratio for a series of returns.

        Parameters:
        returns (np.array): Array of returns.
        risk_free_rate (float): Risk-free rate of return, default is 0.

        Returns:
        float: Sharpe ratio.
        """
        # Calculate the average return
        avg_return = np.mean(returns)

        # Calculate the standard deviation of the returns
        return_std = np.std(returns)

        # Calculate the Sharpe ratio
        sharpe_ratio = (avg_return - risk_free_rate) / return_std

        return sharpe_ratio
