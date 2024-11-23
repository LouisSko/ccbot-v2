"""Concrete implementation of models."""

import os
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.signal import savgol_filter
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from tqdm import tqdm

from src.adapter.processors import IncrementalPSARIndicator
from src.common_packages import create_logger
from src.core.datasource import Data
from src.core.evaluation import adapted_f1_score
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
        """Train the model with hyperparameter tuning.

        Combines all data from training_data, checks for an existing model within the
        same training period, and trains the model if no saved model is found.

        Args:
            training_data (Data): A Data object containing training dataframes for each symbol.

        Returns:
            TrainingInformation: Metadata about the training process, including the symbols,
                                training period, and model file path.
        """

        logger.info("Start training the model.")

        # Concatenate all data and sort by index (time)
        data = pd.concat(training_data.data.values()).sort_index()

        # Separate features (x) and target (y)
        x = data.drop(columns=["target", "close", "atr"])
        y = data["target"]

        # Retrieve symbols and training period
        symbols = list(training_data.data.keys())
        train_start_date = x.index.min()
        train_end_date = x.index.max()

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # Train the model and perform hyperparameter tuning if necessary
            self.model.fit(x, y)
            logger.info("Model training complete.")

        # Return training metadata
        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """
        Generates predictions based on input data using the trained model.

        Args:
            prediction_data (Data): A Data object containing symbol-specific
                                    dataframes for prediction.

        Returns:
            List[Prediction]: A list of Prediction objects containing predictions
                            and associated metadata for each symbol.
        """
        predictions = []

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            df = df.copy()

            # Extract ground truth if present
            if "target" in df.columns:
                ground_truth = df.pop("target")
            else:
                ground_truth = None

            # Extract 'close' and 'atr' columns for further processing
            close = df.pop("close").to_list()
            atr = df.pop("atr").to_list()
            time_stamps = df.index.to_list()

            # Generate predictions
            y_pred_prob = self.model.predict_proba(df)
            predicted_indices = y_pred_prob.argmax(axis=1)
            y_pred = self.model.classes_[predicted_indices]

            # Create a Prediction object for each row of the DataFrame
            for i in range(len(df)):
                # Create and store each individual Prediction object
                predictions.append(
                    Prediction(
                        object_ref=self.config.object_id,
                        symbol=symbol,
                        prediction=y_pred[i],  # Ensure prediction is an integer
                        confidence=dict(zip(self.model.classes_, y_pred_prob[i])),
                        ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                        close=close[i],
                        atr=atr[i],
                        time=time_stamps[i],
                        prediction_type=self.config.prediction_type,
                    )
                )

        return predictions


class LgbmDartClf(Model):
    """LGBM Classification Model"""

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)

        # create model if it does not exist
        if not hasattr(self, "model") or self.model is None:
            self.model = LGBMClassifier(boosting_type="dart", n_jobs=-1, verbose=-1)

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
        """Train the model with hyperparameter tuning.

        Combines all data from training_data, checks for an existing model within the
        same training period, and trains the model if no saved model is found.

        Args:
            training_data (Data): A Data object containing training dataframes for each symbol.

        Returns:
            TrainingInformation: Metadata about the training process, including the symbols,
                                training period, and model file path.
        """

        logger.info("Start training the model.")

        # Concatenate all data and sort by index (time)
        data = pd.concat(training_data.data.values()).sort_index()

        # Separate features (x) and target (y)
        x = data.drop(columns=["target", "close", "atr"])
        y = data["target"]

        # Retrieve symbols and training period
        symbols = list(training_data.data.keys())
        train_start_date = x.index.min()
        train_end_date = x.index.max()

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # Train the model and perform hyperparameter tuning if necessary
            # Define hyperparameters for RandomizedSearch
            param_distributions = {
                "max_depth": np.arange(1, 7),  # Now a continuous range from 1 to 11
                "learning_rate": 10 ** (np.linspace(-4, -1, 100)),  # Continuous range from 10^-5 to 1
                "n_estimators": np.arange(10, 600),  # Continuous range from 50 to 1000
                "reg_lambda": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "reg_alpha": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            }

            # tsv = BlockingTimeSeriesSplit(n_splits=1, test_size=3, margin=0)
            tscv = TimeSeriesSplit(
                n_splits=2,
                test_size=int(len(x) * 0.2),
                max_train_size=int(len(x) * 0.8),
            )

            custom_scorer = make_scorer(adapted_f1_score, greater_is_better=True)

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_distributions,
                n_iter=30,  # Number of parameter settings sampled
                cv=tscv,
                scoring=custom_scorer,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

            logger.info("Start hyperparameter tuning with RandomizedSearchCV...")

            # Fit the model with hyperparameter tuning
            random_search.fit(x, y)

            # Set the best model to self.model
            self.model = random_search.best_estimator_
            # self.model.fit(x, y)

            logger.info("Best parameters found: %s", random_search.best_params_)
            logger.info("Best score achieved: %f", random_search.best_score_)

            logger.info("Model training complete.")

        # Return training metadata
        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Generates predictions based on input data using the trained model.

        Args:
            prediction_data (Data): A Data object containing symbol-specific
                                    dataframes for prediction.

        Returns:
            List[Prediction]: A list of Prediction objects containing predictions
                            and associated metadata for each symbol.
        """

        predictions = []

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            if df.empty:
                continue

            df = df.copy()

            try:
                # Extract ground truth if present
                if "target" in df.columns:
                    ground_truth = df.pop("target")
                else:
                    ground_truth = None

                # Extract 'close' and 'atr' columns for further processing
                close = df.pop("close").to_list()
                atr = df.pop("atr").to_list()
                time_stamps = df.index.to_list()

                # Generate predictions
                y_pred_prob = self.model.predict_proba(df)
                predicted_indices = y_pred_prob.argmax(axis=1)
                y_pred = self.model.classes_[predicted_indices]

            except Exception as e:
                logger.info(df)
                raise ValueError(f"Error for symbol: {symbol}. len df: {len(df)}") from e

            # Create a Prediction object for each row of the DataFrame
            for i in range(len(df)):
                # Create and store each individual Prediction object
                predictions.append(
                    Prediction(
                        object_ref=self.config.object_id,
                        symbol=symbol,
                        prediction=int(y_pred[i]),  # Ensure prediction is an integer
                        confidence=dict(zip(self.model.classes_, y_pred_prob[i])),
                        ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                        close=close[i],
                        atr=atr[i],
                        time=time_stamps[i],
                        prediction_type=self.config.prediction_type,
                    )
                )

        return predictions


class LgbmGbrtClf(Model):
    """LGBM Classification Model which uses randomized search."""

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
        """Train the model with hyperparameter tuning.

        Combines all data from training_data, checks for an existing model within the
        same training period, and trains the model if no saved model is found.

        Args:
            training_data (Data): A Data object containing training dataframes for each symbol.

        Returns:
            TrainingInformation: Metadata about the training process, including the symbols,
                                training period, and model file path.
        """

        logger.info("Start training the model.")

        # Concatenate all data and sort by index (time)
        data = pd.concat(training_data.data.values()).sort_index()

        # Separate features (x) and target (y)
        x = data.drop(columns=["target", "close", "atr"])
        y = data["target"]

        # Retrieve symbols and training period
        symbols = list(training_data.data.keys())
        train_start_date = x.index.min()
        train_end_date = x.index.max()

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # Define hyperparameters for RandomizedSearch
            param_distributions = {
                "max_depth": np.arange(1, 10),
                "learning_rate": 10 ** (np.linspace(-5, -1, 100)),
                "n_estimators": np.arange(50, 1000),
                "reg_lambda": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "reg_alpha": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            }

            # tsv = BlockingTimeSeriesSplit(n_splits=1, test_size=3, margin=0)
            tscv = TimeSeriesSplit(
                n_splits=2,
                test_size=int(len(x) * 0.2),
                max_train_size=int(len(x) * 0.8),
            )

            custom_scorer = make_scorer(adapted_f1_score, greater_is_better=True)

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_distributions,
                n_iter=30,  # Number of parameter settings sampled
                cv=tscv,
                scoring=custom_scorer,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

            logger.info("Start hyperparameter tuning with RandomizedSearchCV...")

            # Fit the model with hyperparameter tuning
            random_search.fit(x, y)

            # Set the best model to self.model
            self.model = random_search.best_estimator_
            # self.model.fit(x, y)

            logger.info("Best parameters found: %s", random_search.best_params_)
            logger.info("Best score achieved: %f", random_search.best_score_)

            logger.info("Model training complete.")

        # Return training metadata
        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Generates predictions based on input data using the trained model.

        Args:
            prediction_data (Data): A Data object containing symbol-specific
                                    dataframes for prediction.

        Returns:
            List[Prediction]: A list of Prediction objects containing predictions
                            and associated metadata for each symbol.
        """

        predictions = []

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            if df.empty:
                continue

            df = df.copy()

            try:
                # Extract ground truth if present
                if "target" in df.columns:
                    ground_truth = df.pop("target")
                else:
                    ground_truth = None

                # Extract 'close' and 'atr' columns for further processing
                close = df.pop("close").to_list()
                atr = df.pop("atr").to_list()
                time_stamps = df.index.to_list()

                # Generate predictions
                y_pred_prob = self.model.predict_proba(df)
                predicted_indices = y_pred_prob.argmax(axis=1)
                y_pred = self.model.classes_[predicted_indices]

            except Exception as e:
                logger.info(df)
                raise ValueError(f"Error for symbol: {symbol}. len df: {len(df)}") from e

            # Create a Prediction object for each row of the DataFrame
            for i in range(len(df)):
                # Create and store each individual Prediction object
                predictions.append(
                    Prediction(
                        object_ref=self.config.object_id,
                        symbol=symbol,
                        prediction=int(y_pred[i]),  # Ensure prediction is an integer
                        confidence=dict(zip(self.model.classes_, y_pred_prob[i])),
                        ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                        close=close[i],
                        atr=atr[i],
                        time=time_stamps[i],
                        prediction_type=self.config.prediction_type,
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
        """Train the model with hyperparameter tuning.

        Combines all data from training_data, checks for an existing model within the
        same training period, and trains the model if no saved model is found.

        Args:
            training_data (Data): A Data object containing training dataframes for each symbol.

        Returns:
            TrainingInformation: Metadata about the training process, including the symbols,
                                training period, and model file path.
        """

        logger.info("Start training the model.")

        # Concatenate all data and sort by index (time)
        data = pd.concat(training_data.data.values()).sort_index()

        # Separate features (x) and target (y)
        x = data.drop(columns=["target", "close", "atr"])
        y = data["target"]

        # Retrieve symbols and training period
        symbols = list(training_data.data.keys())
        train_start_date = x.index.min()
        train_end_date = x.index.max()

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # Train the model and perform hyperparameter tuning if necessary
            param_grid = {
                "n_estimators": [100],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 10, 20],
                "min_samples_leaf": [10, 20, 30],
            }

            tscv = TimeSeriesSplit(
                n_splits=2,
                test_size=int(len(x) * 0.3),
                max_train_size=int(len(x) * 0.7),
            )

            custom_scorer = make_scorer(adapted_f1_score, greater_is_better=True)

            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring=custom_scorer,
                cv=tscv,
                n_jobs=-1,
            )

            # Perform grid search and train the model
            logger.info("Starting hyperparameter tuning with GridSearchCV...")
            grid_search.fit(x, y)

            # Get the best model and update self.model
            self.model = grid_search.best_estimator_
            # self.model.fit(x, y)
            logger.info("Best parameters found: %s", grid_search.best_params_)
            logger.info("Model training complete with tuned parameters.")
            logger.info("Model training complete.")

        # Return training metadata
        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Generates predictions based on input data using the trained model.

        Args:
            prediction_data (Data): A Data object containing symbol-specific
                                    dataframes for prediction.

        Returns:
            List[Prediction]: A list of Prediction objects containing predictions
                            and associated metadata for each symbol.
        """

        predictions = []

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            if df.empty:
                continue

            df = df.copy()

            try:
                # Extract ground truth if present
                if "target" in df.columns:
                    ground_truth = df.pop("target")
                else:
                    ground_truth = None

                # Extract 'close' and 'atr' columns for further processing
                close = df.pop("close").to_list()
                atr = df.pop("atr").to_list()
                time_stamps = df.index.to_list()

                # Generate predictions
                y_pred_prob = self.model.predict_proba(df)
                predicted_indices = y_pred_prob.argmax(axis=1)
                y_pred = self.model.classes_[predicted_indices]

            except Exception as e:
                logger.info(df)
                raise ValueError(f"Error for symbol: {symbol}. len df: {len(df)}") from e

            # Create a Prediction object for each row of the DataFrame
            for i in range(len(df)):
                # Create and store each individual Prediction object
                predictions.append(
                    Prediction(
                        object_ref=self.config.object_id,
                        symbol=symbol,
                        prediction=int(y_pred[i]),  # Ensure prediction is an integer
                        confidence=dict(zip(self.model.classes_, y_pred_prob[i])),
                        ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                        close=close[i],
                        atr=atr[i],
                        time=time_stamps[i],
                        prediction_type=self.config.prediction_type,
                    )
                )

        return predictions


class PSARModel(Model):
    """Random Forest Classifier"""

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)

        # create model if it does not exist
        if not hasattr(self, "model") or self.model is None:
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

        logger.info("Train the model with the following symbols: %s", symbols)

        # limit the training data to the last 2000 entries
        training_data.data = {key: values.iloc[-5000:] for key, values in training_data.data.items()}

        data = pd.concat(training_data.data.values()).sort_index()
        train_start_date: pd.Timestamp = min(data.index)
        train_end_date: pd.Timestamp = max(data.index)

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # Train the model and perform hyperparameter tuning if necessary
            for symbol, df in training_data.data.items():
                logger.info("Find best params for %s", symbol)
                result = self.find_best_step(df.copy())

                self.model[symbol] = result

            logger.info("Model training complete.")

        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Function to make predictions.

        The target is not perfect since we care about making correct trend predictions, not so much about having some downs in an uptrend.
        """
        predictions = []
        y_pred = []
        ground_truth = None

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            for row in df.copy().iterrows():
                # Update the model with high, low, close, and date
                self.model[symbol]["instance"].update(
                    high=row[1]["high"], low=row[1]["low"], close=row[1]["close"], date=row[0]
                )

                # Get the action (is_up_trend) from the model
                action = self.model[symbol]["instance"].is_up_trend()
                y_pred.append(action)

            # Extract ground truth if present
            if "target" in df.columns:
                ground_truth = df.pop("target")
            else:
                ground_truth = None

            # Extract 'close' and 'atr' columns for further processing
            close = df.pop("close").to_list()
            atr = df.pop("atr").to_list()
            time_stamps = df.index.to_list()

            # Generate predictions
            y_pred = self.model.predict(df)

            # Create a Prediction object for each row of the DataFrame
            for i in range(len(df)):
                # Create and store each individual Prediction object
                predictions.append(
                    Prediction(
                        object_ref=self.config.object_id,
                        symbol=symbol,
                        prediction=int(y_pred[i]),  # Ensure prediction is an integer
                        ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                        close=close[i],
                        atr=atr[i],
                        time=time_stamps[i],
                        prediction_type=self.config.prediction_type,
                    )
                )

        return predictions

    def calculate_psar(
        self, df: pd.DataFrame, step: float
    ) -> Tuple[float, float, List[float], IncrementalPSARIndicator]:
        """Calculate the cumulative return and Sharpe ratio for a given step."""

        indicator = IncrementalPSARIndicator(step=step, max_step=0.2, timeframe=self.config.timeframe)
        return_list = []
        buy_price = None

        for row in df.iterrows():
            current_price = row[1]["close"]
            indicator.update(high=row[1]["high"], low=row[1]["low"], close=row[1]["close"], date=row[0])

            action = indicator.is_up_trend()

            # If in an uptrend, buy or hold
            if action:
                if not buy_price:
                    buy_price = current_price
            # If not in an uptrend, sell and record the return
            else:
                if buy_price:
                    return_list.append(((current_price - buy_price) / buy_price) - 0.001)  # Assume 0.1% fees
                    buy_price = None

        if len(return_list) == 0:
            return -np.inf, -np.inf, [], indicator

        return_array = np.array(return_list)
        ret = ((1 + return_array).cumprod()[-1] - 1).item()
        sr = self.calculate_sharpe_ratio(return_array)

        return ret, sr, return_list, indicator

    def find_best_step(self, df: pd.DataFrame) -> Dict[str, float]:
        """Perform a hyperparameter search to find the best step for the PSAR indicator."""

        best_indicator: IncrementalPSARIndicator = None
        best_return_list = None

        all_results = []

        list_steps = 10 ** np.linspace(-6, 0, 1000)

        # Hyperparameter search for the best step
        for step in tqdm(list_steps):
            step = step.item()

            # Call the return and Sharpe ratio calculation function
            ret, sr, _, _ = self.calculate_psar(df, step)

            all_results.append((step, ret, sr))

        # Assume result["all_returns"] has your data
        step_values = [t[0] for t in all_results]
        ret_values = [t[1] for t in all_results]

        # Smoothing using Savitzky-Golay filter (from scipy) - fits polynomial to window
        ret_smoothed_sg = savgol_filter(ret_values, window_length=100, polyorder=2)

        # Find the maximum of the smoothed data (you can use either smoothed data above)
        best_step_value = step_values[np.argmax(ret_smoothed_sg)]

        # calculate it again with the best found step
        ret, sr, best_return_list, best_indicator = self.calculate_psar(df, best_step_value)

        best_result = {
            "instance": best_indicator,
            "step": best_step_value,
            "sharpe_ratio": sr,
            "return": ret,
            "number_of_trades": len(best_return_list),
            "all_results": all_results,
        }

        return best_result

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0) -> float:
        """Calculate the Sharpe Ratio for a series of returns.

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

        return sharpe_ratio.item()


class LgbmTrendClf(Model):
    """LGBM Classification Model which uses randomized search."""

    def __init__(self, config: ModelSettings) -> None:
        super().__init__(config)

        # create model if it does not exist
        if not hasattr(self, "model") or self.model is None:
            self.model = {"model": None, "last_pred_vals": {}}

            self.model["model"] = LGBMClassifier(boosting_type="gbdt", n_jobs=-1, verbose=-1)

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
        """Train the model with hyperparameter tuning.

        Combines all data from training_data, checks for an existing model within the
        same training period, and trains the model if no saved model is found.

        Args:
            training_data (Data): A Data object containing training dataframes for each symbol.

        Returns:
            TrainingInformation: Metadata about the training process, including the symbols,
                                training period, and model file path.
        """

        logger.info("Start training the model.")

        # Concatenate all data and sort by index (time)
        data = pd.concat(training_data.data.values()).sort_index()

        # Separate features (x) and target (y)
        x = data.drop(columns=["target", "close", "atr", "symbol"])
        y = data["target"]

        # Retrieve symbols and training period
        symbols = list(training_data.data.keys())
        train_start_date = x.index.min()
        train_end_date = x.index.max()

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # Define hyperparameters for RandomizedSearch
            param_distributions = {
                "max_depth": np.arange(1, 10),
                "learning_rate": 10 ** (np.linspace(-5, -1, 100)),
                "n_estimators": np.arange(50, 1000),
                "reg_lambda": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "reg_alpha": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            }

            # tsv = BlockingTimeSeriesSplit(n_splits=1, test_size=3, margin=0)
            tscv = TimeSeriesSplit(
                n_splits=3,
                test_size=int(len(x) * 0.2),
                max_train_size=int(len(x) * 0.8),
            )

            custom_scorer = make_scorer(adapted_f1_score, greater_is_better=True)

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=self.model["model"],
                param_distributions=param_distributions,
                n_iter=30,  # Number of parameter settings sampled
                cv=tscv,
                scoring=custom_scorer,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

            logger.info("Start hyperparameter tuning with RandomizedSearchCV...")

            # Fit the model with hyperparameter tuning
            random_search.fit(x, y)

            # Set the best model to self.model
            self.model["model"] = random_search.best_estimator_
            # self.model.fit(x, y)

            logger.info("Best parameters found: %s", random_search.best_params_)
            logger.info("Best score achieved: %f", random_search.best_score_)

            logger.info("Model training complete.")

        # Return training metadata
        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Generates predictions based on input data using the trained model.

        Args:
            prediction_data (Data): A Data object containing symbol-specific
                                    dataframes for prediction.

        Returns:
            List[Prediction]: A list of Prediction objects containing predictions
                            and associated metadata for each symbol.
        """

        predictions = []

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            if df.empty:
                continue

            df = df.copy()

            try:
                # Extract ground truth if present
                if "target" in df.columns:
                    ground_truth = df.pop("target")
                else:
                    ground_truth = None
                if "symbol" in df.columns:
                    df.pop("symbol")

                # Extract 'close' and 'atr' columns for further processing
                close = df.pop("close").to_list()
                atr = df.pop("atr").to_list()
                time_stamps = df.index.to_list()

                # Generate predictions
                y_pred_prob = self.model["model"].predict_proba(df)
                predicted_indices = y_pred_prob.argmax(axis=1)
                y_pred = self.model["model"].classes_[predicted_indices]

            except Exception as e:
                logger.info(df)
                raise ValueError(f"Error for symbol: {symbol}. len df: {len(df)}") from e

            # update trend predictions based on last n predictions
            if symbol not in self.model["last_pred_vals"]:
                self.model["last_pred_vals"][symbol] = {"pred_list": [], "last_date": None}

            # Create a Prediction object for each row of the DataFrame
            for i in range(len(df)):

                # a trend is established, if we have 7 consecutive predictions in one direction
                trend_min_entries = 5

                prediction = int(y_pred[i])
                current_date = time_stamps[i]

                # check whether we have conjungtive predictions, e.g. no gaps
                if (
                    self.model["last_pred_vals"][symbol]["last_date"]
                    and (self.model["last_pred_vals"][symbol]["last_date"] + self.config.timeframe) != current_date
                ):
                    prediction = 0
                    self.model["last_pred_vals"][symbol]["pred_list"] = []

                # attach the new prediction to the list of latest n predictions and only keep the relevant ones
                self.model["last_pred_vals"][symbol]["last_date"] = current_date
                self.model["last_pred_vals"][symbol]["pred_list"].append(prediction)
                self.model["last_pred_vals"][symbol]["pred_list"] = self.model["last_pred_vals"][symbol]["pred_list"][
                    -trend_min_entries:
                ]

                # if not enough values are available, set the prediction to 0
                if len(self.model["last_pred_vals"][symbol]["pred_list"]) < trend_min_entries:
                    prediction = 0
                # if there is no clear trend, set prediction value to 0
                if prediction == 1:
                    if sum(self.model["last_pred_vals"][symbol]["pred_list"]) < (trend_min_entries - 1):
                        prediction = 0
                elif prediction == -1:
                    if sum(self.model["last_pred_vals"][symbol]["pred_list"]) > -(trend_min_entries - 1):
                        prediction = 0

                # Create and store each individual Prediction object
                predictions.append(
                    Prediction(
                        object_ref=self.config.object_id,
                        symbol=symbol,
                        prediction=prediction,  # Ensure prediction is an integer
                        confidence=dict(zip(self.model["model"].classes_, y_pred_prob[i])),
                        ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                        close=close[i],
                        atr=atr[i],
                        time=current_date,
                        prediction_type=self.config.prediction_type,
                    )
                )

        return predictions


class LgbmGbrtSupResClf(Model):
    """LGBM Classification Model which uses randomized search."""

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

    def _expand_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to create the dataset.

        It calculates a few features and expands the data.
        """

        expanded_data = []
        target_available = "target" in df.columns
        for row in df.iterrows():

            for i, (level, date) in enumerate(row[1]["all_support_levels"]):

                # only consider the first 5 levels
                if i == 5:
                    break

                current_date = row[0]
                row_copy = row[1].copy()
                row_copy["support_level"] = level
                row_copy["support_date"] = date

                if target_available:
                    # do not add more entries if we don't have any targets left
                    if len(row[1]["target"]) <= i:
                        break
                    row_copy["target"] = row[1]["target"][i]

                ## calculate additional features ##
                row_copy["broken_levels"] = i

                # distance to previous level
                if i > 0:
                    row_copy["dist_to_previous_level"] = row[1]["all_support_levels"][i - 1][0] / level
                else:
                    row_copy["dist_to_previous_level"] = 1

                # distance to next level
                if len(row[1]["all_support_levels"]) > i + 1:
                    row_copy["dist_to_next_level"] = row[1]["all_support_levels"][i + 1][0] / level
                else:
                    row_copy["dist_to_next_level"] = 1

                # candle drawdown to get to the level
                row_copy["candle_drawdown"] = (level - row_copy["close"]) / row_copy["close"]

                # age of the level
                row_copy["level_maturity"] = (current_date - date).total_seconds() / 3600

                expanded_data.append(row_copy)

        df = pd.DataFrame(expanded_data)

        if not df.empty:

            df = df.drop(
                columns=[
                    "open",
                    "high",
                    "low",
                    "volume",
                    "symbol",
                    "all_support_levels",
                    "all_resistance_levels",
                    "support_date",
                ]
            )

        return df

    def _train(self, training_data: Data) -> TrainingInformation:
        """Train the model with hyperparameter tuning.

        Combines all data from training_data, checks for an existing model within the
        same training period, and trains the model if no saved model is found.

        Args:
            training_data (Data): A Data object containing training dataframes for each symbol.

        Returns:
            TrainingInformation: Metadata about the training process, including the symbols,
                                training period, and model file path.
        """

        logger.info("Start training the model.")

        # Concatenate all data and sort by index (time)
        data = pd.concat(training_data.data.values()).sort_index()
        train_start_date = data.index.min()
        train_end_date = data.index.max()
        symbols = list(training_data.data.keys())

        # Build the file path for the model based on the training period
        model_filename = (
            f"{self.config.object_id.value}_{train_start_date.isoformat()}_{train_end_date.isoformat()}.joblib"
        )
        full_model_path = os.path.join(self.config.data_directory, "models", model_filename)

        # Check if a model already exists for the given period
        if os.path.exists(full_model_path):
            logger.info("Found an existing saved model. Skipping training.")
            self.load(full_model_path)
        else:
            # get the final data for training
            expanded_data = []
            for df in training_data.data.values():

                if df.empty:
                    continue
                
                logger.info(50*"---")
                logger.info(print(len(df)))
                # remove all values where the support level is not hit
                df = df[df["target"].apply(lambda x: len(x) > 0)].copy()

                df = self._expand_dataset(df)
                logger.info(50*"---")
                logger.info(print(len(df)))
                if df.empty:
                    continue

                expanded_data.append(df)

            try:
                data = pd.concat(expanded_data)
            except Exception as e:
                logger.info("Error when concatenating expanded data. Data: %s", expanded_data)
                raise ValueError() from e
            
            data = data.sort_values(by=["time"])

            # Separate features (x) and target (y)
            x = data.drop(columns=["target", "close", "atr"])
            y = data["target"]

            # Define hyperparameters for RandomizedSearch
            param_distributions = {
                "max_depth": np.arange(1, 10),
                "learning_rate": 10 ** (np.linspace(-5, -1, 100)),
                "n_estimators": np.arange(50, 1000),
                "reg_lambda": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "reg_alpha": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            }

            # tsv = BlockingTimeSeriesSplit(n_splits=1, test_size=3, margin=0)
            tscv = TimeSeriesSplit(
                n_splits=2,
                test_size=int(len(x) * 0.2),
                max_train_size=int(len(x) * 0.8),
            )

            custom_scorer = make_scorer(adapted_f1_score, greater_is_better=True)

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_distributions,
                n_iter=30,  # Number of parameter settings sampled
                cv=tscv,
                scoring=custom_scorer,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

            logger.info("Start hyperparameter tuning with RandomizedSearchCV...")

            # Fit the model with hyperparameter tuning
            random_search.fit(x, y)

            # Set the best model to self.model
            self.model = random_search.best_estimator_

            # self.model = LGBMClassifier()
            # self.model.fit(x, y)

            logger.info("Best parameters found: %s", random_search.best_params_)
            logger.info("Best score achieved: %f", random_search.best_score_)

            logger.info("Model training complete.")

        # Return training metadata
        return TrainingInformation(
            symbols=symbols,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            file_path_model=full_model_path,
        )

    def _predict(self, prediction_data: Data) -> List[Prediction]:
        """Generates predictions based on input data using the trained model.

        Args:
            prediction_data (Data): A Data object containing symbol-specific
                                    dataframes for prediction.

        Returns:
            List[Prediction]: A list of Prediction objects containing predictions
                            and associated metadata for each symbol.
        """

        predictions = []

        for symbol, df in prediction_data.data.items():

            # skip symbol if it was not part of the training symbols
            if symbol not in self.config.training_information.symbols:
                continue

            if df.empty:
                continue

            df = df.copy()

            # expand the dataset
            df = self._expand_dataset(df)

            if df.empty:
                continue

            try:
                # Extract ground truth if present
                if "target" in df.columns:
                    ground_truth = df.pop("target")
                else:
                    ground_truth = None

                # Extract 'close' and 'atr' columns for further processing
                close = df.pop("close").to_list()
                atr = df.pop("atr").to_list()
                time_stamps = df.index.to_list()
                support_levels = df["support_level"].to_list()

                # Generate predictions
                y_pred_prob = self.model.predict_proba(df)
                predicted_indices = y_pred_prob.argmax(axis=1)
                y_pred = self.model.classes_[predicted_indices]

                # Create a Prediction object for each row of the DataFrame
                for i in range(len(df)):
                    # Create and store each individual Prediction object
                    predictions.append(
                        Prediction(
                            object_ref=self.config.object_id,
                            symbol=symbol,
                            prediction=int(y_pred[i]),  # Ensure prediction is an integer
                            confidence=dict(zip(self.model.classes_, y_pred_prob[i])),
                            execution_price=support_levels[i],
                            ground_truth=int(ground_truth.iloc[i]) if ground_truth is not None else None,
                            close=close[i],
                            atr=atr[i],
                            time=time_stamps[i],
                            prediction_type=self.config.prediction_type,
                        )
                    )

            except Exception as e:
                logger.info("Error for symbol: %s. len df: %s. Message: %s", symbol, len(df), e)
            #    raise ValueError(f"Error for symbol: {symbol}. len df: {len(df)}") from e

        return predictions
