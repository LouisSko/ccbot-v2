"""Concrete implementation of models."""

import os
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

            # Generate predictions
            y_pred = self.model.predict(df)

            # Collect timestamps for predictions
            time = df.index.to_list()

            # Create and store Prediction object
            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=list(y_pred),
                    ground_truth=ground_truth if ground_truth is None else ground_truth.to_list(),
                    close=close,
                    atr=atr,
                    time=time,
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
                "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],  # Continuous range from 0.1 to 1.0
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

            df = df.copy()

            # Extract ground truth if present
            if "target" in df.columns:
                ground_truth = df.pop("target")
            else:
                ground_truth = None

            # Extract 'close' and 'atr' columns for further processing
            close = df.pop("close").to_list()
            atr = df.pop("atr").to_list()

            # Generate predictions
            y_pred = self.model.predict(df)

            # Collect timestamps for predictions
            time = df.index.to_list()

            # Create and store Prediction object
            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=list(y_pred),
                    ground_truth=ground_truth if ground_truth is None else ground_truth.to_list(),
                    close=close,
                    atr=atr,
                    time=time,
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
                "max_depth": np.arange(1, 11),  # Now a continuous range from 1 to 11
                "learning_rate": 10 ** (np.linspace(-6, 0, 100)),  # Continuous range from 10^-5 to 1
                "n_estimators": np.arange(50, 1001),  # Continuous range from 50 to 1000
                "reg_lambda": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "reg_alpha": sp_uniform(0, 0.25),  # Continuous range from 0 to 0.25
                "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],  # Continuous range from 0.1 to 1.0
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

            # Generate predictions
            y_pred = self.model.predict(df)

            # Collect timestamps for predictions
            time = df.index.to_list()

            # Create and store Prediction object
            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=list(y_pred),
                    ground_truth=ground_truth if ground_truth is None else ground_truth.to_list(),
                    close=close,
                    atr=atr,
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
            "n_estimators": [200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [5, 10, 20],
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

            df = df.copy()

            # Extract ground truth if present
            if "target" in df.columns:
                ground_truth = df.pop("target")
            else:
                ground_truth = None

            close = df.pop("close").to_list()
            atr = df.pop("atr").to_list()

            # Generate predictions
            y_pred = self.model.predict(df)

            # Collect timestamps for predictions
            time = df.index.to_list()

            # Create and store Prediction object
            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=list(y_pred),
                    ground_truth=ground_truth if ground_truth is None else ground_truth.to_list(),
                    close=close,
                    atr=atr,
                    time=time,
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

        # TODO: lets create the df the way they are on the left side with close, action
        self.result_dict = {}

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

            if symbol not in self.result_dict:
                self.result_dict[symbol] = []

            for row in df.copy().iterrows():
                # Update the model with high, low, close, and date
                self.model[symbol]["instance"].update(
                    high=row[1]["high"], low=row[1]["low"], close=row[1]["close"], date=row[0]
                )

                # Get the action (is_up_trend) from the model
                action = self.model[symbol]["instance"].is_up_trend()
                y_pred.append(action)

                # Append OHLC and action data to the list
                self.result_dict[symbol].append(
                    {
                        "symbol": symbol,
                        "date": row[0],
                        "open": row[1].get("open", None),
                        "high": row[1]["high"],
                        "low": row[1]["low"],
                        "close": row[1]["close"],
                        "action": action,
                    }
                )

            # joblib.dump(self.result_dict, "/Users/louisskowronek/Documents/Projects/ccbot-v2/configs/pipeline-psar-bitget-4h/result_dict.joblib")

            time = list(df.index)

            # Extract ground truth if present
            if "target" in df.columns:
                ground_truth = df.pop("target")
            else:
                ground_truth = None

            close = df.pop("close").to_list()
            atr = df.pop("atr").to_list()

            predictions.append(
                Prediction(
                    object_ref=self.config.object_id,
                    symbol=symbol,
                    prediction=y_pred,
                    ground_truth=ground_truth,
                    close=close,
                    atr=atr,
                    time=time,
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
