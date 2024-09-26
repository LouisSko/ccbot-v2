"""Concrete implementation of models."""

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import ta
from lightgbm import LGBMClassifier
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from tqdm import tqdm

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

        # Define hyperparameters for RandomizedSearch
        param_distributions = {
            "max_depth": np.arange(1, 7),  # Now a continuous range from 1 to 11
            "learning_rate": 10 ** (np.linspace(-4, -2, 100)),  # Continuous range from 10^-5 to 1
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
            n_iter=40,  # Number of parameter settings sampled
            cv=tscv,
            scoring=custom_scorer,
            n_jobs=4,
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
            n_iter=40,  # Number of parameter settings sampled
            cv=tscv,
            scoring=custom_scorer,
            n_jobs=4,
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

        param_grid = {
            "n_estimators": [200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
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

    def find_best_step(self, df: pd.DataFrame) -> Dict[str, float]:
        """Find best step for psar indicator."""

        best_sr = -np.inf
        best_return = -np.inf
        best_step = -np.inf
        best_indicator: IncrementalPSARIndicator = None
        best_return_list = None

        df["return"] = (df["close"].shift(-1, freq="4h") - df["close"]) / df["close"]
        df = df.fillna(0)

        for step in tqdm(np.linspace(0.0001, 0.1, 500)):

            step = step.item()
            indicator = IncrementalPSARIndicator(step=step, max_step=0.2, timeframe=self.config.timeframe)
            return_list = []
            buy_price = None

            for row in df.iterrows():

                current_price = row[1]["close"]
                ret = row[1]["return"] - 0.001 #assume 0.1% fees
                indicator.update(high=row[1]["high"], low=row[1]["low"], close=row[1]["close"], date=row[0])

                action = indicator.is_up_trend()

                # get in a new trade
                if action:
                    return_list.append(ret)

                #     if not buy_price:
                #         buy_price = current_price
                # # otherwise close the trade
                # else:
                #     # get out of trade if action says no trade
                #     if buy_price:
                #         return_list.append((current_price - buy_price) / buy_price)
                #         buy_price = None

            if len(return_list) != 0:
                return_array = np.array(return_list)

                ret = (1 + return_array).cumprod()[-1]
                sr = self.calculate_sharpe_ratio(return_array)

                if ret > best_return:
                    best_sr = sr
                    best_step = step
                    best_return = ret
                    best_indicator = indicator
                    best_return_list = return_list

        best_result = {
            "instance": best_indicator,
            "step": best_step,
            "sharpe_ratio": best_sr,
            "return": best_return,
            "number_of_trades": len(best_return_list),
        }

        logger.info("Found best result %s", best_result)

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

        return sharpe_ratio


class IncrementalPSARIndicator:
    """Incremental Parabolic Stop and Reverse (Parabolic SAR)

    This class calculates PSAR incrementally, processing new OHLC data one at a time.

    Args:
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
    """

    def __init__(
        self,
        step: float = 0.02,
        max_step: float = 0.20,
        timeframe: pd.Timedelta = pd.Timedelta("4h"),
    ):
        self._step = step
        self._max_step = max_step

        # Internal state variables
        self._initialized = False
        self._up_trend = True
        self._acceleration_factor = self._step
        self._up_trend_high = None
        self._down_trend_low = None
        self._psar = None
        self._last_high = None
        self._last_low = None
        self._previous_high = None
        self._previous_low = None
        self._psar_up = None
        self._psar_down = None
        self._last_date = None
        self._timeframe = timeframe
        self._counter: int = 0

    def _initialize_state(
        self,
        high: float,
        low: float,
        close: float,
        date: pd.Timestamp,
    ):
        """Initialize state on the first OHLC point."""
        self._up_trend_high = high
        self._down_trend_low = low
        self._psar = close
        self._last_high = high
        self._last_low = low
        self._initialized = True
        self._last_date = date

    def update(self, high: float, low: float, close: float, date: pd.Timestamp):
        """Incrementally update the PSAR indicator with a new OHLC data point."""

        self._counter += 1

        if not self._initialized:
            self._initialize_state(high, low, close, date)
            return

        if self._last_date + self._timeframe != date:
            raise ValueError(f"the last seen date is {self._last_date} but the current date is {date}")

        self._last_date = date

        reversal = False

        # Last PSAR value
        psar_prev = self._psar
        self._previous_high = self._last_high
        self._previous_low = self._last_low
        self._last_high = high
        self._last_low = low

        if self._up_trend:
            # PSAR for uptrend
            psar_new = psar_prev + self._acceleration_factor * (self._up_trend_high - psar_prev)

            # Check for trend reversal
            if low < psar_new:
                reversal = True
                psar_new = self._up_trend_high
                self._down_trend_low = low
                self._acceleration_factor = self._step
            else:
                # Update the trend high and acceleration factor
                if high > self._up_trend_high:
                    self._up_trend_high = high
                    self._acceleration_factor = min(self._acceleration_factor + self._step, self._max_step)

                # Ensure PSAR does not exceed the previous two lows
                if self._previous_low < psar_new:
                    psar_new = self._previous_low
                if self._last_low < psar_new:
                    psar_new = self._last_low

        else:
            # PSAR for downtrend
            psar_new = psar_prev - self._acceleration_factor * (psar_prev - self._down_trend_low)

            # Check for trend reversal
            if high > psar_new:
                reversal = True
                psar_new = self._down_trend_low
                self._up_trend_high = high
                self._acceleration_factor = self._step
            else:
                # Update the trend low and acceleration factor
                if low < self._down_trend_low:
                    self._down_trend_low = low
                    self._acceleration_factor = min(self._acceleration_factor + self._step, self._max_step)

                # Ensure PSAR does not fall below the previous two highs
                if self._previous_high > psar_new:
                    psar_new = self._previous_high
                if self._last_high > psar_new:
                    psar_new = self._last_high

        # If a reversal happened, toggle the trend
        self._up_trend = not self._up_trend if reversal else self._up_trend

        if self._up_trend:
            self._psar_up = psar_new
            self._psar_down = None
        else:
            self._psar_up = None
            self._psar_down = psar_new

        # Update current PSAR
        self._psar = psar_new

    def psar(self) -> float:
        """Return the current PSAR value."""
        return self._psar

    def is_up_trend(self) -> bool:
        """Return True if the current trend is up, False if down."""

        # warmup phase of 100 entries. During this time, we don't want to make any trend predictions.
        if self._counter < 100:
            return False

        return self._up_trend

    def psar_up(self) -> pd.Series:
        """Return the PSAR uptrend values."""
        return self._psar_up

    def psar_down(self) -> pd.Series:
        """Return the PSAR downtrend values."""
        return self._psar_down
