"""Module for Ensemble Model

It creates ensemble predictions based on individual predictions of multiple models.
"""

import os
from collections import Counter
from itertools import chain
from typing import Dict, List, Literal, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.model import Prediction

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class EnsembleSettings(BaseModel):
    """Base Model for defining the settings of the ensemble model."""

    object_id: ObjectId
    depends_on: List[ObjectId]
    ground_truth_object_ref: ObjectId  # which ground truth, close and atr to choose from.
    prediction_type: Literal["regression", "direction", "volatility"]
    agreement_type_clf: Literal["voting", "all"]
    data_directory: str  # directory where the data is getting saved

    @model_validator(mode="after")
    def check_ground_truth_in_depends_on(cls, values):  # pylint: disable=no-self-argument
        """Validate that the ground_truth_object_ref exists in the depends_on list."""
        ground_truth = values.ground_truth_object_ref
        depends_on = values.depends_on

        if ground_truth not in depends_on:
            raise ValueError("The ground_truth_object_ref must be one of the object IDs in depends_on.")
        return values


class EnsembleConfiguration(BaseConfiguration):
    """Configuration which can be used for reloading the ensemble model."""

    component_type: str = Field(
        default="ensemble",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: Union[EnsembleSettings, Type[EnsembleSettings]]


class EnsembleModel(BasePipelineComponent):
    """Ensemble model to combine predictions from individual models."""

    def __init__(self, config: EnsembleSettings):
        """Initialize the EnsembleModel with the settings"""
        self.config = config

    def ensemble(self, predictions: List[List[Prediction]]) -> List[Prediction]:
        """Ensemble the predictions of multiple models for multiple symbols and return a list of Prediction objects.

        Returns:
        List[Prediction]: A list of Prediction objects, one for each symbol.
        """

        # flatten the List[List[Prediction]] to a List[Prediction]
        flat_predictions = list(chain.from_iterable(predictions))

        # Group predictions by symbol and timestamp
        predictions_by_timestamp = self._group_predictions_by_time_symbol(flat_predictions)
        # Create ensemble results for each symbol and timestamp
        ensemble_results = []

        for timestamp, predictions_by_symbol in predictions_by_timestamp.items():
            for symbol, prediction in predictions_by_symbol.items():
                confidence_score = None
                if self.config.prediction_type == "direction":
                    ensemble_predictions = self._classification_voting(prediction["predictions"])
                    confidence_score = self._confidence_mean(prediction["confidence"])
                elif self.config.prediction_type == "volatility":
                    ensemble_predictions = self._classification_voting(prediction["predictions"])
                    confidence_score = self._confidence_mean(prediction["confidence"])
                elif self.config.prediction_type == "regression":
                    ensemble_predictions = self._regression_mean(prediction["predictions"])
                else:
                    raise ValueError(f"Unknown task type: {self.config.prediction_type}")

                # Create a new Prediction object for each symbol
                ensemble_results.append(
                    Prediction(
                        symbol=symbol,
                        object_ref=self.config.object_id,
                        time=timestamp,  # Using the timestamps from the grouped predictions
                        prediction=ensemble_predictions,
                        ground_truth=prediction.get("ground_truth", None),  # Attach ground truth if found
                        close=prediction.get("close"),
                        atr=prediction.get("atr"),
                        prediction_type=self.config.prediction_type,
                        confidence=confidence_score,
                    )
                )

        # filter out predictions based on confidence scores.
        if self.config.prediction_type == "direction":
            ensemble_results = self.filter_predictions(
                ensemble_results, fraction_to_keep=0.2, class_encodings=[-1, 0, 1]
            )
        elif self.config.prediction_type == "volatility":
            ensemble_results = self.filter_predictions(ensemble_results, fraction_to_keep=0.8, class_encodings=[0, 1])

        return ensemble_results

    def _group_predictions_by_time_symbol(self, predictions: List[Prediction]) -> Dict[pd.Timestamp, Dict[str, Dict]]:
        """Group predictions by timestamp and symbol for easier aggregation.

        It also makes sure that the predictions have the correct type.

        Returns:
        Dict[str, Dict[pd.Timestamp, Dict]]: Grouped predictions by symbol and timestamp.
        """

        grouped_predictions = {}

        for pred in predictions:

            if pred.prediction_type != self.config.prediction_type:
                raise ValueError(
                    f"Prediction_types of individual models and overarching ensemble need to match. Make sure they are aligned. Model: {pred.prediction_type}; Ensemble: {self.config.prediction_type}"
                )
            if pred.time not in grouped_predictions:
                grouped_predictions[pred.time] = {}
            # create entry if it doesn't exist
            if pred.symbol not in grouped_predictions[pred.time]:
                grouped_predictions[pred.time][pred.symbol] = {
                    "predictions": [],
                    "confidence": [],
                    "ground_truth": None,
                    "close": None,
                    "atr": None,
                }
            # add predictions
            grouped_predictions[pred.time][pred.symbol]["predictions"].append(pred.prediction)
            if pred.confidence:
                grouped_predictions[pred.time][pred.symbol]["confidence"].append(pred.confidence)

            # add the other variables
            if pred.object_ref == self.config.ground_truth_object_ref:
                # Store the ground truth for the symbol
                grouped_predictions[pred.time][pred.symbol].update(
                    {
                        "ground_truth": pred.ground_truth,
                        "close": pred.close,
                        "atr": pred.atr,
                    }
                )

        return grouped_predictions

    def _confidence_mean(self, confidence: List[dict]) -> dict:
        """Averages the confidence scores."""

        mean_confidence = dict(pd.DataFrame(confidence).mean())
        mean_confidence = {k: float(v) for k, v in mean_confidence.items()}

        return mean_confidence

    def _classification_voting(self, prediction_list: List[int]) -> int:
        """Perform voting for classification tasks.

        Args:
        predictions_by_time (List[int]]): predictions of all individual models

        Returns:
        List[int]: The majority-vote result for each timestamp.
        """

        def majority_vote(preds: List[int]) -> int:
            """checks majority vote.

            Returns 0 in case of a draw
            """
            counts = Counter(preds).most_common()
            # Check for a tie between the top two most common predictions
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return 0  # Return 0 in case of a tie
            return counts[0][0]  # Return the most common prediction otherwise

        def check_all_same(preds: List[int]) -> int:
            """Checks if all entries in the list are the same and returns the value.

            Returns zero in case not all items are the same
            """

            if all(pred == preds[0] for pred in preds):
                return preds[0]
            return 0

        if self.config.agreement_type_clf == "voting":
            return majority_vote(prediction_list)
        if self.config.agreement_type_clf == "all":
            return check_all_same(prediction_list)

        raise ValueError("specified agreement type is invalid. Choose 'voting' or 'all'.")

    def _regression_mean(self, prediction_list: List[float]) -> float:
        """Perform mean averaging for regression tasks.

        Args:
        prediction_list (List[float]): Predictions grouped by timestamp.

        Returns:
        List[float]: The mean value for each timestamp.
        """

        return np.mean(prediction_list)

    def _merge_probabilities(self, probability_list: List[float]) -> float:
        """Perform merging of probabilities

        Args:
        probability_list (List[float]): Predictions grouped by timestamp.

        Returns:
        List[float]: The mean value for each timestamp.
        """

        return np.mean(probability_list)

    def filter_predictions(
        self,
        predictions: List[Prediction],
        class_encodings: List[int],
        fraction_to_keep: float = 0.2,
    ) -> List[Prediction]:
        """Filters predictions by confidence scores for specified class encodings.

        The function first groups the predictions by timestamp, then for each timestamp and class encoding
        (e.g., 1 for buy, -1 for sell), it selects the top `fraction_to_keep` of predictions based on their confidence scores.
        Predictions not selected are neutralized (i.e., their prediction is set to 0).

        Args:
            predictions (List[Prediction]): A list of `Prediction` objects containing predictions for different assets.
            class_encodings (List[int]): A list of class encodings to filter predictions for. E.g. [-1, 1]
            fraction_to_keep (float, optional): The fraction of predictions to keep for each timestamp and class encoding.
                                                Defaults to 0.2, meaning only the top 20% of predictions are retained.

        Returns:
            List[Prediction]: The list of predictions where only the top `fraction_to_keep` have been retained
                              for each class encoding, and the rest have been neutralized (prediction set to 0).
        """

        predictions_by_time = {}

        for prediction in predictions:

            # if no confidence scores are provided, return the same predictions.
            if prediction.confidence is None:
                return predictions

            # Group predictions by timestamp
            if prediction.time not in predictions_by_time:
                predictions_by_time[prediction.time] = []

            predictions_by_time[prediction.time].append(prediction)

        # Filter predictions
        filtered_predictions = []

        for pred_list in predictions_by_time.values():
            for class_encoding in class_encodings:

                # do nothing if class encoding is 0, since the result will be the same
                if class_encoding == 0:
                    continue

                # Filter predictions for each class encoding (e.g., 1 and -1)
                pred_confidences = [
                    (i, pred.confidence[1], pred)
                    for i, pred in enumerate(pred_list)
                    if pred.prediction == class_encoding
                ]

                if pred_confidences:
                    # Sort by confidence in descending order
                    pred_confidences_sorted = sorted(pred_confidences, key=lambda x: x[1], reverse=True)
                    n_to_keep = max(3, int(len(pred_confidences_sorted) * fraction_to_keep))

                    # Keep the top n% predictions based on confidence
                    indices_to_keep = [x[0] for x in pred_confidences_sorted[:n_to_keep]]

                    for i, pred in enumerate(pred_list):
                        if i not in indices_to_keep and pred.prediction == class_encoding:
                            pred.prediction = 0  # Neutralize prediction

            filtered_predictions.extend(pred_list)

        return filtered_predictions

    def create_configuration(self) -> EnsembleConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{EnsembleConfiguration.__module__}.{EnsembleConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return EnsembleConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )
