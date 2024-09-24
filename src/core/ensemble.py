"""Module for Ensemble Model

It creates ensemble predictions based on individual predictions of multiple models.
"""

from collections import Counter
from itertools import chain
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.model import Prediction


class EnsembleSettings(BaseModel):
    """Base Model for defining the settings of the ensemble model."""

    object_id: ObjectId
    depends_on: List[ObjectId]
    ground_truth_object_ref: ObjectId  # which ground to choose from.
    task_type: Literal["classification", "regression"]
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

    settings: EnsembleSettings


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
        predictions_by_symbol = self._group_predictions_by_symbol_and_time(flat_predictions)

        # Extract ground truth predictions if available
        ground_truth_predictions = self._get_ground_truth_predictions(flat_predictions)

        # Create ensemble results for each symbol
        ensemble_results = []
        for symbol, predictions_by_time in predictions_by_symbol.items():
            if self.config.task_type == "classification":
                ensemble_predictions = self._classification_voting(predictions_by_time)
            elif self.config.task_type == "regression":
                ensemble_predictions = self._regression_mean(predictions_by_time)
            else:
                raise ValueError(f"Unknown task type: {self.config.task_type}")

            # Get the ground truth for this symbol if available
            ground_truth_for_symbol = ground_truth_predictions.get(symbol)

            # Create a new Prediction object for each symbol
            ensemble_results.append(
                Prediction(
                    symbol=symbol,
                    object_ref=self.config.object_id,
                    time=list(predictions_by_time.keys()),  # Using the timestamps from the grouped predictions
                    prediction=ensemble_predictions,
                    ground_truth=ground_truth_for_symbol,  # Attach ground truth if found
                )
            )

        return ensemble_results

    def _group_predictions_by_symbol_and_time(
        self, predictions: List[Prediction]
    ) -> Dict[str, Dict[pd.Timestamp, List[float]]]:
        """Group predictions by symbol and timestamp for easier aggregation.

        Returns:
        Dict[str, Dict[pd.Timestamp, List[float]]]: Grouped predictions by symbol and timestamp.
        """

        grouped_predictions = {}

        for pred in predictions:
            if pred.symbol not in grouped_predictions:
                grouped_predictions[pred.symbol] = {}
            for t, p in zip(pred.time, pred.prediction):
                if t not in grouped_predictions[pred.symbol]:
                    grouped_predictions[pred.symbol][t] = []
                grouped_predictions[pred.symbol][t].append(p)
        return grouped_predictions

    def _get_ground_truth_predictions(self, predictions: List[Prediction]) -> Dict[str, List[float]]:
        """Fetch ground truth from predictions based on the specified ground_truth_object_ref.

        Args:
        predictions (List[Prediction]): The list of predictions.

        Returns:
        Dict[str, List[float]]: A dictionary mapping symbols to their ground truth values.
        """

        ground_truth_predictions = {}

        for pred in predictions:
            if pred.object_ref == self.config.ground_truth_object_ref:
                # Store the ground truth for the symbol
                ground_truth_predictions[pred.symbol] = pred.ground_truth

        return ground_truth_predictions

    def _classification_voting(self, predictions_by_time: Dict[pd.Timestamp, List[int]]) -> List[int]:
        """Perform voting for classification tasks.

        Args:
        predictions_by_time (Dict[pd.Timestamp, List[int]]): Predictions grouped by timestamp.

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
            return [majority_vote(preds) for preds in predictions_by_time.values()]
        if self.config.agreement_type_clf == "all":
            return [check_all_same(preds) for preds in predictions_by_time.values()]

        raise ValueError("specified agreement type is invalid. Choose 'voting' or 'all'.")

    def _regression_mean(self, predictions_by_time: Dict[pd.Timestamp, List[float]]) -> List[float]:
        """Perform mean averaging for regression tasks.

        Args:
        predictions_by_time (Dict[pd.Timestamp, List[float]]): Predictions grouped by timestamp.

        Returns:
        List[float]: The mean value for each timestamp.
        """

        return [np.mean(preds) for preds in predictions_by_time.values()]

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
