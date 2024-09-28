"""Module for Ensemble Model

It creates ensemble predictions based on individual predictions of multiple models.
"""

from collections import Counter
from itertools import chain
from typing import Dict, List, Literal, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId
from src.core.model import Prediction


class EnsembleSettings(BaseModel):
    """Base Model for defining the settings of the ensemble model."""

    object_id: ObjectId
    depends_on: List[ObjectId]
    ground_truth_object_ref: ObjectId  # which ground truth, close and atr to choose from.
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
        predictions_by_symbol = self._group_predictions_by_symbol_and_time(flat_predictions)

        # Create ensemble results for each symbol and timestamp
        ensemble_results = []
        for symbol, predictions_by_time in predictions_by_symbol.items():
            for timestamp, prediction in predictions_by_time.items():
                if self.config.task_type == "classification":
                    ensemble_predictions = self._classification_voting(prediction["predictions"])
                elif self.config.task_type == "regression":
                    ensemble_predictions = self._regression_mean(prediction["predictions"])
                else:
                    raise ValueError(f"Unknown task type: {self.config.task_type}")

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
                    )
                )

        return ensemble_results

    def _group_predictions_by_symbol_and_time(
        self, predictions: List[Prediction]
    ) -> Dict[str, Dict[pd.Timestamp, Dict]]:
        """Group predictions by symbol and timestamp for easier aggregation.

        Returns:
        Dict[str, Dict[pd.Timestamp, Dict]]: Grouped predictions by symbol and timestamp.
        """

        grouped_predictions = {}

        for pred in predictions:
            if pred.symbol not in grouped_predictions:
                grouped_predictions[pred.symbol] = {}
                # create entry if it doesn't exist
                if pred.time not in grouped_predictions[pred.symbol]:
                    grouped_predictions[pred.symbol][pred.time] = {
                        "predictions": [],
                        "ground_truth": None,
                        "close": None,
                        "atr": None,
                    }
                # add predictions
                grouped_predictions[pred.symbol][pred.time]["predictions"].append(pred.prediction)

                # add the other variables
                if pred.object_ref == self.config.ground_truth_object_ref:
                    # Store the ground truth for the symbol
                    grouped_predictions[pred.symbol][pred.time].update(
                        {
                            "ground_truth": pred.ground_truth,
                            "close": pred.close,
                            "atr": pred.atr,
                        }
                    )

        return grouped_predictions

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
