"""Module which stores evaluation metrics."""

import json
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import accuracy_score

from src.common_packages import create_logger
from src.core.model import Prediction

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class ClassificationMetrics(BaseModel):
    """Base Model for storing evaluation results.

    Dict[str, float] always refers to that the metric gets calculated for every class.
    """

    accuracy: float
    precision: Dict[int, float]
    recall: Dict[int, float]
    f_score: Dict[int, float]
    distribution_pred: Dict[int, float]
    distribution_true: Dict[int, float]
    absolute_pred: Dict[int, int]
    absolute_true: Dict[int, int]


def evaluate(predictions: List[Prediction], save_dir: str) -> None:
    """
    Evaluates model predictions against ground truth, calculates metrics,
    and stores the results in the specified directory.

    Args:
        predictions (List[Prediction]): A list of Prediction objects containing ground truth, predicted values, and object reference.
        save_dir (str): The directory where evaluation results will be stored.

    Raises:
        ValueError: If more than one unique component ID is present in the predictions.
    """

    logger.info("Start evaluation...")

    all_preds = []
    all_targets = []
    ref = []

    # Extract predictions and ground truths
    for prediction in predictions:
        all_targets.extend(prediction.ground_truth)
        all_preds.extend(prediction.prediction)
        ref.append(prediction.object_ref.value)

    # Ensure there's only one component ID
    unique_refs = set(ref)
    if len(unique_refs) != 1:
        logger.error("Multiple component IDs found: %s", unique_refs)
        raise ValueError("There should only be a single component ID.")

    component_id = unique_refs.pop()

    # Calculate metrics
    metrics = calculate_classification_metrics(np.array(all_targets), np.array(all_preds))

    # Create the directory for storing metrics if it doesn't exist
    metrics_dir = Path(save_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save the metrics to a file
    file_path = metrics_dir / f"{component_id}.json"
    try:
        with file_path.open("w", encoding="utf8") as file:
            json.dump(metrics.model_dump(), file, indent=4)
        logger.info("Evaluation results saved to %s", file_path)
    except Exception as e:
        logger.error("Failed to save evaluation results to %s: %s", file_path, str(e))
        raise


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> ClassificationMetrics:
    """Calculate classification metrics. Predictions need to be 1, 0, -1

    Args:
        y_true (Union[pd.Series, np.ndarray]): True values for the target variable.
        y_pred (Union[pd.Series, np.ndarray]): Predicted values.

    Returns:
        Dict[str, float]: Dictionary with classification metrics.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    elif not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be either a numpy array or a pandas Series.")

    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    elif not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be either a numpy array or a pandas Series.")

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # calculate binary metrics
    precision, recall, f1_score = multi_target_precision_recall_f1(y_true, y_pred)

    accuracy = np.round(accuracy_score(y_true, y_pred), 7).item()

    # Number of positive and negative predictions
    counts_pred = np.unique(y_pred, return_counts=True)
    distr_pred = counts_pred[1] / len(y_pred)
    distribution_pred = {int(v): np.round(d, 3).item() for v, d in zip(counts_pred[0], distr_pred)}

    abs_pred = {int(v): d.item() for v, d in zip(counts_pred[0], counts_pred[1])}

    counts_true = np.unique(y_true, return_counts=True)
    distr_true = counts_true[1] / len(y_true)
    distribution_true = {int(v): np.round(d, 3).item() for v, d in zip(counts_true[0], distr_true)}

    abs_true = {int(v): d.item() for v, d in zip(counts_true[0], counts_true[1])}

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f_score=f1_score,
        distribution_pred=distribution_pred,
        distribution_true=distribution_true,
        absolute_pred=abs_pred,
        absolute_true=abs_true,
    )


def multi_target_precision_recall_f1(y_true, y_pred):
    """Calculates multi target precision and recall for specific classes"""

    precision = {}
    recall = {}
    f1_score = {}
    classes = [-1, 0, 1]  # -1 corresponds to short, 1 corresponds to long
    for c in classes:
        tp = sum((y_true == y_pred) & (y_pred == c))
        fp = sum((y_true != y_pred) & (y_pred == c))
        fn = sum((y_true != y_pred) & (y_true == c))
        if (tp + fp) > 0:
            precision[c] = np.round(tp / (tp + fp), 5).item()
        else:
            precision[c] = 0
        if (tp + fn) > 0:
            recall[c] = np.round(tp / (tp + fn), 5).item()
        else:
            recall[c] = 0

        f1_score[c] = np.round(f_score(precision[c], recall[c], beta=0.5), 5).item()

    return precision, recall, f1_score


def f_score(precision: float, recall: float, beta: int = 1):
    """Calculate the F_n score where n is a parameter.

    Parameters:
    precision (float): The precision of the model.
    recall (float): The recall of the model.
    beta (float): The weighting factor. Default is 1 (equivalent to F1 score).

    Returns:
    float: The F score.
    """

    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
