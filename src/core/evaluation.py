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


def multi_target_precision(y_true: np.ndarray, y_pred: np.ndarray):
    """Function for multi-target precision"""
    precision = {}
    classes = [1, -1]  # 1 for long, -1 for short

    for c in classes:
        tp = sum((y_true == y_pred) & (y_pred == c))
        fp = sum((y_true != y_pred) & (y_pred == c))
        if (tp + fp) > 0:
            precision[c] = tp / (tp + fp)
        else:
            precision[c] = 0

    return np.mean(list(precision.values()))


def multi_target_recall(y_true: np.ndarray, y_pred: np.ndarray):
    """Function for multi-target recall"""

    recall = {}
    classes = [1, -1]  # 1 for long, -1 for short
    for c in classes:
        tp = sum((y_true == y_pred) & (y_pred == c))
        fn = sum((y_true != y_pred) & (y_true == c))
        if (tp + fn) > 0:
            recall[c] = tp / (tp + fn)
        else:
            recall[c] = 0

    return np.mean(list(recall.values()))


def adapted_f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute adapted F1 score where precision is twice as important as recall"""

    precision = multi_target_precision(y_true, y_pred)
    recall = multi_target_recall(y_true, y_pred)

    # Weigh precision as twice as important as recall
    f1_score = (2 * precision * recall) / (2 * precision + recall) if (2 * precision + recall) > 0 else 0

    return f1_score


"""Module for storing plotting functions for visualizing trades."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cumulative_profit(df, col: Literal["profit_percent", "profit"] = "profit"):
    """Plots the cumulative profit and net profit over time.

    Args:
        df (pd.DataFrame): DataFrame containing 'close_date', 'profit', and 'net_profit' columns.
        return_type (str): Type of return to calculate. Options are 'simple' or 'log'.

    Raises:
        ValueError: If an invalid return type is specified.
    """

    df["close_date"] = pd.to_datetime(df["close_date"])

    # Calculate cumulative sum of simple returns (profits)
    df["cumulative_profit"] = df[col].astype(float).cumsum()
    df["cumulative_net_profit"] = df[f"net_{col}"].astype(float).cumsum()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df["close_date"], df["cumulative_profit"], linestyle="-", label="Cumulative Profit")#, marker="o")
    plt.plot(df["close_date"], df["cumulative_net_profit"], linestyle="-", label="Cumulative Net Profit")#, marker="o")

    plt.title("Cumulative Profit Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cumulative_profit_by_symbol(df):
    # Sort the dataframe by date
    df = df.sort_values(by="close_date")

    # Group by symbol and calculate cumulative profit
    df["cumulative_profit"] = df.groupby("symbol")["profit"].cumsum()

    # Plotting
    plt.figure(figsize=(12, 6))
    for symbol, group in df.groupby("symbol"):
        plt.plot(group["close_date"], group["cumulative_profit"], linestyle="-", label=symbol)

    plt.title("Cumulative Profit Over Time by Symbol")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_balance_over_time(df):
    # Convert close_date to datetime if not already
    df["close_date"] = pd.to_datetime(df["close_date"])
    df["total_balance"] = df["total_balance"].astype(float)
    df["free_balance"] = df["free_balance"].astype(float)

    plt.figure(figsize=(12, 6))

    # Plot the total_balance as a line
    plt.plot(df["close_date"], df["total_balance"], marker="o", linestyle="-", color="b", label="Total Balance")

    # Plot the free_balance as bars
    plt.plot(df["close_date"], df["free_balance"], color="orange", alpha=1, label="Free Balance")

    # Customize the plot
    plt.title("Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()


def plot_cumulative_profit_by_symbol_and_side(df, column: str = "profit"):
    symbols = df["symbol"].unique()
    sides = df["side"].unique()
    sides = ["buy", "sell"]

    fig, axs = plt.subplots(len(symbols), len(sides), figsize=(15, 5 * len(symbols)), sharex=True, sharey=True)
    fig.suptitle(f"Cumulative {column} Over Time by Symbol and Trade Side")

    for i, symbol in enumerate(symbols):
        for j, side in enumerate(sides):
            ax = axs[i, j] if len(symbols) > 1 else axs[j]
            subset = df[(df["symbol"] == symbol) & (df["side"] == side)].sort_values(by="close_date")
            if subset.empty:
                continue
            subset["cumulative_profit"] = subset[column].cumsum()
            ax.plot(subset["close_date"], subset["cumulative_profit"], marker="o", linestyle="-")

            # Calculate success percentage
            success_rate = (subset[column] > 0).mean() * 100

            ax.set_title(f"{symbol} ({side})\nSuccess Rate: {success_rate:.2f}%")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"Cumulative {column}")
            ax.grid(True)
            ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_entry_exit_prices(df):
    symbols = df["symbol"].unique()
    sides = df["side"].unique()

    fig, axs = plt.subplots(len(symbols), len(sides), figsize=(15, 5 * len(symbols)))
    fig.suptitle("Entry vs Exit Prices by Symbol and Trade Side")

    for i, symbol in enumerate(symbols):
        for j, side in enumerate(sides):
            ax = axs[i, j] if len(symbols) > 1 else axs[j]
            subset = df[(df["symbol"] == symbol) & (df["side"] == side)]
            if subset.empty:
                continue
            entry_prices = subset["entry_price"]
            exit_prices = subset["exit_price"]

            ax.scatter(entry_prices, exit_prices, c="b", alpha=0.5)
            max_price = max(entry_prices.max(), exit_prices.max())
            min_price = min(entry_prices.min(), exit_prices.min())
            ax.plot([min_price, max_price], [min_price, max_price], "r--")

            ax.set_title(f"{symbol} ({side})")
            ax.set_xlabel("Entry Price")
            ax.set_ylabel("Exit Price")
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
