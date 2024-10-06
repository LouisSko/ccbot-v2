"""Module which stores the logic for a data splitter."""

from typing import Tuple

import pandas as pd

from src.core.datasource import Data


def data_splitter(data: Data, split_date: pd.Timestamp) -> Tuple[Data, Data]:
    """Function to split the data into train and test Data

    Args:
        data (Data): the data for training and evaluation
        split_date (pd.Timestamp): the date at which the data is split into train and test

    Note:
        It might seem a bit cumbersome to provide the split as a timestamp and not like a fraction (e.g. 0.8) or pd.Timedelta (e.g. pd.Timedelta("30d")).
        Hower it helps to ensure that the same data is used for training different models in case we have multiple datasources.
        Those different datasources might have different data resolution which could be problematic.
    """

    data_train, data_test = {}, {}
    
    data_cleaned = {}

    # limit the amount of past data to a certain timehorizon
    first_train_date = split_date - pd.Timedelta(weeks=208)

    # Filter out data with too few entries (e.g. less than 300 rows)
    for key, df in data.data.items():
        if len(df) > 50:
            # Limit the data to only entries from the last 1 year before split_date
            data_cleaned[key] =  df[(df.index >= first_train_date)]

    # limit data to 1 year backwards period

    # split data into train and test
    for key, df in data_cleaned.items():
        data_train[key] = df[df.index <= split_date].copy()
        data_test[key] = df[df.index > split_date].copy()

    return Data(object_ref=data.object_ref, data=data_train), Data(object_ref=data.object_ref, data=data_test)
