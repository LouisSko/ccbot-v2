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

    for key, df in data.data.items():
        data_train[key] = df[df.index <= split_date].copy()
        data_test[key] = df[df.index > split_date].copy()

    return Data(object_ref=data.object_ref, data=data_train), Data(object_ref=data.object_ref, data=data_test)
