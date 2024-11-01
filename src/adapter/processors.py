"""Module for implementing the processors."""

import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import ta

from src.common_packages import create_logger
from src.core.datasource import Data
from src.core.processor import DataProcessorSettings, FeatureGenerator, TargetGenerator

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class FeaturesFearGreed(FeatureGenerator):
    """Data Processor for calculating features for fear and greed index."""

    def create_features(self, data: Data) -> Data:
        """Calculates and adds features to the dataframes in the input dictionary. Optionally filters the output based on given timestamps."""

        df = data.data["all"].copy()

        tf = pd.Timedelta("1d")
        logger.info("Start calculating features...")

        shifts = [1, 3, 7, 14, 21]

        # percentage change
        for shift in shifts:
            df = percentage_change(
                df, "fear_n_greed", drop=False, new_column=f"fear_n_greed/pct_{shift}", shift=shift, timeframe=tf
            )

        # fear_n_greed 3 standard deviation
        df = rolling_feature(
            df,
            metric="fear_n_greed",
            drop=False,
            new_column="fear_n_greed/std_3",
            window=3,
            operation="std",
            timeframe=tf,
        )

        # fear_n_greed 7 standard deviation
        df = rolling_feature(
            df,
            metric="fear_n_greed",
            drop=False,
            new_column="fear_n_greed/std_7",
            window=7,
            operation="std",
            timeframe=tf,
        )

        # fear_n_greed 14 standard deviation
        df = rolling_feature(
            df,
            metric="fear_n_greed",
            drop=False,
            new_column="fear_n_greed/std_14",
            window=14,
            operation="std",
            timeframe=tf,
        )

        # fear_n_greed mean
        df = rolling_feature(
            df,
            metric="fear_n_greed",
            drop=False,
            new_column="fear_n_greed/mean_3",
            window=3,
            operation="mean",
            timeframe=tf,
        )

        df = rolling_feature(
            df,
            metric="fear_n_greed",
            drop=False,
            new_column="fear_n_greed/mean_7",
            window=7,
            operation="mean",
            timeframe=tf,
        )

        df = df.dropna()

        df = resample_with_forward_fill(df, self.config.timeframe)

        logger.info(
            "Calculation of features finished. Calculated features: %s, Dataset entries: %s", len(df.columns), len(df)
        )

        return Data(object_ref=self.config.object_id, data={"all": df})


class FeaturesExchange(FeatureGenerator):
    """Datapreprocessor for Classification Problems."""

    def create_features(self, data: Data) -> Data:
        """Calculates and adds features to the dataframes in the input dictionary. Optionally filters the output based on given timestamps."""

        logger.info("Start calculating features...")

        data_processed = {}

        for symbol, df in data.data.items():

            df = df.copy()

            df = generate_price_related_features(df, self.config.timeframe)
            df = generate_volume_related_features(df, self.config.timeframe)
            df = generate_technical_indicators(df)
            df = generate_kernel_features(df, self.config.timeframe)
            df = generate_ribbon_features(df)
            df = generate_stochastic_features(df)
            df = add_sinusoidal_time_features(df)
            df = filter_low_volume_rows(
                df, timeframe=self.config.timeframe, volume_col="volume", price_col="close", threshold_24h=3_000_000
            )
            df = cleaning_past_features(df)

            if not df.empty:
                data_processed[symbol] = df

                logger.info(
                    "Calculation of features finished for %s. Calculated features: %s, Dataset entries: %s",
                    symbol,
                    len(df.columns),
                    len(df),
                )

        return Data(object_ref=self.config.object_id, data=data_processed)


class FeaturesPSAR(FeatureGenerator):
    """Datapreprocessor for Classification Problems."""

    def create_features(self, data: Data) -> Data:
        """Calculates and adds features to the dataframes in the input dictionary. Optionally filters the output based on given timestamps."""

        logger.info("Start calculating features...")

        data_processed = {}
        atr_window = 14

        for symbol, df in data.data.items():

            df = df.copy()

            df["atr"] = ta.volatility.AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=atr_window, fillna=False
            ).average_true_range()

            df = df.iloc[atr_window - 1 :].copy()
            df = df.dropna()

            if not df.empty:
                data_processed[symbol] = df

            logger.debug(
                "Processed Symbol: %s. Calculated features: %s, Dataset entries: %s",
                symbol,
                len(df.columns),
                len(df),
            )

        logger.info("Calculation of features finished.")

        return Data(object_ref=self.config.object_id, data=data_processed)


class TargetSettings(DataProcessorSettings):
    """settings for target generator."""

    # target_value: float = 0.001  # 5m timeframe
    # target_value: float = 0.005  # 4h timeframe
    target_value: float = 0.01  # 4h timeframe
    # target_value: float = 0.03  # 1d timeframe


class TargetUpDownNo(TargetGenerator):
    """preprocessor for classification predicting"""

    def __init__(self, config: TargetSettings):
        self.config = config

    def create_target(self, data: Data) -> Data:
        """Creates the target variable as the return compared to the previous timestep."""

        logger.info("Start calculating target...")

        target_shift = 1  #  The number of timesteps to look ahead for the return calculation.
        target_series = {}
        # target is the return
        for symbol, df in data.data.items():

            ret: pd.Series = (
                (df["close"].shift(-target_shift, freq=self.config.timeframe) - df["close"]) / df["close"]
            ).dropna()

            # currently we do this before splitting into train and test set. this is not 100% correct
            # Calculate quantiles
            # q_low = ret.quantile(0.25)  # Lower 25% quantile as the threshold for down
            # q_high = ret.quantile(0.75)  # Upper 25% quantile as the threshold for up

            # # Define a function to avoid using cell variables in the lambda
            # def label_target(x, q_low=q_low, q_high=q_high):
            #     if x > q_high:
            #         return 1
            #     if x < q_low:
            #         return -1
            #     return 0

            # target = ret.apply(label_target)

            target = ret.apply(
                lambda x: 1 if x > self.config.target_value else -1 if x < -self.config.target_value else 0
            )

            target.name = "target"

            if len(target) > 0:
                target_series[symbol] = target

                logger.info(
                    "Calculation of targets finished for %s. Dataset entries: %s",
                    symbol,
                    len(target),
                )

        return Data(object_ref=self.config.object_id, data=target_series)


class TargetTrend(TargetGenerator):
    """preprocessor for classification predicting"""

    def __init__(self, config: TargetSettings):
        self.config = config

    def create_target(self, data: Data) -> Data:
        """Creates the target variable as the return compared to the previous timestep."""

        logger.info("Start calculating target...")

        target_series = {}
        # target is the return
        for symbol, df in data.data.items():

            ma = np.array(df["close"].rolling(window=28, center=True).mean())
            trend = []
            min_trend_length = 14
            lookahead = 1
            current_trend_inidices = []
            current_trend = 0

            for i in range(0, len(ma) - lookahead):
                if ma[i + lookahead] > ma[i]:
                    # flip the trend
                    if current_trend != 1:
                        if len(current_trend_inidices) >= min_trend_length:
                            trend.extend([current_trend] * len(current_trend_inidices))
                        else:
                            trend.extend([0] * len(current_trend_inidices))
                        current_trend_inidices = []

                    current_trend_inidices.append(i)
                    current_trend = 1

                elif ma[i + lookahead] < ma[i]:
                    if current_trend != -1:
                        if len(current_trend_inidices) >= min_trend_length:
                            trend.extend([current_trend] * len(current_trend_inidices))
                        else:
                            trend.extend([0] * len(current_trend_inidices))
                        current_trend_inidices = []

                    current_trend_inidices.append(i)
                    current_trend = -1

                else:
                    current_trend = 0
                    trend.append(current_trend)

            trend.extend([current_trend] * len(current_trend_inidices))
            trend.extend([0] * lookahead)  # add zero as last prediction since we have a lookahead

            target = pd.Series(index=df.index, data=trend, name="target")
            
            if len(target) > 0:
                target_series[symbol] = target

                logger.info(
                    "Calculation of targets finished for %s. Dataset entries: %s",
                    symbol,
                    len(target),
                )

        return Data(object_ref=self.config.object_id, data=target_series)


class TargetVolatilityDynamic(TargetGenerator):
    """Preprocessor for classification predicting volatility."""

    def __init__(self, config: TargetSettings):
        self.config = config
        self.volatility_window = 28
        self.min_threshold = 0.002  # 0.2% because of trading fees

    def create_target(self, data: Data) -> Data:
        """Creates the target variable as a binary indicator of high volatility."""

        logger.info("Start calculating target...")

        target_shift = 1  # Number of timesteps to look ahead
        target_series = {}

        for symbol, df in data.data.items():
            # Calculate returns
            ret = (df["close"].shift(-target_shift) - df["close"]) / df["close"]

            # Calculate historical volatility (e.g., using a rolling window)
            historical_volatility = ret.rolling(window=self.volatility_window).std()

            # Calculate rolling quantile (60th percentile) as a dynamic threshold
            rolling_dynamic_threshold = historical_volatility.rolling(window=self.volatility_window).quantile(0.85)

            # Ensure dynamic threshold is at least min_threshold
            rolling_dynamic_threshold = rolling_dynamic_threshold.where(
                rolling_dynamic_threshold >= self.min_threshold, self.min_threshold
            )

            # Create target based on whether returns exceed the rolling dynamic threshold
            target = ret.abs() > rolling_dynamic_threshold
            target.name = "target"
            target_series[symbol] = target.dropna().astype(int)

        logger.info("Calculation of target finished.")

        return Data(object_ref=self.config.object_id, data=target_series)


class TargetVolatility(TargetGenerator):
    """preprocessor for classification predicting. Only predicts if an action should happen or not."""

    def __init__(self, config: TargetSettings):
        self.config = config

    def create_target(self, data: Data) -> Data:
        """Creates the target variable as the return compared to the previous timestep."""

        logger.info("Start calculating target...")

        target_shift = 1  #  The number of timesteps to look ahead for the return calculation.
        target_series = {}
        # target is the return
        for symbol, df in data.data.items():

            ret: pd.Series = (
                (df["close"].shift(-target_shift, freq=self.config.timeframe) - df["close"]) / df["close"]
            ).dropna()
            target = ret.apply(lambda x: 1 if abs(x) > self.config.target_value else 0)
            target.name = "target"

            if len(target) > 0:
                target_series[symbol] = target

                logger.info(
                    "Calculation of targets finished for %s. Dataset entries: %s",
                    symbol,
                    len(target),
                )

        return Data(object_ref=self.config.object_id, data=target_series)


def process_symbol_features(
    df: pd.DataFrame,
    feature_functions: List[Callable[[pd.DataFrame], pd.DataFrame]],
    timeframe: pd.Timedelta,
) -> Optional[pd.DataFrame]:
    """Process features for a given symbol's dataframe using specified feature functions and filters.

    Args:
        df (pd.DataFrame): The dataframe containing OHLCV data for the symbol.
        feature_functions (List[Callable[[pd.DataFrame], pd.DataFrame]]): A list of functions to generate features.
        timeframe (pd.Timedelta): The trading timeframe (e.g., '1h', '4h') used for filtering based on volume thresholds.

    Returns:
        Tuple[Optional[str], Optional[pd.DataFrame]]:
            - The symbol (str) if processing was successful, otherwise None.
            - The processed dataframe (pd.DataFrame) if it contains valid data, otherwise None.
    """

    if len(df) < 28:
        return None

    # Generate features using the passed feature functions
    for func in feature_functions:
        df = func(df, timeframe)

    # Filter based on volume thresholds
    df = filter_low_volume_rows(
        df, timeframe=timeframe, volume_col="volume", price_col="close", threshold_24h=3_000_000
    )

    # Clean and filter dataframe
    df = cleaning_past_features(df)

    if df.empty:
        return None

    return df


def filter_low_volume_rows(
    df: pd.DataFrame,
    timeframe: pd.Timedelta,
    volume_col: str = "volume",
    price_col: str = "close",
    threshold_24h: int = 1_000_000,
) -> pd.DataFrame:
    """Filters out symbols based on trading volume conditions

    1. Trading volume over the last 24 hours should be >= $1,000,000.
    2. The last 2 timestamps should have trading volume >= $100,000.

    Args:
        df (pd.DataFrame): The DataFrame containing the trading data.
        volume_col (str): Column name representing the trading volume.
        price_col (str): Column name representing the price.
        threshold_24h (int/float): The minimum volume required over the last 24 hours.
        timeframe (pd.Timedelta): The timeframe for the data, e.g., '1H', '4H', '15T' (15 minutes).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """

    # Calculate price volume
    df["price_volume"] = df[price_col] * df[volume_col]

    # Calculate the number of periods in 24 hours based on the timeframe
    periods_per_24h = int(pd.Timedelta("24h") / timeframe)

    threshold = threshold_24h / periods_per_24h

    filtered_df = df[df["price_volume"] >= threshold].copy()

    # Drop the auxiliary columns
    filtered_df = filtered_df.drop(columns=["price_volume"])

    return filtered_df


def resample_with_forward_fill(df: pd.DataFrame, timeframe: pd.Timedelta) -> pd.DataFrame:
    """Resamples the given DataFrame to the specified frequency, forward-filling missing values and ensuring
    that the last period aligns with the resampling frequency.

    By extending the date range to fully cover the last date,
    this function ensures that the final period is correctly filled.

    e.g.

    2022-12-29 00:00:00+00:00	28.0
    2022-12-30 00:00:00+00:00	28.0
    2022-12-31 00:00:00+00:00	25.0
    2023-01-01 00:00:00+00:00	26.0

    becomes:
    ...
    2022-12-31 20:00:00+00:00	25.0
    2023-01-01 00:00:00+00:00	26.0
    2023-01-01 04:00:00+00:00	26.0
    2023-01-01 08:00:00+00:00	26.0
    2023-01-01 12:00:00+00:00	26.0
    2023-01-01 16:00:00+00:00	26.0
    2023-01-01 20:00:00+00:00	26.0

    Args:
        df (pd.DataFrame): The input DataFrame with a DateTime index to be resampled.
        freq (str): The desired resampling frequency (e.g., '4h', '1h', 'D').

    Returns:
        pd.DataFrame: The resampled DataFrame with missing values forward-filled.
    """

    # Find the last date in the DataFrame
    last_date = df.index[-1]

    # Calculate the next date after the last date that aligns with the resampling frequency
    next_date = last_date + pd.Timedelta("1d") - timeframe

    # Extend the date range to include this next resampling date
    extended_index = pd.date_range(start=df.index.min(), end=next_date, freq=timeframe)

    # Reindex the DataFrame to include the extended date range
    df_resampled = df.reindex(extended_index, method="ffill")

    return df_resampled


def ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Function to calculate technical analysis indicators using the TA library."""

    # Periods for Bollinger Bands and RSI
    periods = [3, 7, 14]

    for period in periods:
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df["close"], window=period)

        # Calculate features based on Bollinger Bands
        df[f"BBANDS_{period}_pct_b"] = (df["close"] - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband()
        )
        df[f"BBANDS_{period}_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        # Replace NaN values with 0. happens if the close price does not change over whole window
        df.fillna({f"BBANDS_{period}_pct_b": 0}, inplace=True)
        df.fillna({f"BBANDS_{period}_width": 0}, inplace=True)

        # Calculate RSI
        df[f"RSI_{period}"] = ta.momentum.RSIIndicator(close=df["close"], window=period).rsi()

        # Calculate ATR
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=period
        ).average_true_range()

        df[f"atr_ratio_{period}"] = atr / df["close"]

    # calcualte volume_price_trend
    volume_price_trend = ta.volume.volume_price_trend(df["close"], df["volume"], fillna=False, smoothing_factor=2)
    df["volume_price_trend_ratio"] = volume_price_trend / df["close"]

    # calculate vwap
    vwap = ta.volume.volume_weighted_average_price(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=28
    )
    ma = df["close"].rolling(window=28).mean()
    df["vwap_ratio"] = vwap / ma

    df["psar.2"] = indicator_psar_up(df, step=0.2, lookback=28)
    df["psar.02"] = indicator_psar_up(df, step=0.02, lookback=28)
    df["psar.002"] = indicator_psar_up(df, step=0.002, lookback=28)
    df["psar.0002"] = indicator_psar_up(df, step=0.0002, lookback=28)

    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    return df


def indicator_psar_up(df: pd.DataFrame, step: float, lookback: int):
    """Custom psar trend indicator."""

    n = len(df)

    # Initialize the result array with NaN values for the first `lookback` entries
    psar_up_values = np.full(n, np.nan)  # Use NumPy to create an array of NaN

    # Loop through the rest of the data and calculate PSAR
    for i in range(lookback, n):
        df_sub = df.iloc[i - lookback : i]

        indicator = IncrementalPSARIndicator(step)

        for row in df_sub.iterrows():
            indicator.update(high=row[1]["high"], low=row[1]["low"], close=row[1]["close"], date=row[0])

        # Get the PSAR value and store it in the pre-allocated array
        psar_up_values[i] = indicator.psar_up()  # Get the last PSAR up value

    # Create a pandas Series with the calculated PSAR values
    psar_up = pd.Series(psar_up_values, index=df.index, name="psarup")

    psar_up.fillna(0, inplace=True)

    return psar_up / df["close"]


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
        timeframe: Optional[pd.Timedelta] = None,
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

        if self._timeframe:
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
        return self._up_trend

    def psar_up(self) -> pd.Series:
        """Return the PSAR uptrend values."""
        return self._psar_up

    def psar_down(self) -> pd.Series:
        """Return the PSAR downtrend values."""
        return self._psar_down


def percentage_change(
    df: pd.DataFrame,
    col_name: str,
    timeframe: pd.Timedelta,
    drop: bool = True,
    new_column: Optional[str] = None,
    shift: int = 1,
) -> pd.DataFrame:
    """Calculates the percentage change of a column.

    Args:
        df (pd.DataFrame): Input dataframe containing the column to calculate percentage change for.
        col_name (str): The name of the column to calculate percentage change for.
        timeframe (Optional[pd.Timedelta]): the timeframe of the data
        drop (bool): Whether to drop the original column after calculating the percentage change.
        new_column (Optional[str]): The name of the new column to store the percentage change. If None, a default name is used.
        shift (int): The number of periods to shift for the percentage change calculation.

    Returns:
        pd.DataFrame: Dataframe with the percentage change column added.
    """

    if new_column is None:
        new_column = f"{col_name}_%change_{shift}"

    df[new_column] = (df[col_name] - df[col_name].shift(shift, freq=timeframe)) / df[col_name].shift(
        shift, freq=timeframe
    )

    if drop:
        df.drop(columns=[col_name], inplace=True)

    return df


def rolling_feature(
    df: pd.DataFrame,
    metric: str,
    timeframe: pd.Timedelta,
    drop: bool = True,
    new_column: Optional[str] = None,
    window: int = 7,
    min_periods: Optional[int] = None,
    operation: str = "mean",
) -> pd.DataFrame:
    """Calculates a rolling function on a specified metric.

    Args:
        df (pd.DataFrame): Input dataframe containing the metric column.
        metric (str): The name of the metric column to apply the rolling function to.
        timeframe pd.Timedelta]: provide a timeframe for specifying the frequency
        drop (bool): Whether to drop the original metric column after the calculation.
        new_column (Optional[str]): The name of the new column to store the rolling function result. If None, a default name is used.
        window (int): The window size for the rolling function in multiples of the dataframe's timeframe.
        min_periods (Optional[int]): Minimum number of observations in window required to have a value. Defaults to the number of periods in the window.
        operation (str): The aggregation operation to apply (e.g., 'mean', 'sum').

    Returns:
        pd.DataFrame: Dataframe with the rolling function column added.
    """

    if new_column is None:
        new_column = f"{metric}_{operation}_{window}"

    if min_periods is None:
        min_periods = window

    df[new_column] = df[metric].rolling(window=timeframe * window, min_periods=min_periods).aggregate(operation)

    if drop:
        df.drop(columns=[metric], inplace=True)

    return df


def consecutive_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Function to calculate the number of preceeding consecutive green and red days for a coin."""

    df = df.copy()
    df["consecutive_green_candles"] = 0
    df["consecutive_red_candles"] = 0

    for i in range(0, len(df)):
        if df.loc[df.index[i], "mom/return_1"] > 0:
            df.loc[df.index[i], "consecutive_green_candles"] = df.loc[df.index[i - 1], "consecutive_green_candles"] + 1
        elif df.loc[df.index[i], "mom/return_1"] < 0:
            df.loc[df.index[i], "consecutive_red_candles"] = df.loc[df.index[i - 1], "consecutive_red_candles"] + 1

    df["consecutive_candles"] = (df["consecutive_green_candles"] - df["consecutive_red_candles"]).shift(1)
    df.drop(columns=["consecutive_red_candles", "consecutive_green_candles"], inplace=True)

    return df


def kernel_regression(src: np.ndarray, h: float, r: float, x_0: int) -> np.ndarray:
    """
    Perform kernel regression on the given source data.

    Args:
        src (np.ndarray): Source data for regression.
        h (float): Bandwidth parameter for the kernel.
        r (float): Power parameter for the kernel.
        x_0 (int): Number of entries to consider for each regression.

    Returns:
        np.ndarray: The estimated y values after performing kernel regression.
    """

    yhat = np.full_like(src, np.nan)

    for i in range(x_0, len(src)):
        current_weight = 0.0
        cumulative_weight = 0.0

        for k, j in enumerate(range(i - x_0 + 1, i + 1), 0):  # consider the last x_0 bars
            distance = i - j  # distance between samples
            y = src[j]
            w = (1 + (distance**2) / (h**2 * 2 * r)) ** (-r)
            current_weight += y * w
            cumulative_weight += w

        yhat[i] = current_weight / cumulative_weight

    return yhat


def generate_price_related_features(df: pd.DataFrame, timeframe: pd.Timedelta) -> pd.DataFrame:
    """Generate price-related features for the given dataframe."""

    # Percentage change features
    for shift in [1, 2, 3, 7, 14, 21, 28]:
        df = percentage_change(
            df, "close", drop=False, new_column=f"mom/return_{shift}", shift=shift, timeframe=timeframe
        )

    # Rolling statistical features using your existing _rolling_feature function
    df = rolling_feature(
        df, metric="mom/return_1", drop=False, new_column="vol/return_3", window=3, operation="std", timeframe=timeframe
    )
    df = rolling_feature(
        df, metric="mom/return_1", drop=False, new_column="vol/return_7", window=7, operation="std", timeframe=timeframe
    )
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/return_14",
        window=14,
        operation="std",
        timeframe=timeframe,
    )

    # Rolling min, max, and mean features
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/max_return_7",
        window=7,
        operation="max",
        timeframe=timeframe,
    )
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/min_return_7",
        window=7,
        operation="min",
        timeframe=timeframe,
    )
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/max_return_14",
        window=14,
        operation="max",
        timeframe=timeframe,
    )
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/min_return_14",
        window=14,
        operation="min",
        timeframe=timeframe,
    )
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/mean_return_7",
        window=7,
        operation="mean",
        timeframe=timeframe,
    )
    df = rolling_feature(
        df,
        metric="mom/return_1",
        drop=False,
        new_column="vol/mean_return_3",
        window=3,
        operation="mean",
        timeframe=timeframe,
    )

    return df


def generate_volume_related_features(df: pd.DataFrame, timeframe: pd.Timedelta) -> pd.DataFrame:
    """Generate volume-related features for the given dataframe."""

    # Percentage change for volume
    for shift in [1, 2, 3]:
        df = percentage_change(
            df, "volume", drop=False, new_column=f"volume/pct_{shift}", shift=shift, timeframe=timeframe
        )

    # SMA features
    sma3 = df["volume"].rolling(3).mean()
    sma7 = df["volume"].rolling(7).mean()
    sma14 = df["volume"].rolling(14).mean()
    df["volume/sma7_sma14"] = sma7 / sma14
    df["volume/sma3_sma14"] = sma3 / sma14

    # Rolling standard deviation for volume using the existing _rolling_feature function
    df = rolling_feature(
        df,
        metric="volume/pct_1",
        drop=False,
        new_column="volume/std_7",
        window=7,
        operation="std",
        timeframe=timeframe,
    )
    df = rolling_feature(
        df,
        metric="volume/pct_1",
        drop=False,
        new_column="volume/std_14",
        window=14,
        operation="std",
        timeframe=timeframe,
    )

    return df


def generate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical indicators like consecutive candles, TA indicators, and high-low relationships."""

    df = consecutive_candles(df)
    df = ta_indicators(df)

    # High-low and open-close relationships
    df["vol/highlow"] = (df["high"] - df["low"]).rolling(10).mean() / df["low"].rolling(10).mean()
    df["vol/openclose"] = (df["open"] - df["close"]).rolling(5).mean() / df["close"].rolling(5).mean()
    df["vol/closelowconvergence"] = df["close"] / df["low"]
    df["vol/closehighconvergence"] = df["close"] / df["high"]
    df["vol/closeopenconvergence"] = df["close"] / df["open"]
    df["vol/highlowconvergence"] = df["high"] / df["low"]

    return df


def generate_kernel_features(df: pd.DataFrame, timeframe: pd.Timedelta) -> pd.DataFrame:
    """Generate kernel regression features for the dataframe."""

    df["kernel_fast"] = kernel_regression(np.array(df["close"]), h=10, r=0.5, x_0=14)
    df["kernel_slow"] = kernel_regression(np.array(df["close"]), h=25, r=8, x_0=28)
    df["k_ratio"] = df["kernel_fast"] / df["kernel_slow"]
    df["close_ratio_kf"] = df["close"] / df["kernel_fast"]
    df["close_ratio_ks"] = df["close"] / df["kernel_slow"]
    df = percentage_change(df, "kernel_fast", drop=False, new_column="kf_change_3", shift=3, timeframe=timeframe)
    df = df.drop(columns=["kernel_fast", "kernel_slow"])
    return df


def generate_ribbon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate SMA ribbon-based features for the dataframe."""

    sma_high = df["high"].rolling(28).mean()
    sma_low = df["low"].rolling(28).mean()
    sma_14 = df["close"].rolling(14).mean()
    df["vol/sma_fast_above_sma_high"] = sma_high / sma_14
    df["vol/sma_high_low"] = sma_high / sma_low
    df["vol/low_within_ribbon"] = sma_high / df["low"]
    df["vol/close_above_ribbon"] = sma_high / df["close"]
    return df


def generate_stochastic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate stochastic oscillator features for the dataframe."""

    high14 = df["high"].rolling(window=14).max()
    low14 = df["low"].rolling(window=14).min()
    df["stochastic_k"] = 100 * ((df["close"] - low14) / (high14 - low14))
    df["stochastic_d"] = df["stochastic_k"].rolling(window=7).mean()
    return df


def cleaning_past_features(df: pd.DataFrame) -> pd.DataFrame:
    """Function to clean dataset after feature creation (to be implemented)."""

    # Drop rows with any NaN values
    df = df.dropna()

    # Drop the specified columns
    df = df.drop(columns=["open", "high", "low", "volume", "symbol"])

    return df


def add_sinusoidal_time_features(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """Adds sinusoidal date-time features to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a DateTime index or a DateTime column.
        date_col (str): Optional. Column name if the DateTime is in a column instead of the index.

    Returns:
        pd.DataFrame: DataFrame with added sinusoidal features for hour, day of week, day of year, and month.
    """
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    # Extracting time components
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month

    # Applying sinusoidal transformations
    # df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

    # Drop original time components if not needed
    df = df.drop(["hour", "day_of_week", "day_of_year", "month"], axis=1)

    return df
