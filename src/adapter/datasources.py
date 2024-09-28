"""Adapters for datasources"""

import concurrent.futures
import os
import time
from typing import List, Optional

import ccxt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.common_packages import create_logger
from src.core.datasource import Data, Datasource, DatasourceSettings

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class ExchangeDatasourceSettings(DatasourceSettings):
    """Settings for datasource"""

    exchange_id: str = "bitget"
    timeframe: pd.Timedelta = pd.Timedelta("4h")
    current_data_scrape_limit: int = 29


class ExchangeDatasource(Datasource):
    """Datasource for the exchange."""

    def __init__(self, config: ExchangeDatasourceSettings):
        """Initialize the DataSource object with specific parameters."""

        super().__init__(config)

        self.config = config
        self.exchange = self._load_exchange()

        if self.config.symbols is None:
            self._determine_symbols(number=20, threshold=2_000_000)

    def _load_exchange(self) -> ccxt.Exchange:
        """Loads the exchange."""

        exchange_class = getattr(ccxt, self.config.exchange_id)
        exchange_config = {
            "timeout": 30000,
            "enableRateLimit": True,
        }

        # necessary for bitget futures
        if self.config.exchange_id == "bitget":
            exchange_config["options"] = {"defaultType": "swap"}

        return exchange_class(exchange_config)

    def _determine_symbols(self, number: int = 50, threshold: int = 5_000_000) -> None:
        """Determines and updates the top symbols with the highest trading volumes. Only available in live trading.

        This method fetches ticker data from the exchange to find symbols that end with "USDT."
        It then fetches OHLCV data for these symbols, calculates the daily volume (as the product
        of closing price and volume), and filters symbols where the daily volume consistently
        exceeds the specified threshold over the past 7 days.

        After filtering, the symbols are sorted by their average volume in descending order,
        and the top 'n' symbols are selected for trading.

        Args:
            number (int): The number of symbols to select based on highest average volume. Defaults to 10.
            threshold (int): The minimum daily volume (calculated as close * volume) a symbol must have over the last 7 days to be considered. Defaults to 1,000,000.

        Returns:
            None: Updates the `self.config.symbols` attribute with the top symbols that meet the volume criteria.
        """

        logger.info(
            "Symbols to trade get selected based on their past volume. The top %d coins with a volume >%s are selected.",
            number,
            f"{threshold:,}",
        )

        # find all possible symbols
        tickers = self.exchange.fetch_tickers()
        all_symbols = [symbol for symbol in tickers if symbol.endswith("USDT")]

        # fetch the data for all symbols and select the n best coins with the highest volume
        result = self._fetch_ohlcv(symbols=all_symbols, limit=7, timeframe=pd.Timedelta("1d"))

        volumes = []
        for symbol, df in result.data.items():
            vol = df["close"] * df["volume"]
            if all(vol > threshold):
                volumes.append((symbol, vol.mean().item()))

        volumes = sorted(volumes, key=lambda x: x[1], reverse=True)

        self.config.symbols = [symbol for symbol, _ in volumes[:number]]

        logger.info("The following symbols are going to be traded %s", self.config.symbols)

    # how to handle the symbols?

    def _scrape_mock_data_historic(self, symbols: List[str]) -> Data:
        """Scrape historic data from mock data.

        Returns:
            Dict[str, pd.DataFrame]: Historic data from mock data up to the current date.
        """

        data = {}
        for symbol, df in self.mock_data.data.items():
            if symbols is None or symbol in symbols:
                df = df.loc[: self.simulation_current_date].iloc[:-1].copy()
                
                # only return the symbols with enough data
                if len(df) >= self.config.current_data_scrape_limit:
                    data[symbol] = df

        return Data(data=data, object_ref=self.config.object_id)

    def _scrape_real_data_historic(self, symbols: List[str]) -> Data:
        """Scrape historic data from the real exchange.

        Returns:
            Dict[str, pd.DataFrame]: Historic data from the real exchange.
        """

        return self._fetch_ohlcv(symbols=symbols, start=self.config.scrape_start_date, end=self.config.scrape_end_date)

    def _scrape_mock_data_current(self, symbols: List[str]) -> Data:
        """Scrape and return historical OHLCV data up to the most recently completed candle.

        This method retrieves OHLCV data from the mock data source for the current simulation time,
        ensuring that only fully completed candles are returned to prevent data leakage. The current
        simulation date is then advanced by the configured timeframe. If the simulation date exceeds
        the end date, the simulation is marked as complete.

        Args:
            symbols (List[str]): provide symbols from whom to fetch the data

        """

        data = {}

        for symbol, df in self.mock_data.data.items():

            if symbols is None or symbol in symbols:
                df = df.loc[: self.simulation_current_date][-self.config.current_data_scrape_limit :].copy()

                # only return the symbols with enough data
                if len(df) >= self.config.current_data_scrape_limit:
                    data[symbol] = df

        self.simulation_current_date += pd.Timedelta(self.config.timeframe)

        if self.simulation_current_date > self.config.simulation_end_date:
            self.simululation_end = True

        return Data(data=data, object_ref=self.config.object_id)

    def _scrape_real_data_current(self, symbols: List[str]) -> Data:
        """Scrape current data from the real exchange.

        Args:
            symbols (Optional[List[str]]): optionally provide symbols from whom to fetch the data

        Returns:
            Data: Current data from the real exchange.
        """

        return self._fetch_ohlcv(symbols=symbols)

    def _fetch_ohlcv(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
        timeframe: Optional[pd.Timedelta] = None,
    ) -> Data:
        """Fetch historical OHLCV data for multiple symbols within a specified timeframe in parallel.

        Args:
            symbols (list): List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
            start (pd.Timestamp): Start timestamp.
            end (Optional[pd.Timestamp]): End timestamp.
            limit (Optional[int]): Number of entries to fetch
            timeframe (Optional[pd.Timedelta]): Optionally provide a timeframe, e.g. 1m, 5m, 1h, 1d,..

        Returns:
            Dict[str, pd.DataFrame]: Dict of dataframes containing formatted OHLCV data for all symbols.
        """
        data = {}
        symbols = symbols or self.config.symbols

        if not symbols:
            logger.warning("No symbols provided. Trading settings can't be adjusted.")
            return Data(object_ref=self.config.object_id, data=data)


        if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
            logger.error("Invalid symbols format. Symbols should be a list of strings.")
            return Data(object_ref=self.config.object_id, data=data)


        # Current time in milliseconds
        current_time_in_ms = int(time.time() * 1000)

        # Define a helper function for fetching and formatting data for a single symbol
        def fetch_and_format(symbol):
            try:
                ohlcv = self._fetch_single_ohlcv(
                    symbol, current_time_in_ms, start=start, end=end, limit=limit, timeframe=timeframe
                )
                return symbol, self._format_data(ohlcv, symbol)
            except Exception as e:
                logger.error("Failed to fetch data for %s - %s: %s", symbol, self.config.timeframe, e)
                return symbol, None

        # Fetch data in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_and_format, symbol): symbol for symbol in symbols}

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(symbols), desc="Fetching OHLCV data", unit="symbol"
            ):
                symbol, result = future.result()
                if result is not None:
                    data[symbol] = result

        if data is None:
            raise ValueError("Non data fetched.")

        # only return the symbols with enough data
        data = {key: values for key, values in data.items() if len(values) >= self.config.current_data_scrape_limit}

        for key, values in data.items():
            logger.info("Fetched Symbol: %s. Dataset entries: %s", key, len(values))

        return Data(object_ref=self.config.object_id, data=data)

    def _fetch_single_ohlcv(
        self,
        symbol: str,
        current_time_in_ms: int,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
        timeframe: Optional[pd.Timedelta] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ) -> np.ndarray:
        """Fetch OHLCV (Open-High-Low-Close-Volume) history for a given symbol and timeframe.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
            current_time_in_ms (int): current time in ms.
            start (pd.Timestamp): Start timestamp.
            end (Optional[pd.Timestamp]): End timestamp.
            limit (Optional[int]): Number of entries to fetch
            timeframe (Optional[pd.Timedelta]): Optionally provide a timeframe
            max_retries (int): Maximum number of retries in case of failure.
            retry_delay (float): Delay between retries in seconds.

        Returns:
            numpy.ndarray: Array containing OHLCV data.

        Note: the data is shifted by one timeframe to avoid a lookahead.
            Therefore, the timestamps on the exchange do not match the timestamps of the scraped data.
        """

        timeframe = timeframe or self.config.timeframe
        timeframe = extract_timedelta_str(timeframe)

        if start is not None:
            since = int(start.timestamp()) * 1000
            until = (int(end.timestamp()) * 1000) if end is not None else None

            diff = int(pd.Timedelta("28d").total_seconds() * 1000)
            stop_time = until or (int(pd.Timestamp.now(tz="UTC").timestamp()) * 1000)
            ohlcv_all = []
            params = {"until": until}
            # TODO: fix end date, this is currently not handled correctly. We can only scrape up until the current date

            while True:
                retries = 0
                while retries < max_retries:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(
                            symbol=symbol, timeframe=timeframe, since=since, limit=None, params=params
                        )
                        break  # Break out of the retry loop if successful
                    except Exception as e:
                        retries += 1
                        logger.error(
                            "Failed to fetch OHLCV for %s - %s: %s (attempt %d/%d)",
                            symbol,
                            timeframe,
                            e,
                            retries,
                            max_retries,
                        )
                        if retries < max_retries:
                            time.sleep(retry_delay * retries)  # Wait before retrying
                        else:
                            return np.array([])  # Return an empty array if max retries exceeded

                ohlcv_all.extend(ohlcv)

                if len(ohlcv_all) == 0:
                    if since >= stop_time:
                        logger.warning("No data found for symbol: %s", symbol)
                        break

                    since += diff
                elif len(ohlcv) == 0:
                    break
                else:
                    since = ohlcv[-1][0] + 1

                if len(ohlcv) == 1 or (until is not None and ohlcv[-1][0] >= until):
                    break

        else:
            retries = 0
            limit = limit or self.config.current_data_scrape_limit

            while retries < max_retries:
                try:
                    ohlcv_all = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit + 1)
                    break  # Break out of the retry loop if successful
                except Exception as e:
                    retries += 1
                    logger.error(
                        "Failed to fetch OHLCV for %s - %s: %s (attempt %d/%d)",
                        symbol,
                        self.config.timeframe,
                        e,
                        retries,
                        max_retries,
                    )
                    if retries < max_retries:
                        time.sleep(retry_delay)  # Wait before retrying
                    else:
                        return np.array([])  # Return an empty array if max retries exceeded

        # Shift timestamp by one timeframe to avoid lookahead
        timeframe_in_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000)
        for i in range(len(ohlcv_all)):
            ohlcv_all[i][0] += timeframe_in_ms

        # Check if no end date is specified and if the last candle might be incomplete (timestamp in the future)
        if end is None and len(ohlcv_all) > 0:
            last_candle_time = ohlcv_all[-1][0]
            if last_candle_time > current_time_in_ms:
                ohlcv_all = ohlcv_all[:-1]

        return np.array(ohlcv_all)

    def _format_data(self, ohlcv: np.ndarray, symbol: str) -> pd.DataFrame:
        """Format raw OHLCV data into a Pandas DataFrame.

        Args:
            ohlcv (numpy.ndarray): Array containing OHLCV data.
            symbol (str): Trading pair symbol.

        Returns:
            pandas.DataFrame: Formatted DataFrame with columns 'time', 'open', 'high', 'low', 'close', 'volume'.
        """
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["symbol"] = symbol
        df.set_index("time", inplace=True)
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)
        df = df.astype(
            {"open": "float32", "high": "float32", "low": "float32", "close": "float32", "volume": "float32"}
        )

        return df


def extract_timedelta_str(timedelta: pd.Timedelta) -> str:
    """extracts the string from a timedelta object.

    # Examples
    td_4h = pd.Timedelta("4h")
    td_15min = pd.Timedelta("15min")
    td_1d = pd.Timedelta("1d")

    print(extract_timedelta_str(td_4h))   # Output: "4h"
    print(extract_timedelta_str(td_15min))  # Output: "15m"
    print(extract_timedelta_str(td_1d))    # Output: "1d"
    """

    components = timedelta.components

    if components.days > 0:
        return f"{components.days}d"
    elif components.hours > 0:
        return f"{components.hours}h"
    elif components.minutes > 0:
        return f"{components.minutes}m"
    else:
        raise ValueError("No valid format.")


class FearGreedDatasourceSettings(DatasourceSettings):
    """Settings for fear and greed datasource"""

    timeframe: pd.Timedelta = pd.Timedelta("4h")
    current_data_scrape_limit: int = 29


class FearGreedDataSource(Datasource):
    """Datasource for the exchange."""

    def __init__(self, config: ExchangeDatasourceSettings):
        """Initialize the DataSource object with specific parameters."""

        super().__init__(config)

        self.config = config

    def _scrape_mock_data_historic(self, symbols: List[str]) -> Data:
        """Scrape historic data from mock data.

        Returns:
            Dict[str, pd.DataFrame]: Historic data from mock data up to the current date.
        """

        df = self.mock_data.data.get("all").loc[: self.simulation_current_date].iloc[:-1].copy()

        return Data(data={"all": df}, object_ref=self.config.object_id)

    def _scrape_real_data_historic(self, symbols: List[str]) -> Data:
        """Scrape historic data from the datasource.

        Returns:
            Data: Historic data
        """

        return self.fetch_data(start=self.config.scrape_start_date, end=self.config.scrape_end_date)

    def _scrape_mock_data_current(self, symbols: List[str]) -> Data:
        """Scrape and return historical data up to the most recently completed candle.

        This method retrieves data from the mock data source for the current simulation time,
        ensuring that only fully completed candles are returned to prevent data leakage. The current
        simulation date is then advanced by the configured timeframe. If the simulation date exceeds
        the end date, the simulation is marked as complete.

        Args:
            symbols (List[str]): provide symbols from whom to fetch the data

        """

        df = (
            self.mock_data.data.get("all")
            .loc[: self.simulation_current_date][-self.config.current_data_scrape_limit :]
            .copy()
        )

        self.simulation_current_date += pd.Timedelta(self.config.timeframe)

        if self.simulation_current_date > self.config.simulation_end_date:
            self.simululation_end = True

        return Data(data={"all": df}, object_ref=self.config.object_id)

    def _scrape_real_data_current(self, symbols: List[str]) -> Data:
        """Scrape current data from the real exchange.

        Returns:
            Data: Current fear and greed index from the api.
        """

        return self.fetch_data()

    def fetch_data(self, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> Data:
        """scrape all data from fear and greed index.

        Fear and greed index data from alternative.me. Data is only available on daily basis

        Args:
            start (Optional[pd.Timestamp]): optional start date for scraping.
            end: (Optional[pd.Timestamp]): optional end date for scraping.
        """

        if start:
            now = pd.Timestamp.now(tz="UTC")
            limit = (now - start).days + 1
        else:
            limit = self.config.limit_current

        response = requests.get(f"https://api.alternative.me/fng/?limit={limit}", timeout=10)
        result = response.json()

        df = self._format_data(result)

        if end:
            df = df.loc[:end].copy()

        return Data(data={"all": df}, object_ref=self.config.object_id)

    def _format_data(self, result: pd.DataFrame) -> pd.DataFrame:
        """Format raw OHLCV data into a Pandas DataFrame.

        Args:
            ohlcv (pd.DataFrame): dataframe containing fear and greed data as dict

        Returns:
            pandas.DataFrame: Formatted DataFrame
        """

        df = pd.DataFrame(result["data"])
        df[["timestamp", "value"]] = df[["timestamp", "value"]].astype("float32")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.rename(columns={"timestamp": "time", "value": "fear_n_greed"})
        df = df.drop(["time_until_update", "value_classification"], axis=1)
        df = df.set_index("time")
        df = df.sort_index()

        return df
