"""Mock exchange for simluating an exchange."""

import json
import os
import uuid
from importlib import import_module
from typing import Dict, List, Optional, Union

import ccxt
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from src.adapter.datasources import ExchangeDatasource, ExchangeDatasourceSettings
from src.common_packages import CustomJSONEncoder, create_logger, timestamp_decoder
from src.core.base import BaseConfiguration, ObjectId
from src.core.datasource import Data

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)

MOCK_DATA_FILE = "mock_exchange.joblib"
CONFIG_FILE = "mock_exchange_config.json"


class MockExchangeSettings(BaseModel):
    """
    Data class for storing configuration parameters for a mock exchange.

    Args:
        data_path (Optional[str]):
            Path to the directory or file where the mock data is stored.

        simulation_start_date (pd.Timestamp):
            The start time for the simulation.

        simulation_end_date (pd.Timestamp):
            The end time for the simulation.

        timeframe (str):
            The time interval for the data scraping, specified as a string.
            Default is '4h', meaning four-hour intervals.

        config_dir (Optional[str]):
            Path to the configuration directory.

        symbols (List[str]):
            A list of trading symbols (e.g., ['BTC/USD', 'ETH/USD']) that the
            simulation will operate on.
    """

    object_id: ObjectId
    data_directory: str
    scrape_start_date: pd.Timestamp
    scrape_end_date: Optional[pd.Timestamp] = None
    simulation_start_date: pd.Timestamp
    simulation_end_date: pd.Timestamp  # TODO: Simulation End date is not used anywhere
    timeframe: pd.Timedelta = pd.Timedelta("4h")
    symbols: Optional[List[str]] = None

    exchange_config: Dict  # those are the actual values used for initialising a real ccxt exchangee

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator(
        "simulation_start_date",
        "simulation_end_date",
        "scrape_start_date",
        "scrape_end_date",
        mode="before",
    )
    def check_timestamp(cls, value):  # pylint: disable=no-self-argument
        """check if timestamps are set correctly."""

        if isinstance(value, pd.Timestamp):
            if value != value.round(pd.Timedelta("1d")):
                raise ValueError(f"{value} must be defined as a full day: like '2023-08-08T00:00:00+00:00'")

        return value


class MockExchangeConfiguration(BaseConfiguration):
    """Configuration specific to a model."""

    component_type: str = Field(
        default="mock_exchange",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: MockExchangeSettings


# TODO: Hardcoded value for the exchange.
class MockExchange(ccxt.bitget):
    """Mock exchange class for bitget."""

    def __init__(self, config: MockExchangeSettings):

        super().__init__(config.exchange_config)
        self.config = config

        self.positions = {}
        self.pending_orders = []
        self.trade_history = []

        # Initialize balance
        self.balance_total = 1000
        self.balance_free = self.balance_total
        self.locked_balance = 0.0  # Balance reserved for limit orders

        self.current_date = None
        self.current_data = None

        self.data = None

        if os.path.exists(os.path.join(self.config.data_directory, MOCK_DATA_FILE)):
            self.load_mock_data()

        self.order_fees = self._get_order_fees()

    def _get_order_fees(self) -> Dict:
        """get the order fee structure for the exchange."""

        order_fees = {}
        self.load_markets()
        for symbol in self.config.symbols:
            market_info = self.market(symbol)
            order_fees[symbol] = {
                "maker": market_info["maker"],
                "taker": market_info["taker"],
            }
            logger.info("Fetched order details for symbol %s. %s", symbol, order_fees[symbol])

        return order_fees

    def create_and_save_mock_data(self):
        """Function that creates and stores the mock data.

        Next time the mock exchange gets initialized, the data is automatically loaded."""

        # create an exchange datasource and fetch the data and store it.
        ex_ds_config = ExchangeDatasourceSettings(
            object_id=ObjectId(value="mock_exchange_datasource"),
            exchange_id="bitget",
            symbols=self.config.symbols,
            scrape_start_date=self.config.scrape_start_date,
            scrape_end_date=self.config.scrape_end_date,
            timeframe=self.config.timeframe,
        )

        exchange_datasource = ExchangeDatasource(ex_ds_config)

        self.data: Data = exchange_datasource.scrape_data_historic()

        mock_data_path = os.path.join(self.config.data_directory, MOCK_DATA_FILE)
        exchange_datasource.save_mock_data(self.data, mock_data_path)
        self.export_config()

    def load_mock_data(self) -> None:
        """Load mock data from a joblib file."""

        joblib_file = os.path.join(self.config.data_directory, MOCK_DATA_FILE)

        if not os.path.exists(joblib_file):
            logger.error("Data file does not exist at %s", joblib_file)
            raise ValueError("Mock data needs to get created first.")

        logger.info("Loading historic mock dataset from: %s", joblib_file)

        data: Data = joblib.load(joblib_file)

        # Ensure that each entry is a DataFrame and set the index correctly
        for symbol, df in data.data.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"The data for {symbol} is not a pandas DataFrame.")
            if "close" not in df.columns:
                raise ValueError(f"Data for symbol {symbol} does not contain 'close' column")

        self.data = data

        logger.info("Exchange mock data loaded successfully.")

        self.next_step()

    def _create_configuration(self) -> MockExchangeConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{MockExchangeConfiguration.__module__}.{MockExchangeConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return MockExchangeConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            settings_path=settings_path,
            config_path=config_path,
            settings=self.config,
        )

    def export_config(self) -> None:
        """Export configuration as a json file."""

        config = self._create_configuration()

        file_path = os.path.join(self.config.data_directory, CONFIG_FILE)

        with open(file_path, "w", encoding="utf8") as file:
            json.dump(config.model_dump(), file, indent=4, cls=CustomJSONEncoder)

        logger.info("Configuration stored at %s", file_path)

    def save_trade_history(self, path: str) -> None:
        """Save the trade history to a specified file path.

        Args:
            path (str): The file path where the trade history will be saved.
        """

        with open(path, "w", encoding="utf8") as file:
            json.dump(self.trade_history, file, indent=4, default=str)
        logger.info("Trade history saved to: %s", path)

    def next_step(self):
        """set the next date and data"""

        if self.current_date is None:
            self.current_date = self.config.simulation_start_date
        else:
            # set current date
            self.current_date += self.config.timeframe

        # set current data based on the current date
        self.current_data = {symbol: df[df.index == self.current_date] for symbol, df in self.data.data.items()}

        logger.info("Simulation current date %s", self.current_date)

    def check_limit_orders(self) -> None:
        """Check all pending limit orders to see if they should be executed."""

        executed_orders = []

        for order in self.pending_orders:
            # TODO: problem is that we might overwrite a long position or a short position in case its triggered in both directions. Can mitigate this problem byhaving more finegrained data but cannot solve it.
            symbol = order.get("symbol")
            side = order.get("side")
            price = order.get("price")

            # Get the current high/low price
            current_data = self._fetch_ohlcv(symbol)
            if not current_data:
                continue

            current_low = current_data["low"]
            current_high = current_data["high"]

            #  # a new position is opened. Check if limit order should be executed, TODO: only a single position can be added. How does this work for the real exchange?
            if (side == "buy" and current_low <= price) or (side == "sell" and current_high >= price):
                # only add the position, in case no position is open. This is not completly realistic Otherwise do a self.position long and short or a list of dict positions for each symbol
                if symbol not in self.positions:

                    self._create_position(symbol, order)
                    logger.info(
                        "Executed limit order for symbol: %s, side: %s, amount: %s at price: %s",
                        symbol,
                        side,
                        order["total"],
                        price,
                    )

                    executed_orders.append(order)

        # Remove executed orders from pending orders
        for order in executed_orders:
            self.pending_orders.remove(order)

    def check_stop_losses(self) -> None:
        """Check all open positions to see if the stop loss condition is met.

        Returns:
            None
        """

        for symbol, position in list(self.positions.items()):
            stop_loss_price = position.get("stopLoss", {}).get("triggerPrice", None)

            if stop_loss_price is None:
                continue

            current_data = self._fetch_ohlcv(symbol)
            if not current_data:
                continue

            current_low = current_data["low"]
            current_high = current_data["high"]
            hold_side = position["holdSide"]

            if hold_side == "long" and current_low < stop_loss_price:
                logger.info("Stop loss triggered for long position: %s", symbol)
                self.close_position(symbol, side="long", stop_loss_triggered=True)
            elif hold_side == "short" and current_high > stop_loss_price:
                logger.info("Stop loss triggered for short position: %s", symbol)
                self.close_position(symbol, side="short", stop_loss_triggered=True)

    def check_take_profit(self) -> None:
        """Check all open positions to see if the stop loss condition is met.

        Returns:
            None
        """

        for symbol, position in list(self.positions.items()):
            take_profit_price = position.get("takeProfit", {}).get("triggerPrice", None)

            if take_profit_price is None:
                continue

            current_data = self._fetch_ohlcv(symbol)
            current_low = current_data["low"]
            current_high = current_data["high"]
            current_close = current_data["close"]
            hold_side = position["holdSide"]
            open_date = position["open_date"]

            # restricted take profit simulation if the open date is the current date because based on ohlc candles its not possible to tell if take profit was hit
            if open_date == self.current_date:
                if hold_side == "long" and current_close > take_profit_price:
                    logger.info("Take profit triggered for long position: %s", symbol)
                    self.close_position(symbol, side="long", take_profit_triggered=True)
                elif hold_side == "short" and current_close < take_profit_price:
                    logger.info("Take profit triggered for short position: %s", symbol)
                    self.close_position(symbol, side="short", take_profit_triggered=True)
            else:
                if hold_side == "long" and current_high > take_profit_price:
                    logger.info("Take profit triggered for long position: %s", symbol)
                    self.close_position(symbol, side="long", take_profit_triggered=True)
                elif hold_side == "short" and current_low < take_profit_price:
                    logger.info("Take profit triggered for short position: %s", symbol)
                    self.close_position(symbol, side="short", take_profit_triggered=True)

    def _generate_order_id(self) -> str:
        """Generate a unique order ID using UUID."""
        return str(uuid.uuid4())

    def _remove_position(self, symbol: str) -> None:
        """helper function to remove a position"""

        del self.positions[symbol]

    def _fetch_ohlcv(self, symbol: str) -> Dict[str, float]:
        """Fetch the ticker data for a given symbol at the current date.

        Args:
            symbol (str): The trading symbol to fetch the ticker for.

        Returns:
            Dict[str, float]: A dictionary containing the ticker data.
        """

        df = self.current_data.get(symbol)
        if df is not None and not df.empty:
            low_price = df.iloc[-1]["low"].item()
            high_price = df.iloc[-1]["high"].item()
            close_price = df.iloc[-1]["close"].item()
            vol = df.iloc[-1]["volume"].item()
            timestamp = df.index[-1]

            return {
                "symbol": symbol,
                "time": timestamp,
                "close": close_price,
                "low": low_price,
                "high": high_price,
                "volume": vol,
            }
        else:
            logger.warning("No data for symbol: %s on date: %s", symbol, self.current_date)
            return {}

    def _create_position(self, symbol: str, order: dict) -> None:
        """Helper function to open a position.

        Args:
        symbol (str): symbol such as "BTC/USDT:USDT"
        order (dict); the order for which a position should be created.
        """

        order_type = order.get("order_type")
        if order_type == "market":
            entry_price = self.fetch_ticker(symbol)["last"]
            amount_usd = entry_price * order["total"]
        elif order_type == "limit":
            entry_price = order.get("price")
            # If it's a limit order, reduce the reserved balance when the order is filled
            reserved_amount = entry_price * order["total"]
            self.locked_balance -= reserved_amount
            amount_usd = 0  # thats why we set the amount_usd to 0 because the capital got locked already

        # calculate fee
        fee = (
            self.order_fees.get(symbol).get("maker")
            if order_type == "limit"
            else self.order_fees.get(symbol).get("taker")
        )
        fee_amount_usd_opening = amount_usd * fee

        position = {
            "entry_price": entry_price,
            "open_date": self.current_date,
            "opening_fee": fee_amount_usd_opening,
        }

        position.update(order)

        # update balance
        self.balance_total -= fee_amount_usd_opening
        self.balance_free = self.balance_free - fee_amount_usd_opening - amount_usd
        self.balance_total = round(self.balance_total, 3)
        self.balance_free = round(self.balance_free, 3)

        # assert self.balance_free >= 0

        # add to position
        self.positions[symbol] = position

    ####### the following functions are implemented in ccxt #######

    def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        """Fetch the ticker data for a given symbol at the current date.

        Args:
            symbol (str): The trading symbol to fetch the ticker for.

        Returns:
            Dict[str, float]: A dictionary containing the ticker data.
        """

        df = self.current_data.get(symbol)
        if df is not None and not df.empty:
            current_price = df.iloc[-1]["close"].item()
            timestamp = df.index[-1]

        else:
            timestamp = None
            current_price = None

        return {"symbol": symbol, "time": timestamp, "last": current_price}

        # raise ValueError(f"No data for symbol: {symbol} on date: {self.current_date}")

    def fetch_positions(self) -> List[Dict[str, Union[str, float]]]:
        """Fetch all open positions.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of dictionaries containing open position details.
        """

        return [{"symbol": symbol, "info": info} for symbol, info in self.positions.items()]

    def fetch_position(self, symbol: str) -> Dict[str, Union[str, float]]:
        """Fetch the position for a given symbol.

        Args:
            symbol (str): The trading symbol to fetch the position for.

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the position details.
        """

        return {"symbol": symbol, "info": self.positions.get(symbol, {})}

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Union[str, float]]:
        """Create a new order."""

        order_id = self._generate_order_id()  # Generate a unique order ID

        # for key in ["stopLoss", "takeProfit"]:
        #    if key not in params:
        #        raise ValueError(f"Provide {key} in the params when creating an order.")

        order = {
            "id": order_id,
            "order_type": type,
            "symbol": symbol,
            "type": type,
            "side": side,
            "holdSide": "long" if side == "buy" else "short",
            "total": amount,
            "price": price,
        }

        order.update(params)

        # Add limit order to pending orders
        if type == "limit":
            if price is None:
                raise ValueError("Limit price must be specified for limit orders.")

            # Calculate the reserved amount for the limit order
            reserved_amount = price * amount

            # Update the balances to reflect the reserved amount
            self.balance_free -= reserved_amount
            self.locked_balance += reserved_amount

            self.pending_orders.append(order)

        # Handle market orders. Here we create the position straight away
        else:
            self._create_position(symbol, order)

            logger.info(
                "Created market order for symbol: %s, side: %s, amount: %s at price: %s", symbol, side, amount, price
            )

        return order

    def close_position(
        self,
        symbol: str,
        side: str,
        stop_loss_triggered: Optional[bool] = False,
        take_profit_triggered: Optional[bool] = False,
    ) -> None:
        """Close an existing position and calculate profit.

        Args:
            symbol (str): The trading symbol.
            side (str): The side of the position to close ("long" or "short").
            stop_loss_triggered (Optional[bool]): if a stop loss is triggered, the exit price is the stop loss price
            take_profit_triggered (Optional[bool]): if a take profit is triggered, the exit price is the take profit price

        Returns:
            None
        """

        if symbol in self.positions:
            position = self.positions[symbol]
            entry_price = position["entry_price"]

            amount = position["total"]
            hold_side = position["holdSide"]
            stop_loss_price = position.get("stopLoss", {}).get("triggerPrice", None)
            take_profit_price = position.get("takeProfit", {}).get("triggerPrice", None)
            order_type = position["order_type"]
            fee_amount_usd_opening = position["opening_fee"]

            # determine exit price. if stop_loss or take_profit is triggered, its always a limit order and therefore maker fees
            if stop_loss_triggered:
                exit_price = stop_loss_price
                fee = self.order_fees.get(symbol).get("taker")
            elif take_profit_triggered:
                exit_price = take_profit_price
                fee = self.order_fees.get(symbol).get("taker")
            else:
                exit_price = self.fetch_ticker(symbol)["last"]
                fee = (
                    self.order_fees.get(symbol).get("maker")
                    if order_type == "limit"
                    else self.order_fees.get(symbol).get("taker")
                )

            if exit_price is None:
                logger.warning("No price information available for %s", symbol)
                return None

            # Calculate profit
            if hold_side == "long":
                profit = (exit_price - entry_price) * amount
                profit_percent = (exit_price - entry_price) / entry_price
            else:
                profit = (entry_price - exit_price) * amount
                profit_percent = (entry_price - exit_price) / entry_price

            # TODO: simplification always assuming the same order type for an open and close
            amount_usd = exit_price * amount  # this is wrong
            fee_amount_usd_closing = amount_usd * fee
            fee_amount_usd = fee_amount_usd_closing + fee_amount_usd_opening
            net_profit = profit - fee_amount_usd
            net_profit_percent = profit_percent - fee

            # Update balance
            self.balance_total = self.balance_total - fee_amount_usd_closing + profit
            self.balance_free = self.balance_free - fee_amount_usd_closing + (entry_price * amount) + profit
            self.balance_total = self.balance_total
            self.balance_free = self.balance_free

            # assert self.balance_free <= self.balance_total

            logger.info("-------------------------------------")
            logger.info("Profit %s", profit)
            logger.info("Updated balance: %s", self.fetch_balance())
            logger.info("-------------------------------------")

            # Log the trade
            trade = {
                "exit_price": exit_price,
                "profit": profit,
                "profit_percent": profit_percent,
                "net_profit": net_profit,
                "net_profit_percent": net_profit_percent,
                "closing_fee": fee_amount_usd_closing,
                "total_fee": fee_amount_usd,
                "total_balance": round(self.balance_total, 2),
                "free_balance": round(self.balance_free, 2),
                "close_date": self.current_date,
            }

            # add the position information to the trade and add it to the history
            trade.update(position)
            self.trade_history.append(trade)

            # Remove position
            self._remove_position(symbol)

        else:
            logger.warning("No position to close for symbol: %s, side: %s", symbol, side)

    def fetch_balance(self) -> dict:
        """Fetch the current balance.

        Returns:
            dict: The current balance.
        """

        return {
            "USDT": {
                "free": self.balance_free,
                "total": self.balance_total,
                "used": self.balance_total - self.balance_free,
            }
        }

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Union[str, float]]]:
        """Fetch all open orders, optionally filtered by symbol.

        Args:
            symbol (Optional[str]): The trading symbol to filter orders by.

        Returns:
            List[Dict[str, Union[str, float]]]: A list of dictionaries containing open order details.
        """

        if symbol:
            return [order for order in self.pending_orders if order["symbol"] == symbol]
        return self.pending_orders

    def cancel_orders(self, order_ids: List[str], symbol: str) -> None:
        """Cancel multiple open orders by their IDs and symbol.

        Args:
            order_ids (List[str]): List of IDs of the orders to cancel.
            symbol (str): The trading symbol for the orders.
        """

        orders_to_remove = [
            order for order in self.pending_orders if (order["id"] in order_ids and order["symbol"] == symbol)
        ]

        for order in orders_to_remove:
            # Calculate the reserved amount to be released
            reserved_amount = order["price"] * order["total"]

            # Adjust the balances
            self.balance_free += reserved_amount
            self.locked_balance -= reserved_amount

            # Remove the order from pending orders
            self.pending_orders.remove(order)
            logger.info("Canceled order with ID: %s", order["id"])

        logger.info("Canceled %d orders for symbol %s", len(orders_to_remove), symbol)

    def cancel_all_orders(self, symbol: str, params: Optional[Dict] = None) -> None:
        """Cancels all orders for a specific symbol.

        Args:
            symbol (str): symbol for which the orders should be cancelled.
        """

        existing_orders = self.fetch_open_orders(symbol=symbol)
        ids_to_delete = [order["id"] for order in existing_orders]
        # Cancel each open order for the specific symbol
        self.cancel_orders(ids_to_delete, symbol=symbol)


def load_mock_exchange(save_directory: str) -> MockExchange:
    """Create a mock exchange object based on a stored configuration json
    Args:
        file_path (str): File directory of the configuration.
    """

    file_path = os.path.join(save_directory, CONFIG_FILE)

    if not os.path.exists(file_path):
        raise ValueError(f"Specified path does not exist: {file_path}")

    with open(file_path, "r", encoding="utf8") as f:
        config_dict = json.load(f, object_hook=timestamp_decoder)

    module_path_res, cls_name_res = config_dict["resource_path"].rsplit(".", 1)
    module_res = import_module(module_path_res)
    exchange_cls: MockExchange = getattr(module_res, cls_name_res)

    module_path_settings, cls_name_settings = config_dict.get("settings_path").rsplit(".", 1)
    module_settings = import_module(module_path_settings)
    settings_cls = getattr(module_settings, cls_name_settings)

    settings: MockExchangeSettings = TypeAdapter(settings_cls).validate_python(config_dict["settings"])

    return exchange_cls(config=settings)
