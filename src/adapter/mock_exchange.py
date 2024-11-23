"""Mock exchange for simluating an exchange."""

import json
import os
import uuid
from importlib import import_module
from typing import Dict, List, Literal, Optional, Union

import ccxt
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from src.adapter.datasources import ExchangeDatasource, ExchangeDatasourceSettings
from src.common_packages import CustomJSONEncoder, create_logger, timestamp_decoder
from src.core.base import BaseConfiguration, ObjectId
from src.core.datasource import Data
from src.core.engine import Order

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


class Transaction(BaseModel):
    """Transaction on the exchange"""

    time: pd.Timestamp
    symbol: str
    type: Literal["Funding fee", "Close Long", "Close Short", "Open Long", "Open Short"]
    fee_usd: float
    price: float
    quantity: float
    profit_usd: float
    profit_pct: float
    balance: float
    order: Optional[Order] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Ohlcv(BaseModel):
    """ohlcv data for a coin"""

    symbol: str
    time: pd.Timestamp
    close: float
    low: float
    high: float
    volume: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExchangeOrder(Order):
    """exchange order having a specific id."""

    id: str
    time: pd.Timestamp
    fee_usd: Optional[float] = None  # an executed order has a fee associated
    amount_usd: Optional[float] = None
    linked_id: Optional[str] = None  # links a SL, TP market order to its actual order
    reduce_only: bool = False
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Position(BaseModel):
    """Position for a symbol"""

    holdSide: Literal["long", "short"]
    symbol: str
    orders: List[ExchangeOrder]

    def get_average_price(self) -> float:
        """Calculate the average price of the position based on orders."""

        total = self.get_total()
        if total == 0:
            return 0.0  # Avoid division by zero
        sum_amount_usd = sum(order.amount_usd or 0.0 for order in self.orders)
        return sum_amount_usd / total

    def get_total(self) -> float:
        """Calculate the total position size based on orders (in native currency)"""

        return sum(order.amount_usd or 0.0 for order in self.orders)


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

        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[ExchangeOrder] = []
        self.transaction_history: List[Transaction] = []

        # self.trade_history = []

        # Initialize balance
        self.balance_total = 10000
        self.balance_free = self.balance_total
        self.balance_locked = 0.0  # Balance reserved for limit orders

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

        serialized_history = [transaction.model_dump() for transaction in self.transaction_history]

        with open(path, "w", encoding="utf8") as file:
            json.dump(serialized_history, file, indent=4, default=str, cls=CustomJSONEncoder)
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

            # ignore reduce only orders
            if order.reduce_only:
                continue

            ohlcv = self._fetch_ohlcv(order.symbol)

            if not ohlcv:
                continue

            #  # a new position is opened. Check if limit order should be executed, TODO: only a single position can be added. How does this work for the real exchange?
            if (order.side == "buy" and ohlcv.low <= order.price) or (
                order.side == "sell" and ohlcv.high >= order.price
            ):
                self._remove_pending_order(order)
                self._add_position(order)
                logger.info("Executed limit order: %s", order)

                executed_orders.append(order)

    def check_stop_losses(self) -> None:
        """Check all open positions to see if the stop loss condition is met.

        Returns:
            None
        """
        for order in self.pending_orders:
            if order.reduce_only is False:
                continue
            if order.type != "SL-market":
                continue
            if order.symbol not in self.positions:
                self._remove_pending_order(order)
                continue
            open_order_ids = [o.id for o in self.positions[order.symbol].orders]
            if order.linked_id not in open_order_ids:
                self._remove_pending_order(order)
                continue

            stop_loss_price = order.price

            ohlcv = self._fetch_ohlcv(order.symbol)
            if not ohlcv:
                logger.info("No ohlcv data for this entry.")
                continue

            if self.positions[order.symbol].holdSide == "long" and ohlcv.low < stop_loss_price:
                logger.info("Stop loss triggered for long position: %s", order.symbol)
                self.close_position(order.symbol, side="long", order_ids=[order.linked_id])
            elif self.positions[order.symbol].holdSide == "short" and ohlcv.high > stop_loss_price:
                logger.info("Stop loss triggered for short position: %s", order.symbol)
                self.close_position(order.symbol, side="short", order_ids=[order.linked_id])

            self._remove_pending_order(order)

    def check_take_profit(self) -> None:
        """Check all open positions to see if the stop loss condition is met.

        Returns:
            None
        """

        for order in self.pending_orders:
            if order.reduce_only is False:
                continue
            if order.type != "TP-market":
                continue
            if order.symbol not in self.positions:
                self._remove_pending_order(order)
                continue
            open_order_ids = [o.id for o in self.positions[order.symbol].orders]
            if order.linked_id not in open_order_ids:
                self._remove_pending_order(order)
                continue

            take_profit_price = order.price

            ohlcv = self._fetch_ohlcv(order.symbol)

            if not ohlcv:
                logger.info("No ohlcv data for this entry.")
                continue

            position_side = self.positions[order.symbol].holdSide

            # restricted take profit simulation if the open date is the current date because based on ohlc candles its not possible to tell if take profit was hit
            if order.time == self.current_date:
                if position_side == "long" and ohlcv.close > take_profit_price:
                    logger.info("Take profit triggered for long position: %s", order.symbol)
                    self.close_position(order.symbol, side="long", order_ids=[order.linked_id])
                elif position_side == "short" and ohlcv.close < take_profit_price:
                    logger.info("Take profit triggered for short position: %s", order.symbol)
                    self.close_position(order.symbol, side="short", order_ids=[order.linked_id])
            else:
                if position_side == "long" and ohlcv.high > take_profit_price:
                    logger.info("Take profit triggered for long position: %s", order.symbol)
                    self.close_position(order.symbol, side="long", order_ids=[order.linked_id])
                elif position_side == "short" and ohlcv.low < take_profit_price:
                    logger.info("Take profit triggered for short position: %s", order.symbol)
                    self.close_position(order.symbol, side="short", order_ids=[order.linked_id])

            self._remove_pending_order(order)

    def _generate_order_id(self) -> str:
        """Generate a unique order ID using UUID."""
        return str(uuid.uuid4())

    def _remove_pending_order(self, order: ExchangeOrder) -> None:
        """Helper function to remove pending orders"""

        if order.reduce_only is False:
            self.balance_free += order.amount_usd
            self.balance_locked -= order.amount_usd

        self.pending_orders.remove(order)

        logger.debug("Canceled order with ID: %s.", order.id)
        logger.debug("New free balance: %s/%s", self.balance_free, self.balance_total)

    def _add_pending_orders(self, order: ExchangeOrder) -> bool:
        """Add order to pending orders.

        Usually for limit orders.
        """

        if order.reduce_only is False:
            if order.amount_usd > self.balance_free:
                logger.info("Position cannot be opened. Not enough capital.")
                return False

            # Calculate the locked capital because of the pending order
            self.balance_free -= order.amount_usd
            self.balance_locked += order.amount_usd

        self.pending_orders.append(order)

        logger.debug("Added order with ID: %s.", order.id)
        logger.debug("New free balance: %s/%s", self.balance_free, self.balance_total)

        return True

    def _fetch_ohlcv(self, symbol: str) -> Union[Ohlcv, None]:
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

            return Ohlcv(symbol=symbol, time=timestamp, close=close_price, low=low_price, high=high_price, volume=vol)

        logger.warning("No data for symbol: %s on date: %s", symbol, self.current_date)
        return None

    def get_fee_pct(self, symbol: str, order_type: Literal["market, limit"]) -> float:
        """Get fees for the symbol."""

        if order_type == "limit":
            return self.order_fees.get(symbol).get("maker")

        return self.order_fees.get(symbol).get("taker")

    def _add_position(self, order: ExchangeOrder) -> bool:
        """Helper function to add a position.

        Args:
        order (Order); the order for which a position should be created.
        """

        # get order price and amount in usd
        if order.type == "market":
            order.price = self.fetch_ticker(order.symbol)["last"]
            order.amount_usd = order.price * order.amount

        if order.amount_usd > self.balance_free:
            logger.info("Position cannot be opened. Not enough capital.")
            return False

        # calculate fee
        order.fee_usd = -order.amount_usd * self.get_fee_pct(order.symbol, order.type)

        # add to existing position if one exists
        if order.symbol in self.positions:
            position = self.positions[order.symbol]

            if (position.holdSide == "long" and order.side == "sell") or (
                position.holdSide == "short" and order.side == "buy"
            ):
                logger.info("Currently its not supported to reduce a position.")
                return False

            # add order to position
            position.orders.append(order)

        # otherwise create a new position
        else:
            self.positions[order.symbol] = Position(
                symbol=order.symbol, holdSide="long" if order.side == "buy" else "short", orders=[order]
            )

        # update balance
        self.balance_total += order.fee_usd  # round(order.fee, 3)
        self.balance_free -= order.amount_usd - order.fee_usd  # round(order.amount_usd - order.fee, 3)

        # add a transaction
        if order.side == "buy":
            transaction_type = "Open Long"
        else:
            transaction_type = "Open Short"

        self.transaction_history.append(
            Transaction(
                time=self.current_date,
                symbol=order.symbol,
                type=transaction_type,
                fee_usd=order.fee_usd,
                price=order.price,
                quantity=order.amount,
                profit_usd=0,
                profit_pct=0,
                balance=self.balance_total,
                order=order,
            )
        )

        return True

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

        return [{"symbol": symbol, "info": info.model_dump()} for symbol, info in self.positions.items()]

    def fetch_position(self, symbol: str) -> Dict[str, Union[str, float]]:
        """Fetch the position for a given symbol.

        Args:
            symbol (str): The trading symbol to fetch the position for.

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the position details.
        """

        if symbol in self.positions:
            return {"symbol": symbol, "info": self.positions.get(symbol).model_dump()}

        return {"symbol": symbol, "info": {}}

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Union[str, float]]:
        """Create a new order.

        Note: it is currently not supported to close an open position via create order.
        For doing that close_position() should be used.
        """

        order_id = self._generate_order_id()

        order = ExchangeOrder(
            symbol=symbol,
            time=self.current_date,
            type=type,
            side=side,
            amount=amount,
            price=price,
            params=params,
            id=order_id,
        )

        if price and order.type == "limit":
            order.amount_usd = order.price * order.amount

        if order.type == "limit":
            result = self._add_pending_orders(order)
        else:  # market oder open position straight away
            result = self._add_position(order)

        # Helper function for adding SL/TP orders
        def add_linked_order(order_type: str, trigger_key: str):
            if trigger_key in params and "triggerPrice" in params[trigger_key]:
                linked_order = ExchangeOrder(
                    symbol=symbol,
                    time=self.current_date,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=params[trigger_key]["triggerPrice"],
                    linked_id=order_id,
                    id=self._generate_order_id(),
                    reduce_only=True,
                )
                self._add_pending_orders(linked_order)

        # Add Stop Loss (SL) and Take Profit (TP) orders
        add_linked_order("SL-market", "stopLoss")
        add_linked_order("TP-market", "takeProfit")

        # Return the created order's details
        return order.model_dump() if result else None

    def close_position(
        self,
        symbol: str,
        side: Literal["long", "short"],
        order_ids: Optional[List[str]] = None,
    ) -> None:
        """Close an existing position and calculate profit.

        Args:
            symbol (str): The trading symbol.
            side (str): The side of the position to close ("long" or "short").

        Returns:
            None
        """

        if symbol not in self.positions:
            logger.warning("No position to close for symbol: %s, side: %s", symbol, side)
            return

        exit_price = self.fetch_ticker(symbol)["last"]

        if exit_price is None:
            logger.warning("No price information available for %s", symbol)
            return None

        position = self.positions[symbol]
        profit = 0

        # close position by iterating over all individual executed orders
        for exec_order in position.orders:

            # if order_ids is defined skip orders which don't match the defined ids
            if order_ids and exec_order not in order_ids:
                continue

            if position.holdSide == "long":
                profit = (exit_price - exec_order.price) * exec_order.amount
                profit_pct = (exit_price - exec_order.price) / exec_order.price
                transaction_type = "Close Long"
            else:  # short
                profit = (exec_order.price - exit_price) * exec_order.amount
                profit_pct = (exec_order.price - exit_price) / exec_order.price
                transaction_type = "Close Short"

            fee_usd = -(exit_price * exec_order.amount) * self.get_fee_pct(symbol, order_type="market")

            position_size_usd = exec_order.amount_usd

            position.orders.remove(exec_order)

            self.balance_total += profit + fee_usd
            self.balance_free += position_size_usd + profit + fee_usd
            self.balance_locked -= position_size_usd

            # add transaction
            self.transaction_history.append(
                Transaction(
                    time=self.current_date,
                    symbol=symbol,
                    type=transaction_type,
                    fee_usd=fee_usd,
                    price=exit_price,
                    quantity=exec_order.amount,
                    profit_usd=profit,
                    profit_pct=profit_pct,
                    balance=self.balance_total,
                )
            )

        # del position symbol if no open positions
        if len(position.orders) == 0:
            del self.positions[symbol]

        logger.info("-------------------------------------")
        logger.info("Profit %s", profit)
        logger.info("Updated balance: %s", self.fetch_balance())
        logger.info("-------------------------------------")

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
            return [order.model_dump() for order in self.pending_orders if order.symbol == symbol]
        return [order.model_dump() for order in self.pending_orders]

    def cancel_orders(self, order_ids: List[str], symbol: str) -> None:
        """Cancel multiple open orders by their IDs and symbol.

        Args:
            order_ids (List[str]): List of IDs of the orders to cancel.
            symbol (str): The trading symbol for the orders.
        """

        orders_to_remove = [
            order for order in self.pending_orders if (order.id in order_ids and order.symbol == symbol)
        ]

        for order in orders_to_remove:
            self._remove_pending_order(order)

        logger.info("Canceled %d orders for symbol %s.", len(orders_to_remove), symbol)

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
