"""Mock exchange for simluating an exchange."""

import json
import os
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Union

import ccxt
import joblib
import pandas as pd
from pydantic import BaseModel
from pydantic.config import ConfigDict

from src.common_packages import CustomJSONEncoder, create_logger

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)

MOCK_DATA_FILE = "data.joblibe"
CONFIG_FILE = "config.json"


class MockExchangeSettings(BaseModel):
    """
    Data class for storing configuration parameters for a mock exchange.

    Args:
        data_path (Optional[str]):
            Path to the directory or file where the mock data is stored.

        simulation_start (Union[str, pd.Timestamp]):
            The start time for the simulation.

        simulation_end (Union[str, pd.Timestamp]):
            The end time for the simulation.

        timeframe (str):
            The time interval for the data scraping, specified as a string.
            Default is '4h', meaning four-hour intervals.

        config_dir (Optional[str]):
            Path to the configuration directory.

        symbols (List[str]):
            A list of trading symbols (e.g., ['BTC/USD', 'ETH/USD']) that the
            simulation will operate on.

    Example:
        # Create instance from JSON file
        config = MockExchangeConfig.from_json('path/to/config.json')
    """

    data_path: Optional[str] = None
    scrape_start: Optional[pd.Timestamp] = None
    scrape_end: Optional[pd.Timestamp] = None
    simulation_start: pd.Timestamp
    simulation_end: pd.Timestamp
    timeframe: pd.Timedelta = pd.Timedelta("4h")
    config_dir: Optional[str] = None
    symbols: List[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MockExchange(ccxt.bitget):
    """Mock exchange class for bitget."""

    def __init__(self, exchange_config: Dict, config: MockExchangeSettings):

        super().__init__(exchange_config)
        self.config = config

        # self.data = self._load_mock_data(self.config.data_path)

        self.positions = {}
        self.pending_orders = []
        self.trade_history = []

        # Initialize balance
        self.balance_total = 1000
        self.balance_free = self.balance_total
        self.locked_balance = 0.0  # Balance reserved for limit orders

        self.current_date = None
        self.current_data = None

    def _load_mock_data(self, joblib_file: str) -> Dict[str, pd.DataFrame]:
        """Load mock data from a joblib file.

        Args:
            joblib_file (str): Path to the joblib file containing mock data.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing mock data with timestamps as index.
        """

        logger.info("Loading historic mock dataset from: %s", joblib_file)

        data_dict = joblib.load(joblib_file)

        # Ensure that each entry is a DataFrame and set the index correctly
        for symbol, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"The data for {symbol} is not a pandas DataFrame.")
            if "close" not in df.columns:
                raise ValueError(f"Data for symbol {symbol} does not contain 'close' column")

            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.set_index("time", inplace=True)
            data_dict[symbol] = df

        logger.info("Exchange mock data loaded successfully.")
        return data_dict


class MockExchangeOld:
    """Mock class to simulate interaction with an exchange API via ccxt interface."""

    def __init__(self, config: MockExchangeSettings):
        """Initialize the MockExchange object with data from a specified file path.

        Args:
            data_path (str): Path to the joblib file containing the data.
        """

        self.config = config
        exchange_class = getattr(ccxt, self.config.exchange_id)
        self.exchange = exchange_class(
            {
                # "apiKey": env_parser.EX_API_KEY,
                # "secret": env_parser.SECRET_KEY,
                # "password": env_parser.PASSWORD,
                "options": {"defaultType": "swap"},
                "timeout": 30000,
                "enableRateLimit": True,
            }
        )

        # scrape taker and maker fees for each coin
        self.order_fees = {}
        self.exchange.load_markets()
        for symbol in self.config.symbols:
            market_info = self.exchange.market(symbol)
            self.order_fees[symbol] = {
                "maker": market_info["maker"],
                "taker": market_info["taker"],
            }
            logger.info("Fetched order details for symbol %s. %s", symbol, self.order_fees[symbol])

        self.positions = {}
        self.pending_orders = []

        # Initialize balance
        self.balance_total = 1000
        self.balance_free = self.balance_total
        self.locked_balance = 0.0  # Balance reserved for limit orders

        self.trade_history = []

        self.data = self._load_mock_data(self.config.data_path)

        # first and last date of the mock data
        self.config.scrape_end = max([max(df.index) for df in self.data.values()])
        self.config.scrape_start = min([min(df.index) for df in self.data.values()])
        self.config.simulation_end = min(self.config.simulation_end, self.config.scrape_end)
        self.current_date = None
        self.current_data = None

        # Ensure 'close' column is present in all DataFrames
        for symbol, df in self.data.items():
            if "close" not in df.columns:
                raise ValueError(f"Data for symbol {symbol} does not contain 'close' column")

    def save_config(self, config_dir: Optional[str] = None):
        """Function to create a configuration file.

        Args:
            config_dir (optional[str]): Optionally provide a config directory to save the results
        """

        config = self.get_config()
        metadata = {
            "init_params": config.get("init_params", {}),
        }

        self._write_config(metadata, config_dir)

    def get_config(self) -> dict:
        """Returns the configuration of the class.

        Returns:
            dict: A dictionary containing the configuration.
        """

        return {"init_params": asdict(self.config)}

    def _write_config(self, metadata: Dict, config_dir: Optional[str] = None):
        """Save the metadata to a file.

        Args:
        metadata (Dict): Configuration to store.
        config_dir (str): File directory to store the configuration.
        """

        config_dir = config_dir or self.config.config_dir
        if config_dir is None:
            raise ValueError("Please provide a configuration directory via config_dir.")

        info_file_path = os.path.join(config_dir, CONFIG_FILE)
        with open(info_file_path, "w", encoding="utf-8") as info_file:
            json.dump(metadata, info_file, indent=4, cls=CustomJSONEncoder)

        logger.info("Mock exchange information saved to %s", config_dir)

    def _set_data(self):
        """Set the data"""

        # set current data based on the current date
        self.current_data = {symbol: df[df.index == self.current_date] for symbol, df in self.data.items()}
        # if any(df.empty for df in self.current_data.values()):
        #    raise ValueError(f"No data available for the date: {self.current_date}")

    def _next_step(self):
        """set the next date and data"""

        if self.current_date is None:
            self.current_date = self.config.simulation_start
        else:
            # set current date
            self.current_date += pd.Timedelta(self.config.timeframe)

        self._set_data()

    def _load_mock_data(self, joblib_file: str) -> Dict[str, pd.DataFrame]:
        """Load mock data from a joblib file.

        Args:
            joblib_file (str): Path to the joblib file containing mock data.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing mock data with timestamps as index.

        Raises:
            ValueError: If the data for any symbol is not a pandas DataFrame.
        """
        logger.info("Loading historic mock dataset from: %s", joblib_file)

        data_dict = joblib.load(joblib_file)

        # Ensure that each entry is a DataFrame and set the index correctly
        for symbol, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"The data for {symbol} is not a pandas DataFrame.")
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.set_index("time", inplace=True)
            data_dict[symbol] = df

        logger.info("Exchange mock data loaded successfully.")
        return data_dict

    def _save_trade_history(self, path: str) -> None:
        """Save the trade history to a specified file path.

        Args:
            path (str): The file path where the trade history will be saved.
        """

        with open(path, "w", encoding="utf8") as file:
            json.dump(self.trade_history, file, indent=4, default=str)
        logger.info("Trade history saved to: %s", path)

    def _check_limit_orders(self) -> None:
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

    def _check_stop_losses(self) -> None:
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

    def _check_take_profit(self) -> None:
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

    # the following functions are implemented in ccxt

    def load_markets(self):
        """Mock function for loading markets."""

        return self.exchange.load_markets()

    def market(self, symbol: str) -> dict:
        """mock market information.

        Args:
            symbol (str): symbol to scrape market information

        Returns:
            dict: dictionary containing the information.

        """

        return self.exchange.market(symbol)

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

        Returns:
            None
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

    def cancel_all_orders(self, symbol: str, params: Optional[Dict] = None):
        """Cancels all orders for a specific symbol.

        Args:
            symbol (str): symbol for which the orders should be cancelled.

        Returns:
            None
        """

        existing_orders = self.fetch_open_orders(symbol=symbol)
        ids_to_delete = [order["id"] for order in existing_orders]
        # Cancel each open order for the specific symbol
        self.cancel_orders(ids_to_delete, symbol=symbol)
