import math
import os
import time
from typing import List, Literal, Optional

from src.common_packages import create_logger
from src.core.engine import Order, TradeSignal, TradingEngine

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


class CCXTFuturesTradingEngine(TradingEngine):
    """Class to interact with the exchange API."""

    def set_trading_settings(self) -> None:
        """Set the settings for the trading.

        This includes leverage, fetching minimal position size, price precision information and so on.
        """

        logger.info("Set the trading settings for the selected symbols: %s.", self.config.symbols)

        self.config.exchange.load_markets()

        for symbol in self.config.symbols:
            try:
                # fetch minimum order quantity
                market_info = self.config.exchange.market(symbol)
                self.order_details[symbol] = {
                    "min_amount": market_info["limits"]["amount"]["min"],
                    "precision": count_decimal_places(market_info["precision"]["amount"]),  # number of decimal places
                    "maker": market_info["maker"],
                    "taker": market_info["taker"],
                }

                logger.info("Fetched order details for symbol %s. %s", symbol, self.order_details[symbol])

            except Exception as e:
                logger.error("Failed to fetch order quantities for symbol %s: %s", symbol, str(e))
                continue

        for symbol in self.config.symbols:
            try:
                self.config.exchange.setPositionMode(True, symbol)  # set position mode to hedge
                self.config.exchange.setMarginMode("isolated", symbol)  # sets the margin mode to isolated
                self.config.exchange.setPositionMode(True, symbol)  # sets the position mode to hedge_mode
                self.config.exchange.setLeverage(1, symbol, params={"holdSide": "short"})  # set leverage for short
                self.config.exchange.setLeverage(1, symbol, params={"holdSide": "long"})  # set leverage for long

                logger.info("Set trading settings for symbol %s.", symbol)

            except Exception as e:
                logger.error(
                    "Failed to set settings for symbol %s: %s. This behaviour is expected when using a mock exchange.",
                    symbol,
                    str(e),
                )
                continue

        logger.info("Completed setting trading settings for all provided symbols.")

    def log_open_positions(self) -> List[str]:
        """Function to log all open positions."""

        positions = self.config.exchange.fetch_positions()
        log_messages = []
        symbols = []
        if not positions:
            log_messages.append("No open position")
        else:
            log_messages.append("Log open positions:")
            for position in positions:
                symbol = position["info"].get("symbol", "N/A")
                hold_side = position["info"].get("holdSide", "N/A")
                position = position["info"].get("total", "N/A")
                message = f"Symbol: {symbol}, holdSide: {hold_side}, PositionSize: {position}"
                log_messages.append(message)
                symbols.append(symbol)

        balance = self.config.exchange.fetch_balance()

        # Join all messages with a newline and log them as a single entry
        logger.info(" ".join(log_messages))
        logger.info("Balance: %s", balance)

        return symbols

    def determine_batch_order_sizes(
        self,
        trade_signals: List[TradeSignal],
        method: Literal["balance", "risk_based"] = "risk_based",
        risk_pct: float = 0.01,
    ) -> List[TradeSignal]:
        """
        Determine the position sizes for all trade signals collectively, considering open positions and total balance.

        Args:
            trade_signals (List[TradeSignal]): A list of trade signals for multiple symbols.
            method (str): The method to use for determining the order size. Options: 'balance', 'risk_based'.
            risk_pct (float): The percentage of the balance to risk.

        Returns:
            List[TradeSignal]: Updated trade signals with determined position sizes.
        """

        def fetch_cached_ticker(symbol: str, ticker_cache: dict) -> float:
            """Fetches the ticker price, using the cache if available."""
            if symbol not in ticker_cache:
                ticker_cache[symbol] = self.config.exchange.fetch_ticker(symbol)["last"]
            return ticker_cache[symbol]

        def calculate_stop_loss_pct(trade_signal: TradeSignal) -> Optional[float]:
            """Calculates the stop loss percentage for a given trade signal."""

            if trade_signal.stop_loss_price:
                entry_price = signal.limit_price or ticker_cache[signal.symbol]
                return abs((entry_price - trade_signal.stop_loss_price) / entry_price)
            return None

        def calculate_usd_order_size(
            balance: dict, method: str, stop_loss_pct: Optional[float], risk_pct: float
        ) -> float:
            """Calculates the USD order size based on the chosen method and stop loss percentage."""
            if method == "balance":
                return balance["total"] * risk_pct
            if method == "risk_based" and stop_loss_pct is not None:
                return (balance["total"] * risk_pct) / stop_loss_pct
            return 0.0

        def calculate_scaling_factor(total_allocated_usd: float, free_balance: float) -> float:
            """Calculates a scaling factor to ensure total allocation doesn't exceed available balance."""
            return min(1.0, free_balance / total_allocated_usd) if total_allocated_usd > 0 else 1.0

        # Fetch balance and initialize variables
        balance = self.config.exchange.fetch_balance()["USDT"]
        logger.info("Available balance: %s", balance["free"])
        free_balance = balance["free"]

        ticker_cache = {}
        total_allocated_usd = 0.0
        usd_order_sizes = {}

        # Calculate USD order sizes and total allocation
        for signal in trade_signals:

            # check whether there is currently an entry for the coin
            current_price = fetch_cached_ticker(signal.symbol, ticker_cache)
            if current_price is None:
                usd_order_sizes[id(signal)] = 0.0
            else:
                stop_loss_pct = calculate_stop_loss_pct(signal)
                usd_order_size = calculate_usd_order_size(balance, method, stop_loss_pct, risk_pct)
                usd_order_sizes[id(signal)] = usd_order_size
                total_allocated_usd += usd_order_size

        # Apply scaling factor
        scale_factor = calculate_scaling_factor(total_allocated_usd, free_balance)

        # Assign scaled USD order sizes to trade signals
        for signal in trade_signals:
            usd_order_size = usd_order_sizes[id(signal)] * scale_factor

            # Calculate position size in base currency (e.g., BTC)
            entry_price = signal.limit_price or ticker_cache[signal.symbol]
            amount = usd_order_size / entry_price

            # Validate and adjust amount
            min_amount = self.order_details.get(signal.symbol, {}).get("min_amount", 0)
            if amount < min_amount:
                logger.warning(
                    "Specified amount (%s) is smaller than the minimum order quantity (%s) for symbol (%s). Skipping.",
                    amount,
                    min_amount,
                    signal.symbol,
                )
                signal.order_amount = None
                continue

            precision = self.order_details.get(signal.symbol, {}).get("precision", 8)
            amount = round_down(amount, precision)

            # Set the determined order size in the trade signal
            signal.order_amount = amount
            logger.info("Position size determined for symbol %s: %s (USD: %s)", signal.symbol, amount, usd_order_size)

        return trade_signals

    def execute_orders(self, trade_signals: List[TradeSignal]):
        """
        Places trades based on predictions from the model, after determining the batch order sizes.

        Args:
            trade_signals (List[TradeSignal]): A list of TradeSignal objects, each containing trade details for a symbol.
        """

        logger.info("+" * 50)
        logger.info("Trade signals: %s", trade_signals)

        # Log current open positions
        symbols = self.log_open_positions()

        # Close all open orders for the symbols in trade signals
        for trade_signal in trade_signals:
            open_orders = self.config.exchange.fetch_open_orders(symbol=trade_signal.symbol)
            if open_orders:
                self.config.exchange.cancel_all_orders(symbol=trade_signal.symbol)

            # Remove symbol from the list of open positions if it exists
            if trade_signal.symbol in symbols:
                symbols.remove(trade_signal.symbol)

        # Close positions for remaining symbols where no prediction exists
        for symbol in symbols:
            hold_side = self._determine_position(symbol=symbol)
            self._close_trade(symbol=symbol, hold_side=hold_side)

        # Determine order sizes for all trade signals in batch
        trade_signals = self.determine_batch_order_sizes(trade_signals, method="balance", risk_pct=0.2)

        # Open new trades or reverse existing ones based on predictions
        for trade_signal in trade_signals:
            if trade_signal.order_amount is None:
                continue

            mapping = {"long": "buy", "short": "sell"}
            hold_side = self._determine_position(symbol=trade_signal.symbol)
            if trade_signal.position_side == mapping.get(hold_side):
                logger.info("Keeping current position for symbol: %s with side: %s", trade_signal.symbol, hold_side)
            else:
                if hold_side:
                    logger.info("Reversing position for symbol: %s", trade_signal.symbol)
                    self._close_trade(symbol=trade_signal.symbol, hold_side=hold_side)

                # Place a new order
                self._create_order(trade_signal)

        # Log updated open positions
        self.log_open_positions()

    def execute_orders_simple(self, trade_signals: List[TradeSignal]):
        """Places trades based on predictions from the model, after determining the batch order sizes.

        This function always closes the full position at the open price
        Args:
            trade_signals (List[TradeSignal]): A list of TradeSignal objects, each containing trade details for a symbol.
        """

        # logger.info("+" * 50)
        # logger.info("Trade signals: %s", trade_signals)

        # Log current open positions
        self.log_open_positions()

        # close all open positions
        positions = self.config.exchange.fetch_positions()
        for position in positions:
            symbol = position["info"].get("symbol")
            hold_side = position["info"].get("holdSide")
            self._close_trade(symbol=symbol, hold_side=hold_side)

        # Close all open orders for the symbols in trade signals. Alternatively we could disable all orders
        for trade_signal in trade_signals:
            open_orders = self.config.exchange.fetch_open_orders(symbol=trade_signal.symbol)
            if open_orders:
                self.config.exchange.cancel_all_orders(symbol=trade_signal.symbol)

        # Determine order sizes for all trade signals in batch
        trade_signals = self.determine_batch_order_sizes(trade_signals, method="balance", risk_pct=0.1)

        # Open new orders
        for trade_signal in trade_signals:
            self._create_order(trade_signal)

        # Log updated open positions
        self.log_open_positions()

    def _close_trade(self, symbol: str, hold_side: Literal["long", "short"]) -> None:
        """Closes a trade.

        Args:
            symbol (str): The trading symbol for which the position should be closed.
            hold_side (str): The side of the position to close ('buy', 'sell') -> this needs to be converted to "long", "short"

        Returns:
            None
        """

        self.config.exchange.close_position(symbol=symbol, side=hold_side)
        logger.info("Closed position for symbol: %s with side: %s", symbol, hold_side)

    def _create_order(self, trade_signal: TradeSignal) -> None:
        """Creates an order.

        Args:
            trade_signal (TradeSignal): the trade signal
        """

        if trade_signal.order_amount is None:
            logger.info("Amount is None for %s. No order gets placed.", trade_signal.symbol)
            return

        params = {"hedged": True, "marginMode": "isolated"}

        if trade_signal.stop_loss_price:
            params["stopLoss"] = {"triggerPrice": trade_signal.stop_loss_price}

        if trade_signal.take_profit_price:
            params["takeProfit"] = {"triggerPrice": trade_signal.take_profit_price}

        order = Order(
            symbol=trade_signal.symbol,
            type=trade_signal.order_type,
            side=trade_signal.position_side,
            amount=trade_signal.order_amount,
            price=trade_signal.limit_price if trade_signal.order_type == "limit" else None,
            params=params,
        )

        # Place the order on the exchange
        self.config.exchange.create_order(**order.model_dump())

        logger.info("Created order: %s", order)

    def _determine_position(self, symbol: str) -> Literal["long", "short", "no_trade"]:
        """
        Determines the current trading position for a given symbol.

        Args:
            symbol (str): The trading symbol for which the position is to be determined.

        Returns:
            Literal: The current action based on the open position. Possible values are 'buy', 'sell', or 'no_trade'.
        """

        current_position = self.config.exchange.fetch_position(symbol=symbol)

        if current_position["info"] == {}:  # no position is open
            return "no_trade"
        else:  # position is open
            return current_position["info"].get("holdSide")  # either 'long' or 'short'


def count_decimal_places(value: float) -> int:
    """count the number of decimal places of a float"""

    # Convert the float to its full decimal representation
    str_value = f"{value:.15f}".rstrip("0")

    if "." in str_value:
        # Split at the decimal point
        _, decimal_part = str_value.split(".")
        return len(decimal_part)
    return 0


def round_down(number: float, decimal_places: int):
    """function to round down a number based on the specified decimal places."""

    factor = 10**decimal_places
    return math.floor(number * factor) / factor
