import math
import os
from typing import List, Literal, Optional

from src.common_packages import create_logger
from src.core.engine import TradeSignal, TradingEngine

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

    def log_open_positions(self):
        """Function to log all open positions."""

        positions = self.config.exchange.fetch_positions()
        log_messages = []
        self.symbols_with_open_positions = []
        if not positions:
            log_messages.append("No open position")
        else:
            log_messages.append("Log open positions:")
            for position in positions:
                symbol = position["info"].get("symbol", "N/A")
                self.symbols_with_open_positions.append(symbol)

                hold_side = position["info"].get("holdSide", "N/A")
                position = position["info"].get("total", "N/A")
                message = f"Symbol: {symbol}, holdSide: {hold_side}, PositionSize: {position}"
                log_messages.append(message)

        balance = self.config.exchange.fetch_balance()

        # Join all messages with a newline and log them as a single entry
        logger.info(" ".join(log_messages))
        logger.info("Balance: %s", balance)

    def determine_batch_order_sizes(
        self,
        trade_signals: List[TradeSignal],
        method: Literal["balance", "risk_based"] = "risk_based",
        risk_pct: float = 0.01,
    ) -> List[TradeSignal]:
        """
        Determine the position sizes for all trade signals collectively, considering open positions and total balance.

        NOTE: the current implementation only allows a single trade_signal for each symbol.

        Args:
            trade_signals (List[TradeSignal]): A list of trade signals for multiple symbols.
            method (str): The method to use for determining the order size. Options: 'free_balance', 'total_balance', 'risk_based'.
            risk_pct (float): The percentage of the balance to risk.

        Returns:
            List[TradeSignal]: Updated trade signals with determined position sizes.
        """

        def fetch_cached_ticker(symbol: str, ticker_cache: dict) -> float:
            """Fetches the ticker price, using the cache if available."""
            if symbol not in ticker_cache:
                ticker_cache[symbol] = self.config.exchange.fetch_ticker(symbol)["last"]
            return ticker_cache[symbol]

        def calculate_stop_loss_pct(trade_signal: TradeSignal, entry_price: float) -> Optional[float]:
            """Calculates the stop loss percentage for a given trade signal."""
            if trade_signal.stop_loss_price:
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

        def calculate_scaling_factor():
            """calculates the scaling factor"""
            # Cap total allocation at available balance
            if total_allocated_usd > balance["free"]:
                scale_factor = balance["free"] / total_allocated_usd
            else:
                scale_factor = 1.0

            return scale_factor

        # Fetch balance and log open positions
        balance = self.config.exchange.fetch_balance()["USDT"]
        logger.info("Available balance: %s", balance["free"])

        # Initialize total allocated USD for all signals
        total_allocated_usd = 0.0

        # Cache for fetched ticker prices
        ticker_cache = {}

        # Cache for calculated USD order sizes
        usd_order_sizes = {}

        for trade_signal in trade_signals:
            entry_price = fetch_cached_ticker(trade_signal.symbol, ticker_cache)

            if entry_price is None:
                usd_order_sizes[trade_signal.symbol] = 0
            else:
                stop_loss_pct = calculate_stop_loss_pct(trade_signal, entry_price)
                usd_order_sizes[trade_signal.symbol] = calculate_usd_order_size(
                    balance, method, stop_loss_pct, risk_pct
                )
                total_allocated_usd += usd_order_sizes[trade_signal.symbol]

        scale_factor = calculate_scaling_factor()

        # Second loop to assign the scaled USD order sizes to each trade signal
        for trade_signal in trade_signals:

            # Adjust the USD order size based on the scale factor
            usd_order_size = usd_order_sizes[trade_signal.symbol] * scale_factor

            if usd_order_size == 0:
                trade_signal.order_amount = None
                continue


            # Calculate the position size in base currency (e.g., BTC)
            amount = usd_order_size / ticker_cache[trade_signal.symbol]

            # Ensure amount is above minimum order size
            min_amount = self.order_details.get(trade_signal.symbol).get("min_amount")
            
            if min_amount and amount < min_amount:
                logger.warning(
                    "Specified amount (%s) is smaller than the minimum order quantity (%s) for symbol (%s). Skipping.",
                    amount,
                    min_amount,
                    trade_signal.symbol,
                )
                trade_signal.order_amount = None
                continue

            # Round to the required precision
            precision = self.order_details.get(trade_signal.symbol).get("precision")
            amount = round_down(amount, precision)

            # Set the determined order size in the trade signal
            trade_signal.order_amount = amount
            logger.info(
                "Position size determined for symbol %s: %s (USD: %s)", trade_signal.symbol, amount, usd_order_size
            )

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
        self.log_open_positions()

        # Close all open orders for the symbols in trade signals, if applicable
        for trade_signal in trade_signals:
            open_orders = self.config.exchange.fetch_open_orders(symbol=trade_signal.symbol)
            if open_orders:
                self.config.exchange.cancel_all_orders(symbol=trade_signal.symbol)

            # Remove symbol from the list of open positions if it exists
            if trade_signal.symbol in self.symbols_with_open_positions:
                self.symbols_with_open_positions.remove(trade_signal.symbol)

        # Close positions for symbols where no prediction exists
        for symbol in list(self.symbols_with_open_positions):
            hold_side = self._determine_position(symbol=symbol)
            self._close_trade(symbol=symbol, side=hold_side)

        # Determine order sizes for all trade signals in batch
        trade_signals = self.determine_batch_order_sizes(trade_signals, method="risk_based", risk_pct=0.01)

        # Open new trades or reverse existing ones based on predictions
        for trade_signal in trade_signals:
            if trade_signal.order_amount is None:
                continue

            hold_side = self._determine_position(symbol=trade_signal.symbol)

            if trade_signal.position_side == hold_side:
                logger.info("Keeping current position for symbol: %s with side: %s", trade_signal.symbol, hold_side)
            else:
                if hold_side:
                    logger.info("Reversing position for symbol: %s", trade_signal.symbol)
                    self._close_trade(symbol=trade_signal.symbol, side=hold_side)

                # Place a new order
                self._create_order(trade_signal)

        # Log updated open positions
        self.log_open_positions()

    def _close_trade(self, symbol: str, side: Literal["buy", "sell"]) -> None:
        """Closes a trade.

        Args:
            symbol (str): The trading symbol for which the position should be closed.
            side (str): The side of the position to close ('buy', 'sell') -> this needs to be converted to "long", "short"

        Returns:
            None
        """

        action_mapping = {"buy": "long", "sell": "short"}

        self.config.exchange.close_position(symbol=symbol, side=action_mapping.get(side))
        logger.info("Closed position for symbol: %s with side: %s", symbol, side)

    def _create_order(self, trade_signal: TradeSignal) -> None:
        """Creates an order.

        Args:
            trade_signal (TradeSignal): the trade signal
        """

        logger.info(50 * "-")
        logger.info("create new order")

        if trade_signal.order_amount is None:
            logger.info("Amount is None. No order gets placed.")
            return

        params = {"hedged": True, "marginMode": "isolated"}

        if trade_signal.stop_loss_price:
            params["stopLoss"] = {"triggerPrice": trade_signal.stop_loss_price}

        if trade_signal.take_profit_price:
            params["takeProfit"] = {"triggerPrice": trade_signal.take_profit_price}

        # Create the order dictionary
        order = {
            "symbol": trade_signal.symbol,
            "type": trade_signal.order_type,
            "side": trade_signal.position_side,
            "amount": trade_signal.order_amount,
        }

        # If it's a limit order, include the price
        if trade_signal.order_type == "limit":
            order["price"] = trade_signal.limit_price

        # Place the order with the exchange
        self.config.exchange.create_order(**order, params=params)

        logger.info("Created order: %s", order)
        logger.info(50 * "-")

    def _determine_position(self, symbol: str) -> Literal["buy", "sell", "no_trade"]:
        """
        Determines the current trading position for a given symbol.

        Args:
            symbol (str): The trading symbol for which the position is to be determined.

        Returns:
            Literal: The current action based on the open position. Possible values are 'buy', 'sell', or 'no_trade'.
        """

        action_mapping = {"long": "buy", "short": "sell"}

        current_position = self.config.exchange.fetch_position(symbol=symbol)

        if current_position["info"] == {}:  # no position is open
            current_action = "no_trade"
        else:  # position is open
            position_side = current_position["info"].get("holdSide")  # either 'long' or 'short'
            current_action = action_mapping.get(position_side, "no_trade")

        return current_action


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
