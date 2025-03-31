import json
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Logger boilerplate required for the Prosperity Visualizer
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state,
                        self.truncate(state.traderData, max_item_length)
                        if hasattr(state, "traderData")
                        else ""
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


# Global logger instance
logger = Logger()


class Trader:
    def __init__(self):
        # Position tracking for each product
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        # Price history for Kelp to calculate moving average
        self.kelp_prices_history: List[float] = []
        # Window length for moving average (e.g., last 20 prices)
        self.ma_window = 20

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from the order book."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return None

    def moving_average(self, prices: List[float]) -> float:
        """Calculate the simple moving average over the last ma_window prices."""
        if len(prices) < self.ma_window:
            return sum(prices) / len(prices)
        return sum(prices[-self.ma_window:]) / self.ma_window

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {product: [] for product in state.order_depths.keys()}
        conversions = 0
        trader_data = ""

        # Update positions from state
        for product in state.position:
            self.position[product] = state.position[product]

        # --- Strategy for RAINFOREST_RESIN (unchanged) ---
        # Scalping around 10000: Buy between 9995 and 10000, Sell between 10000 and 10005.
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            # Buy if best ask is between 9995 and 10000
            if best_ask < float("inf") and 9995 <= best_ask <= 10000:
                available_to_buy = limit + pos
                if available_to_buy > 0:
                    volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                    if volume > 0:
                        result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_ask, volume))
                        logger.print(f"RAINFOREST_RESIN: Buy order at {best_ask} for {volume}")
            # Sell if best bid is between 10000 and 10005
            if best_bid > 0 and 10000 <= best_bid <= 10005:
                available_to_sell = limit - pos
                if available_to_sell > 0:
                    volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                    if volume > 0:
                        result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_bid, -volume))
                        logger.print(f"RAINFOREST_RESIN: Sell order at {best_bid} for {volume}")

        # --- Mean Reversion Scalping Strategy for KELP ---
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price is not None:
                # Update price history (keep up to last 50 points)
                self.kelp_prices_history.append(mid_price)
                if len(self.kelp_prices_history) > 50:
                    self.kelp_prices_history = self.kelp_prices_history[-50:]

                # Only proceed if we have enough history to compute a moving average
                if len(self.kelp_prices_history) >= 5:
                    ma = self.moving_average(self.kelp_prices_history)
                    pos = self.position["KELP"]
                    limit = self.position_limits["KELP"]

                    # If price is below the moving average by at least 0.5, consider buying (expecting reversion)
                    if mid_price < ma - 0.5:
                        available_to_buy = limit + pos
                        if available_to_buy > 0 and order_depth.sell_orders:
                            entry_price = min(order_depth.sell_orders.keys())
                            target = entry_price + 1  # target profit of 1 point
                            # Only take the trade if the current best bid covers our target profit
                            if order_depth.buy_orders and max(order_depth.buy_orders.keys()) >= target:
                                trade_volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                                if trade_volume > 0:
                                    result["KELP"].append(Order("KELP", entry_price, trade_volume))
                                    result["KELP"].append(Order("KELP", max(order_depth.buy_orders.keys()), -trade_volume))
                                    logger.print(f"KELP: Mean reversion BUY at {entry_price}, exit at {max(order_depth.buy_orders.keys())} for {trade_volume}")

                    # If price is above the moving average by at least 0.5, consider selling (expecting reversion)
                    if mid_price > ma + 0.5:
                        available_to_sell = limit - pos
                        if available_to_sell > 0 and order_depth.buy_orders:
                            entry_price = max(order_depth.buy_orders.keys())
                            target = entry_price - 1  # target profit of 1 point
                            if order_depth.sell_orders and min(order_depth.sell_orders.keys()) <= target:
                                trade_volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                                if trade_volume > 0:
                                    result["KELP"].append(Order("KELP", entry_price, -trade_volume))
                                    result["KELP"].append(Order("KELP", min(order_depth.sell_orders.keys()), trade_volume))
                                    logger.print(f"KELP: Mean reversion SELL at {entry_price}, exit at {min(order_depth.sell_orders.keys())} for {trade_volume}")

        # Flush logs in the required format before returning
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def toJSON(self, result: Any) -> str:
        return json.dumps(result, cls=ProsperityEncoder)


def get_trader_instance() -> Trader:
    return Trader()


def process_exchange_update(state: TradingState):
    trader = get_trader_instance()
    result = trader.run(state)
    return trader.toJSON(result)
