import json
import numpy as np
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = []  # Store logs as a list of dictionaries
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        message = sep.join(map(str, objects))
        try:
            log_data = json.loads(message)  # Ensure it's valid JSON
        except json.JSONDecodeError:
            log_data = {"message": message}  # Fallback for plain text logs
        
        self.logs.append(log_data)  # Store as JSON objects

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", []]))
        max_item_length = (self.max_log_length - base_length) // 3
        truncated_logs = self.logs[:max_item_length]
        
        print(self.to_json([self.compress_state(state, trader_data), self.compress_orders(orders), conversions, trader_data, truncated_logs]))
        self.logs = []

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            str(trader_data).strip(),
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [order_depth.buy_orders, order_depth.sell_orders] for symbol, order_depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp] for arr in trades.values() for trade in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {product: [
            observation.bidPrice, observation.askPrice, observation.transportFees,
            observation.exportTariff, observation.importTariff,
            observation.sugarPrice, observation.sunlightIndex
        ] for product, observation in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for arr in orders.values() for order in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

logger = Logger()

class Trader:
    def __init__(self):
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_prices = []  
        self.bb_period = 20  

    def calculate_bollinger_bands(self):
        if len(self.kelp_prices) < self.bb_period:
            return None, None
        prices = np.array(self.kelp_prices[-self.bb_period:])
        sma = np.mean(prices)
        std_dev = np.std(prices)
        return sma - (2 * std_dev), sma + (2 * std_dev)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0
        trader_data = ""

        for product in state.position:
            self.position[product] = state.position[product]

        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            if best_ask < float("inf") and 9995 <= best_ask <= 10000:
                available_to_buy = limit + pos
                if available_to_buy > 0:
                    volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                    if volume > 0:
                        result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_ask, volume))
                        logger.print(f"RAINFOREST_RESIN: Buy {volume} @ {best_ask}")

            if best_bid > 0 and 10000 <= best_bid <= 10005:
                available_to_sell = limit - pos
                if available_to_sell > 0:
                    volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                    if volume > 0:
                        result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_bid, -volume))
                        logger.print(f"RAINFOREST_RESIN: Sell {volume} @ {best_bid}")

        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))

            self.kelp_prices.append(best_ask)
            lower_band, upper_band = self.calculate_bollinger_bands()

            if lower_band is not None:
                pos = self.position["KELP"]
                limit = self.position_limits["KELP"]
                if best_ask <= lower_band:
                    if pos < limit:
                        result["KELP"].append(Order("KELP", best_ask, 5))
                if best_bid >= upper_band:
                    if pos > -limit:
                        result["KELP"].append(Order("KELP", best_bid, -5))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

def get_trader_instance() -> Trader:
    return Trader()

def process_exchange_update(state: TradingState):
    trader = get_trader_instance()
    result = trader.run(state)
    return json.dumps(result, cls=ProsperityEncoder)
