import json
import numpy as np
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json(["", orders, conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json(["", orders, conversions, "", self.truncate(self.logs, max_item_length)]))
        self.logs = ""

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_prices = []  # Store KELP prices for Bollinger Bands
        self.bb_period = 20  # Rolling window period

    def calculate_bollinger_bands(self):
        if len(self.kelp_prices) < self.bb_period:
            return None, None
        prices = np.array(self.kelp_prices[-self.bb_period:])
        sma = np.mean(prices)
        std_dev = np.std(prices)
        lower_band = sma - (2 * std_dev)
        upper_band = sma + (2 * std_dev)
        return lower_band, upper_band

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
                    available_to_buy = limit + pos
                    if available_to_buy > 0:
                        buy_volume = min(5, available_to_buy)
                        result["KELP"].append(Order("KELP", best_ask, buy_volume))
                        logger.print(f"KELP: Buy {buy_volume} @ {best_ask} (Lower BB: {lower_band})")
                
                if best_bid > 0 and best_bid >= upper_band:
                    available_to_sell = limit - pos
                    if available_to_sell > 0:
                        sell_volume = min(5, available_to_sell)
                        result["KELP"].append(Order("KELP", best_bid, -sell_volume))
                        logger.print(f"KELP: Sell {sell_volume} @ {best_bid} (Upper BB: {upper_band})")

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

def get_trader_instance() -> Trader:
    return Trader()

def process_exchange_update(state: TradingState):
    trader = get_trader_instance()
    result = trader.run(state)
    return json.dumps(result, cls=ProsperityEncoder)
