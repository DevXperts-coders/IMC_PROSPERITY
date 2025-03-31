import json
from typing import Any, Dict, List
from datamodel import Order, OrderDepth, Symbol, Trade, TradingState, ProsperityEncoder


# Required Logger for Prosperity Visualizer
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self) -> None:
        print(self.logs)
        self.logs = ""


# Global logger instance
logger = Logger()


class Trader:
    def __init__(self):
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_price_history: List[float] = []
        self.resistance = None
        self.support = None

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid-price from the order book."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            return (best_bid + best_ask) / 2
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return None

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {"RAINFOREST_RESIN": [], "KELP": []}
        conversions = 0

        for product in self.position:
            if product in state.position:
                self.position[product] = state.position[product]

        # --- RAINFOREST RESIN SCALPING ---
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            if best_ask <= 10000 and best_ask >= 9995:
                available_to_buy = limit - pos
                volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                if volume > 0:
                    result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_ask, volume))
                    logger.print(f"RAINFOREST_RESIN: Buy {volume} @ {best_ask}")

            if best_bid >= 10000 and best_bid <= 10005:
                available_to_sell = limit + pos
                volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                if volume > 0:
                    result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_bid, -volume))
                    logger.print(f"RAINFOREST_RESIN: Sell {volume} @ {best_bid}")

        # --- KELP SCALPING STRATEGY ---
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            mid_price = self.calculate_mid_price(order_depth)

            if mid_price is not None:
                self.kelp_price_history.append(mid_price)
                if len(self.kelp_price_history) > 50:
                    self.kelp_price_history.pop(0)

                best_bid = max(order_depth.buy_orders.keys(), default=0)
                best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
                pos = self.position["KELP"]
                limit = self.position_limits["KELP"]

                # Determine Support & Resistance dynamically
                self.resistance = max(self.kelp_price_history, default=mid_price) + 5
                self.support = min(self.kelp_price_history, default=mid_price) - 5

                # **BUY when price near support and enough liquidity**
                if best_ask <= self.support + 3:
                    available_to_buy = limit - pos
                    volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                    target_price = best_ask + 4
                    if volume > 0 and max(order_depth.buy_orders.keys(), default=0) >= target_price:
                        result["KELP"].append(Order("KELP", best_ask, volume))
                        result["KELP"].append(Order("KELP", target_price, -volume))
                        logger.print(f"KELP: Buy {volume} @ {best_ask}, TP @ {target_price}")

                # **SELL when price near resistance and enough liquidity**
                if best_bid >= self.resistance - 3:
                    available_to_sell = limit + pos
                    volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                    target_price = best_bid - 4
                    if volume > 0 and min(order_depth.sell_orders.keys(), default=float("inf")) <= target_price:
                        result["KELP"].append(Order("KELP", best_bid, -volume))
                        result["KELP"].append(Order("KELP", target_price, volume))
                        logger.print(f"KELP: Sell {volume} @ {best_bid}, TP @ {target_price}")

        # **Ensure logs are formatted correctly**
        logger.flush()
        return result, conversions, ""


def get_trader_instance() -> Trader:
    return Trader()
