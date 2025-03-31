import json
from typing import Any
from datamodel import TradingState, Order, Symbol, ProsperityEncoder

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750  # Ensure logs do not exceed limits

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        """IMC-Compatible Logging: Capture logs as a single formatted string"""
        log_entry = sep.join(map(str, objects)) + end
        self.logs += log_entry

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        """Flushes logs in the correct JSON format for IMC's visualizer"""
        
        log_data = json.dumps(
            [
                state.timestamp,
                self.compress_orders(orders),
                conversions,
                trader_data,
                self.truncate_logs(self.logs),
            ],
            cls=ProsperityEncoder,  # Ensure IMC format compatibility
            separators=(",", ":")  # Minimize space usage
        )
        
        print(log_data)  # Ensure correct format for IMC
        self.logs = ""  # Reset logs after flushing

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        """Converts orders to a compact format"""
        return [[order.symbol, order.price, order.quantity] for symbol_orders in orders.values() for order in symbol_orders]

    def truncate_logs(self, logs: str) -> str:
        """Truncate logs to ensure they fit within the max length"""
        return logs[: self.max_log_length - 3] + "..." if len(logs) > self.max_log_length else logs

logger = Logger()


class Trader:
    def __init__(self):
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_price_history: List[float] = []
        self.resistance = None
        self.support = None

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            return (best_bid + best_ask) / 2
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
                    logger.print("Buy order placed", product="RAINFOREST_RESIN", price=best_ask, volume=volume)

            if best_bid >= 10000 and best_bid <= 10005:
                available_to_sell = limit + pos
                volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                if volume > 0:
                    result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_bid, -volume))
                    logger.print("Sell order placed", product="RAINFOREST_RESIN", price=best_bid, volume=volume)

        # --- KELP BREAKOUT STRATEGY ---
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

                # Buy near support, sell near resistance with 4-point TP
                if best_ask <= self.support + 3:
                    available_to_buy = limit - pos
                    volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                    target_price = best_ask + 4
                    if volume > 0 and max(order_depth.buy_orders.keys(), default=0) >= target_price:
                        result["KELP"].append(Order("KELP", best_ask, volume))
                        result["KELP"].append(Order("KELP", target_price, -volume))
                        logger.print("Buy order placed", product="KELP", price=best_ask, volume=volume)

                if best_bid >= self.resistance - 3:
                    available_to_sell = limit + pos
                    volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                    target_price = best_bid - 4
                    if volume > 0 and min(order_depth.sell_orders.keys(), default=float("inf")) <= target_price:
                        result["KELP"].append(Order("KELP", best_bid, -volume))
                        result["KELP"].append(Order("KELP", target_price, volume))
                        logger.print("Sell order placed", product="KELP", price=best_bid, volume=volume)

        # ✅ Ensure logs are JSON-formatted
        logger.flush()
        return result, conversions, ""


def get_trader_instance() -> Trader:
    return Trader()
