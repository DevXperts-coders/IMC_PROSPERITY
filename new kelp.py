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

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        return {symbol: [order_depth.buy_orders, order_depth.sell_orders] for symbol, order_depth in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        return [
            [
                trade.symbol,
                trade.price,
                trade.quantity,
                trade.buyer,
                trade.seller,
                trade.timestamp,
            ]
            for trade_list in trades.values()
            for trade in trade_list
        ]

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {
            product: [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
            for product, observation in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_price_history = []

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {product: [] for product in state.order_depths.keys()}
        conversions = 0

        for product in state.position:
            self.position[product] = state.position[product]

        # --- KELP Trading Strategy (Scalping) ---
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["KELP"]
            limit = self.position_limits["KELP"]

            # Update mid-price history
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.kelp_price_history.append(mid_price)
                if len(self.kelp_price_history) > 50:  # Keep history manageable
                    self.kelp_price_history.pop(0)

            # Load trader_data
            trader_data_dict = json.loads(state.traderData) if state.traderData else {}
            open_trade = trader_data_dict.get("open_trade", False)
            direction = trader_data_dict.get("direction", None)
            entry_price = trader_data_dict.get("entry_price", None)
            quantity = trader_data_dict.get("quantity", 10)  # Fixed quantity
            initial_position = trader_data_dict.get("initial_position", 0)

            # Check if trade is still open based on position
            if open_trade:
                expected_position = initial_position + quantity if direction == "long" else initial_position - quantity
                if pos != expected_position:
                    open_trade = False
                    logger.print(f"KELP: Trade closed, position updated from {expected_position} to {pos}")

            # Entry logic: Detect trend and open a position
            if not open_trade and len(self.kelp_price_history) >= 3:
                p0, p1, p2 = self.kelp_price_history[-3:]
                if p2 > p1 > p0 and pos + quantity <= limit:
                    # Upward trend: Buy at best_ask
                    result["KELP"].append(Order("KELP", best_ask, quantity))
                    open_trade = True
                    direction = "long"
                    entry_price = best_ask
                    initial_position = pos
                    logger.print(f"KELP: Buying {quantity} at {best_ask} (Uptrend)")
                elif p2 < p1 < p0 and pos - quantity >= -limit:
                    # Downward trend: Sell at best_bid
                    result["KELP"].append(Order("KELP", best_bid, -quantity))
                    open_trade = True
                    direction = "short"
                    entry_price = best_bid
                    initial_position = pos
                    logger.print(f"KELP: Selling {quantity} at {best_bid} (Downtrend)")

            # Exit logic: Place take-profit and stop-loss orders
            if open_trade:
                if direction == "long":
                    # Take-profit: Sell at entry_price + 4
                    result["KELP"].append(Order("KELP", entry_price + 4, -quantity))
                    # Stop-loss: Sell at best_bid if price drops too low
                    if best_bid <= entry_price - 1:
                        result["KELP"].append(Order("KELP", best_bid, -quantity))
                        logger.print(f"KELP: Stop-loss triggered at {best_bid}")
                elif direction == "short":
                    # Take-profit: Buy at entry_price - 4
                    result["KELP"].append(Order("KELP", entry_price - 4, quantity))
                    # Stop-loss: Buy at best_ask if price rises too high
                    if best_ask >= entry_price + 1:
                        result["KELP"].append(Order("KELP", best_ask, quantity))
                        logger.print(f"KELP: Stop-loss triggered at {best_ask}")

            # Save trader_data
            trader_data = json.dumps({
                "open_trade": open_trade,
                "direction": direction,
                "entry_price": entry_price,
                "quantity": quantity,
                "initial_position": initial_position
            })

        # --- RAINFOREST_RESIN Trading Strategy (Unchanged) ---
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            if best_ask < float("inf") and 9995 <= best_ask <= 10000:
                available_to_buy = limit + pos
                if available_to_buy > 0:
                    volume = min(available_to_buy, -sum(order_depth.sell_orders.values()))
                    if volume > 0:
                        result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_ask, volume))
                        logger.print(f"RAINFOREST_RESIN: Buy order at {best_ask} for {volume}")
            if best_bid > 0 and 10000 <= best_bid <= 10005:
                available_to_sell = limit - pos
                if available_to_sell > 0:
                    volume = min(available_to_sell, sum(order_depth.buy_orders.values()))
                    if volume > 0:
                        result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", best_bid, -volume))
                        logger.print(f"RAINFOREST_RESIN: Sell order at {best_bid} for {volume}")

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