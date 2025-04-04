import json
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Logger class remains unchanged
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
        # KELP (volatile)
        self.kelp_vw_mid_price_history = []  # Volume-weighted midprice history
        self.kelp_atr_history = []
        self.last_trade_timestamp = 0
        # RAINFOREST_RESIN (stable, OU-based)
        self.resin_mid_price_history = []
        self.resin_atr_history = []
        self.atr_period = 14
        self.mean_period = 20
        self.theta = 0.25  # Tuned for faster mean reversion
        self.trade_size = 15
        self.inventory_penalty = 0.1  # Penalty factor for positions far from zero

    def calculate_atr(self, price_history, period):
        if len(price_history) < period:
            return None
        tr_values = []
        for i in range(1, len(price_history)):
            high = price_history[i]
            low = price_history[i-1]
            tr = abs(high - low)
            tr_values.append(tr)
        if len(tr_values) < period - 1:
            return None
        return round(sum(tr_values[-period+1:]) / (period - 1))

    def calculate_mean(self, price_history, period):
        if len(price_history) < period:
            return None
        return round(sum(price_history[-period:]) / period)

    def calculate_vw_mid_price(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]):
        total_buy_volume = sum(buy_orders.values())
        total_sell_volume = sum(sell_orders.values())
        if total_buy_volume == 0 or total_sell_volume == 0:
            return None
        vw_bid = sum(price * qty for price, qty in buy_orders.items()) / total_buy_volume
        vw_ask = sum(price * qty for price, qty in sell_orders.items()) / total_sell_volume
        return round((vw_bid + vw_ask) / 2)

    def calculate_hidden_fair_value(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]):
        # Average of prices with large quantities
        large_qty_threshold = self.trade_size
        large_bids = [p for p, q in buy_orders.items() if q >= large_qty_threshold]
        large_asks = [p for p, q in sell_orders.items() if q >= large_qty_threshold]
        if not large_bids or not large_asks:
            return None
        return round((max(large_bids) + min(large_asks)) / 2)

    def calculate_inventory_adjustment(self, position, limit):
        # Penalize positions far from zero
        distance_from_zero = abs(position) / limit
        return 1 - self.inventory_penalty * distance_from_zero

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0

        # Update positions from state
        for product in state.position:
            self.position[product] = state.position[product]

        # --- KELP Trading Strategy (Market-Making for Volatile Asset) ---
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            pos = self.position["KELP"]
            limit = self.position_limits["KELP"]

            # Volume-weighted midprice
            vw_mid_price = self.calculate_vw_mid_price(order_depth.buy_orders, order_depth.sell_orders)
            if vw_mid_price:
                self.kelp_vw_mid_price_history.append(vw_mid_price)
                self.kelp_atr_history.append(vw_mid_price)
                if len(self.kelp_vw_mid_price_history) > 50:
                    self.kelp_vw_mid_price_history.pop(0)
                    self.kelp_atr_history.pop(0)

            # Calculate ATR and dynamic spread
            atr_raw = self.calculate_atr(self.kelp_atr_history, self.atr_period)
            if atr_raw is None or vw_mid_price is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            atr = max(atr_raw, 2)
            market_depth = len(order_depth.buy_orders) + len(order_depth.sell_orders)
            base_spread = max(1, atr // (2 if market_depth > 10 else 3))  # Tighter spread with higher depth

            # Liquidity check
            current_timestamp = state.timestamp
            if current_timestamp - self.last_trade_timestamp > 300 and base_spread > 1:
                base_spread = max(1, base_spread // 2)
                logger.print(f"KELP: Liquidity check triggered, reduced spread to {base_spread}")

            # Inventory adjustment
            inventory_factor = self.calculate_inventory_adjustment(pos, limit)
            bid_price = int(vw_mid_price - base_spread * inventory_factor)
            ask_price = int(vw_mid_price + base_spread * inventory_factor)

            # Place orders
            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(self.trade_size, available_to_buy)
            sell_volume = min(self.trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > 0:
                result["KELP"].append(Order("KELP", bid_price, buy_volume))
                logger.print(f"KELP: Bid at {bid_price} for {buy_volume}, ATR: {atr}, Pos: {pos}")
                self.last_trade_timestamp = current_timestamp
            if sell_volume > 0 and ask_price < float("inf"):
                result["KELP"].append(Order("KELP", ask_price, -sell_volume))
                logger.print(f"KELP: Ask at {ask_price} for {sell_volume}, ATR: {atr}, Pos: {pos}")
                self.last_trade_timestamp = current_timestamp

        # --- RAINFOREST_RESIN Trading Strategy (Market-Making with OU) ---
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            # Midprice and hidden fair value
            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            hidden_fair_value = self.calculate_hidden_fair_value(order_depth.buy_orders, order_depth.sell_orders)
            if mid_price:
                self.resin_mid_price_history.append(mid_price)
                self.resin_atr_history.append(mid_price)
                if len(self.resin_mid_price_history) > 50:
                    self.resin_mid_price_history.pop(0)
                    self.resin_atr_history.pop(0)

            # OU parameters
            mu = self.calculate_mean(self.resin_mid_price_history, self.mean_period) or (hidden_fair_value or 10000)
            atr_raw = self.calculate_atr(self.resin_atr_history, self.atr_period)
            if atr_raw is None or mid_price is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            sigma = max(atr_raw, 1)
            fair_price = round(mu + self.theta * (mu - mid_price))
            spread = max(1, sigma // 2)

            # Inventory adjustment
            inventory_factor = self.calculate_inventory_adjustment(pos, limit)
            bid_price = int(fair_price - spread * inventory_factor)
            ask_price = int(fair_price + spread * inventory_factor)

            # Place orders
            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(self.trade_size, available_to_buy)
            sell_volume = min(self.trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > 0:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, buy_volume))
                logger.print(f"RESIN: Bid at {bid_price} for {buy_volume}, Fair: {fair_price}, Pos: {pos}")
            if sell_volume > 0 and ask_price < float("inf"):
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -sell_volume))
                logger.print(f"RESIN: Ask at {ask_price} for {sell_volume}, Fair: {fair_price}, Pos: {pos}")

        # Enhanced trader_data for multi-round tuning
        trader_data = json.dumps({
            "last_trade_timestamp": self.last_trade_timestamp,
            "kelp_atr": atr_raw if "KELP" in state.order_depths else None,
            "resin_theta": self.theta,
            "resin_mu": mu if "RAINFOREST_RESIN" in state.order_depths else None
        })
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