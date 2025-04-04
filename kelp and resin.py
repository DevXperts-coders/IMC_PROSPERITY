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
        # For KELP (volatile)
        self.kelp_mid_price_history = []
        self.kelp_atr_history = []
        self.last_trade_timestamp_kelp = 0  # Track last trade for KELP
        self.last_trade_timestamp_resin = 0  # Track last trade for RAINFOREST_RESIN
        # For RAINFOREST_RESIN (stable, OU-based)
        self.resin_mid_price_history = []
        self.resin_atr_history = []
        self.atr_period = 10  # Period for Average True Range
        self.mean_period = 20  # Period for long-term mean (mu) in OU
        self.theta = 0.3  # Reduced mean reversion speed
        self.base_trade_size = 10  # Reduced base trade size
        self.boosted_trade_size = 15  # Reduced boosted trade size

    def calculate_atr(self, high_low_history, period):
        if len(high_low_history) < period:
            return None
        tr_values = []
        for i in range(1, len(high_low_history)):
            high = high_low_history[i]
            low = high_low_history[i-1]
            tr = abs(high - low)
            tr_values.append(tr)
        if len(tr_values) < period - 1:
            return None
        return round(sum(tr_values[-period+1:]) / (period - 1))

    def calculate_mean(self, price_history, period):
        if len(price_history) < period:
            return None
        return round(sum(price_history[-period:]) / period)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0

        # Update positions from state
        for product in state.position:
            self.position[product] = state.position[product]

        # --- KELP Trading Strategy (Market-Making for Volatile Asset) ---
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["KELP"]
            limit = self.position_limits["KELP"]

            # Check market depth
            bid_depth = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            ask_depth = sum(abs(q) for q in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            min_depth = 5  # Minimum depth to place orders

            # Calculate mid-price and update history
            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.kelp_mid_price_history.append(mid_price)
                self.kelp_atr_history.append(mid_price)
                if len(self.kelp_mid_price_history) > 50:
                    self.kelp_mid_price_history.pop(0)
                    self.kelp_atr_history.pop(0)

            # Calculate ATR for volatility adjustment
            atr_raw = self.calculate_atr(self.kelp_atr_history, self.atr_period)
            if atr_raw is None:
                trader_data = json.dumps({
                    "last_trade_timestamp_kelp": self.last_trade_timestamp_kelp,
                    "last_trade_timestamp_resin": self.last_trade_timestamp_resin
                })
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            atr = max(atr_raw, 2)
            base_spread = max(1, atr // 3)  # Slightly wider spread

            # Liquidity check: Halve spread if no trades in 300 timestamps
            current_timestamp = state.timestamp
            if current_timestamp - self.last_trade_timestamp_kelp > 300 and base_spread > 1:
                base_spread = max(1, base_spread // 2)
                logger.print(f"KELP: Liquidity check triggered, reduced spread to {base_spread}")

            # Dynamic trade size: Boost if no trades for 500 timestamps
            trade_size = self.base_trade_size
            if current_timestamp - self.last_trade_timestamp_kelp > 500:
                trade_size = self.boosted_trade_size
                logger.print(f"KELP: Boosted trade size to {trade_size} due to inactivity")

            # Calculate bid and ask prices
            bid_price = int(mid_price - base_spread) if mid_price else best_bid
            ask_price = int(mid_price + base_spread) if mid_price else best_ask

            # Inventory management (use full limit)
            available_to_buy = limit - pos
            available_to_sell = limit + pos

            # Place orders only if sufficient market depth
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > 0 and bid_depth >= min_depth:
                result["KELP"].append(Order("KELP", bid_price, buy_volume))
                logger.print(f"KELP: Market-making bid at {bid_price} for {buy_volume}, ATR: {atr}, Position: {pos}")
                self.last_trade_timestamp_kelp = current_timestamp
            if sell_volume > 0 and ask_price < float("inf") and ask_depth >= min_depth:
                result["KELP"].append(Order("KELP", ask_price, -sell_volume))
                logger.print(f"KELP: Market-making ask at {ask_price} for {sell_volume}, ATR: {atr}, Position: {pos}")
                self.last_trade_timestamp_kelp = current_timestamp

        # --- RAINFOREST_RESIN Trading Strategy (Market-Making with OU) ---
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            # Check market depth
            bid_depth = sum(order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            ask_depth = sum(abs(q) for q in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            min_depth = 5  # Minimum depth to place orders

            # Calculate mid-price and update history
            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.resin_mid_price_history.append(mid_price)
                self.resin_atr_history.append(mid_price)
                if len(self.resin_mid_price_history) > 50:
                    self.resin_mid_price_history.pop(0)
                    self.resin_atr_history.pop(0)

            # Calculate OU parameters
            mu = self.calculate_mean(self.resin_mid_price_history, self.mean_period) or 10000
            atr_raw = self.calculate_atr(self.resin_atr_history, self.atr_period)
            if atr_raw is None or mid_price is None:
                trader_data = json.dumps({
                    "last_trade_timestamp_kelp": self.last_trade_timestamp_kelp,
                    "last_trade_timestamp_resin": self.last_trade_timestamp_resin
                })
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            sigma = max(atr_raw, 1)
            fair_price = round(mu + self.theta * (mu - mid_price))
            spread = max(1, sigma // 2)  # Wider spread to reduce overtrading

            # Dynamic trade size: Boost if no trades for 500 timestamps
            trade_size = self.base_trade_size
            if current_timestamp - self.last_trade_timestamp_resin > 500:
                trade_size = self.boosted_trade_size
                logger.print(f"RAINFOREST_RESIN: Boosted trade size to {trade_size} due to inactivity")

            # Calculate bid and ask prices
            bid_price = int(fair_price - spread)
            ask_price = int(fair_price + spread)

            # Inventory management (use full limit)
            available_to_buy = limit - pos
            available_to_sell = limit + pos

            # Place orders only if sufficient market depth
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > 0 and bid_depth >= min_depth:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, buy_volume))
                logger.print(f"RAINFOREST_RESIN: OU bid at {bid_price} for {buy_volume}, Fair: {fair_price}, Position: {pos}")
                self.last_trade_timestamp_resin = current_timestamp
            if sell_volume > 0 and ask_price < float("inf") and ask_depth >= min_depth:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -sell_volume))
                logger.print(f"RAINFOREST_RESIN: OU ask at {ask_price} for {sell_volume}, Fair: {fair_price}, Position: {pos}")
                self.last_trade_timestamp_resin = current_timestamp

        # Save trader_data
        trader_data = json.dumps({
            "last_trade_timestamp_kelp": self.last_trade_timestamp_kelp,
            "last_trade_timestamp_resin": self.last_trade_timestamp_resin
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