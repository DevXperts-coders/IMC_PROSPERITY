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
        # For RAINFOREST_RESIN (stable, OU-based)
        self.resin_mid_price_history = []
        self.resin_atr_history = []
        self.atr_period = 14  # Period for Average True Range
        self.mean_period = 20  # Period for long-term mean (mu) in OU
        self.theta = 0.1  # Mean reversion speed (fixed, adjustable)
        self.trade_size = 3  # Base trade size for market-making
        self.max_position = 30  # Maximum position in either direction

    def calculate_atr(self, high_low_history, period):
        if len(high_low_history) < period:
            return None
        tr_values = []
        for i in range(1, len(high_low_history)):
            high = high_low_history[i]
            low = high_low_history[i-1]
            tr = abs(high - low)  # Simplified True Range (high - low)
            tr_values.append(tr)
        if len(tr_values) < period - 1:
            return None
        return round(sum(tr_values[-period+1:]) / (period - 1))

    def calculate_mean(self, price_history, period):
        if len(price_history) < period:
            return None
        return round(sum(price_history[-period:]) / period)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {product: [] for product in state.order_depths.keys()}
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

            # Calculate mid-price and update history
            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.kelp_mid_price_history.append(mid_price)
                self.kelp_atr_history.append(mid_price)
                if len(self.kelp_mid_price_history) > 100:
                    self.kelp_mid_price_history.pop(0)
                    self.kelp_atr_history.pop(0)

            # Calculate ATR for volatility adjustment
            atr_raw = self.calculate_atr(self.kelp_atr_history, self.atr_period)
            if atr_raw is None:
                trader_data = json.dumps({})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            atr = max(atr_raw, 2)
            base_spread = max(2, atr // 2)

            # Calculate bid and ask prices
            bid_price = int(mid_price - base_spread) if mid_price else best_bid
            ask_price = int(mid_price + base_spread) if mid_price else best_ask

            # Inventory management
            available_to_buy = min(self.max_position - pos, limit - pos)
            available_to_sell = min(self.max_position + pos, limit + pos)
            buy_volume = min(self.trade_size, available_to_buy)
            sell_volume = min(self.trade_size, available_to_sell)
            if pos > 15:  # Skewed long
                sell_volume = min(self.trade_size * 2, available_to_sell)
                buy_volume = min(self.trade_size // 2, available_to_buy)
            elif pos < -15:  # Skewed short
                buy_volume = min(self.trade_size * 2, available_to_buy)
                sell_volume = min(self.trade_size // 2, available_to_sell)

            # Place orders
            buy_volume = min(buy_volume, sum([q for p, q in order_depth.buy_orders.items() if p >= bid_price]))
            sell_volume = min(sell_volume, -sum([q for p, q in order_depth.sell_orders.items() if p <= ask_price]))
            if buy_volume > 0 and bid_price > 0:
                result["KELP"].append(Order("KELP", bid_price, buy_volume))
                logger.print(f"KELP: Market-making bid at {bid_price} for {buy_volume}, ATR: {atr}, Position: {pos}")
            if sell_volume > 0 and ask_price < float("inf"):
                result["KELP"].append(Order("KELP", ask_price, -sell_volume))
                logger.print(f"KELP: Market-making ask at {ask_price} for {sell_volume}, ATR: {atr}, Position: {pos}")

        # --- RAINFOREST_RESIN Trading Strategy (Market-Making with OU) ---
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            # Calculate mid-price and update history
            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.resin_mid_price_history.append(mid_price)
                self.resin_atr_history.append(mid_price)
                if len(self.resin_mid_price_history) > 100:
                    self.resin_mid_price_history.pop(0)
                    self.resin_atr_history.pop(0)

            # Calculate OU parameters
            mu = self.calculate_mean(self.resin_mid_price_history, self.mean_period) or 10000  # Default to 10000 if not enough data
            atr_raw = self.calculate_atr(self.resin_atr_history, self.atr_period)
            if atr_raw is None or mid_price is None:
                trader_data = json.dumps({})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            sigma = max(atr_raw, 1)  # Minimum volatility of 1 for stable asset

            # Estimate fair price with OU mean-reversion (simplified discrete step)
            fair_price = round(mu + self.theta * (mu - mid_price))  # Adjust toward mean
            spread = max(2, sigma)  # Spread based on volatility, minimum 2 for stable asset

            # Calculate bid and ask prices
            bid_price = int(fair_price - spread)
            ask_price = int(fair_price + spread)

            # Inventory management
            available_to_buy = min(self.max_position - pos, limit - pos)
            available_to_sell = min(self.max_position + pos, limit + pos)
            buy_volume = min(self.trade_size, available_to_buy)
            sell_volume = min(self.trade_size, available_to_sell)
            if pos > 15:  # Skewed long
                sell_volume = min(self.trade_size * 2, available_to_sell)
                buy_volume = min(self.trade_size // 2, available_to_buy)
            elif pos < -15:  # Skewed short
                buy_volume = min(self.trade_size * 2, available_to_buy)
                sell_volume = min(self.trade_size // 2, available_to_sell)

            # Place orders
            buy_volume = min(buy_volume, sum([q for p, q in order_depth.buy_orders.items() if p >= bid_price]))
            sell_volume = min(sell_volume, -sum([q for p, q in order_depth.sell_orders.items() if p <= ask_price]))
            if buy_volume > 0 and bid_price > 0:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, buy_volume))
                logger.print(f"RAINFOREST_RESIN: OU bid at {bid_price} for {buy_volume}, Fair: {fair_price}, Position: {pos}")
            if sell_volume > 0 and ask_price < float("inf"):
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -sell_volume))
                logger.print(f"RAINFOREST_RESIN: OU ask at {ask_price} for {sell_volume}, Fair: {fair_price}, Position: {pos}")

        # Save trader_data (empty for simplicity)
        trader_data = json.dumps({})
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