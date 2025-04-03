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
        self.kelp_volume_history = []
        self.last_trade_timestamp = 0
        # For RAINFOREST_RESIN (stable, OU-based)
        self.resin_mid_price_history = []
        self.resin_atr_history = []
        self.resin_short_ma = []  # 5-period moving average for trend
        self.atr_period = 14
        self.mean_period = 20
        self.short_ma_period = 5
        self.theta = 0.2
        self.base_trade_size = 15

    def calculate_atr(self, price_history, period, trades=None):
        if len(price_history) < period:
            return None
        tr_values = []
        for i in range(1, len(price_history)):
            high = price_history[i]
            low = price_history[i-1]
            tr = abs(high - low)
            if trades and i < len(trades):
                tr = max(tr, abs(price_history[i] - trades[i].price))  # Include price change
            tr_values.append(tr)
        return round(sum(tr_values[-period:]) / period) if len(trades) >= period else None

    def calculate_mean(self, price_history, period):
        if len(price_history) < period:
            return None
        return round(sum(price_history[-period:]) / period)

    def calculate_short_ma(self, price_history, period):
        if len(price_history) < period:
            return None
        return round(sum(price_history[-period:]) / period)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0

        # Update positions from state
        for product in state.position:
            self.position[product] = state.position[product]

        # --- KELP Trading Strategy (Market-Making with Dynamic Spread) ---
        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["KELP"]
            limit = self.position_limits["KELP"]

            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.kelp_mid_price_history.append(mid_price)
                self.kelp_atr_history.append(mid_price)
                self.kelp_volume_history.append(sum(order_depth.buy_orders.values()) + sum(order_depth.sell_orders.values()))
                if len(self.kelp_mid_price_history) > 50:
                    self.kelp_mid_price_history.pop(0)
                    self.kelp_atr_history.pop(0)
                    self.kelp_volume_history.pop(0)

            atr = self.calculate_atr(self.kelp_atr_history, self.atr_period, state.market_trades.get("KELP", []))
            if atr is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            atr = max(atr, 2)

            # Dynamic spread based on ATR and volume
            avg_volume = self.calculate_mean(self.kelp_volume_history, self.mean_period) or 1
            volume_factor = min(1.0, avg_volume / 20)  # Normalize volume (assume 20 as baseline)
            base_spread = max(1, int(atr // (3 * volume_factor)))  # Tighter spread with higher volume

            current_timestamp = state.timestamp
            if current_timestamp - self.last_trade_timestamp > 300 and base_spread > 1:
                base_spread = max(1, base_spread // 2)
                logger.print(f"KELP: Liquidity check triggered, reduced spread to {base_spread}")

            bid_price = int(mid_price - base_spread) if mid_price else best_bid
            ask_price = int(mid_price + base_spread) if mid_price else best_ask

            # Dynamic trade size based on order book depth
            buy_depth = sum(qty for qty in order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            sell_depth = sum(qty for qty in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            max_depth = max(buy_depth, sell_depth) or 1
            trade_size = min(self.base_trade_size, int(self.base_trade_size * (max_depth / 30)))  # Scale with depth

            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)

            if buy_volume > 0 and bid_price > 0:
                result["KELP"].append(Order("KELP", bid_price, buy_volume))
                logger.print(f"KELP: Bid at {bid_price} for {buy_volume}, ATR: {atr}, Volume Factor: {volume_factor}")
                self.last_trade_timestamp = current_timestamp
            if sell_volume > 0 and ask_price < float("inf"):
                result["KELP"].append(Order("KELP", ask_price, -sell_volume))
                logger.print(f"KELP: Ask at {ask_price} for {sell_volume}, ATR: {atr}, Volume Factor: {volume_factor}")
                self.last_trade_timestamp = current_timestamp

            # Position rebalancing
            if abs(pos) > limit * 0.8:  # Rebalance if 80% of limit
                if pos > 0 and ask_price < float("inf"):
                    rebalance_volume = min(pos, trade_size)
                    result["KELP"].append(Order("KELP", ask_price, -rebalance_volume))
                    logger.print(f"KELP: Rebalancing sell {rebalance_volume} at {ask_price}")
                elif pos < 0 and bid_price > 0:
                    rebalance_volume = min(-pos, trade_size)
                    result["KELP"].append(Order("KELP", bid_price, rebalance_volume))
                    logger.print(f"KELP: Rebalancing buy {rebalance_volume} at {bid_price}")

        # --- RAINFOREST_RESIN Trading Strategy (OU with Trend) ---
        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            if mid_price:
                self.resin_mid_price_history.append(mid_price)
                self.resin_atr_history.append(mid_price)
                self.resin_short_ma.append(mid_price)
                if len(self.resin_mid_price_history) > 50:
                    self.resin_mid_price_history.pop(0)
                    self.resin_atr_history.pop(0)
                    self.resin_short_ma.pop(0)

            mu = self.calculate_mean(self.resin_mid_price_history, self.mean_period) or 10000
            atr = self.calculate_atr(self.resin_atr_history, self.atr_period, state.market_trades.get("RAINFOREST_RESIN", []))
            short_ma = self.calculate_short_ma(self.resin_short_ma, self.short_ma_period)
            if atr is None or mid_price is None or short_ma is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data
            sigma = max(atr, 1)
            fair_price = round(mu + self.theta * (mu - mid_price))
            spread = max(1, sigma // 2)

            # Trend adjustment
            trend = 1 if mid_price > short_ma else -1 if mid_price < short_ma else 0
            if trend == 1:  # Uptrend, tighten buy spread
                bid_price = int(fair_price - spread // 2)
            elif trend == -1:  # Downtrend, tighten sell spread
                ask_price = int(fair_price + spread // 2)
            else:
                bid_price = int(fair_price - spread)
                ask_price = int(fair_price + spread)

            # Dynamic trade size
            buy_depth = sum(qty for qty in order_depth.buy_orders.values()) if order_depth.buy_orders else 0
            sell_depth = sum(qty for qty in order_depth.sell_orders.values()) if order_depth.sell_orders else 0
            max_depth = max(buy_depth, sell_depth) or 1
            trade_size = min(self.base_trade_size, int(self.base_trade_size * (max_depth / 30)))

            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)

            if buy_volume > 0 and bid_price > 0:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, buy_volume))
                logger.print(f"RAINFOREST_RESIN: Bid at {bid_price} for {buy_volume}, Fair: {fair_price}, Trend: {trend}")
            if sell_volume > 0 and ask_price < float("inf"):
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -sell_volume))
                logger.print(f"RAINFOREST_RESIN: Ask at {ask_price} for {sell_volume}, Fair: {fair_price}, Trend: {trend}")

            # Position rebalancing
            if abs(pos) > limit * 0.8:
                if pos > 0 and ask_price < float("inf"):
                    rebalance_volume = min(pos, trade_size)
                    result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -rebalance_volume))
                    logger.print(f"RAINFOREST_RESIN: Rebalancing sell {rebalance_volume} at {ask_price}")
                elif pos < 0 and bid_price > 0:
                    rebalance_volume = min(-pos, trade_size)
                    result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, rebalance_volume))
                    logger.print(f"RAINFOREST_RESIN: Rebalancing buy {rebalance_volume} at {bid_price}")

        trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp})
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