import json
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_vw_mid_price_history = []
        self.kelp_atr_history = []
        self.last_trade_timestamp = 0
        self.resin_mid_price_history = []
        self.resin_atr_history = []
        self.atr_period = 10
        self.mean_period = 20
        self.theta = 0.3
        self.base_trade_size = 15
        self.kelp_profit = 0
        self.resin_profit = 0
        self.inventory_penalty = 0.03

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
    
    def calculate_vw_mid_price(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]):
        top_n = 3
        sorted_bids = sorted(buy_orders.items(), key=lambda x: x[0], reverse=True)[:top_n]
        sorted_asks = sorted(sell_orders.items(), key=lambda x: x[0])[:top_n]
        total_buy_volume = sum(qty for _, qty in sorted_bids)
        total_sell_volume = sum(qty for _, qty in sorted_asks)
        if total_buy_volume == 0 or total_sell_volume == 0:
            return None
        vw_bid = sum(price * qty for price, qty in sorted_bids) / total_buy_volume
        vw_ask = sum(price * qty for price, qty in sorted_asks) / total_sell_volume
        return round((vw_bid + vw_ask) / 2)

    def calculate_hidden_fair_value(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]):
        large_qty_threshold = self.base_trade_size
        large_bids = [p for p, q in buy_orders.items() if q >= large_qty_threshold]
        large_asks = [p for p, q in sell_orders.items() if q >= large_qty_threshold]
        if not large_bids or not large_asks:
            return None
        return round((max(large_bids) + min(large_asks)) / 2)

    def calculate_inventory_adjustment(self, position, limit):
        distance_from_zero = abs(position) / limit
        return 1 - self.inventory_penalty * distance_from_zero

    def update_profit(self, state: TradingState, vw_mid_price: float = None, fair_price: float = None):
        for symbol, trades in state.own_trades.items():
            for trade in trades:
                if trade.timestamp == state.timestamp:
                    if symbol == "KELP" and vw_mid_price is not None:
                        if trade.buyer == "":
                            self.kelp_profit += trade.quantity * (trade.price - vw_mid_price)
                        elif trade.seller == "":
                            self.kelp_profit += trade.quantity * (vw_mid_price - trade.price)
                    elif symbol == "RAINFOREST_RESIN" and fair_price is not None:
                        if trade.buyer == "":
                            self.resin_profit += trade.quantity * (trade.price - fair_price)
                        elif trade.seller == "":
                            self.resin_profit += trade.quantity * (fair_price - trade.price)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0

        for product in state.position:
            self.position[product] = state.position[product]

        if "KELP" in state.order_depths:
            order_depth = state.order_depths["KELP"]
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=float("inf"))
            pos = self.position["KELP"]
            limit = self.position_limits["KELP"]

            vw_mid_price = self.calculate_vw_mid_price(order_depth.buy_orders, order_depth.sell_orders)
            hidden_fair = self.calculate_hidden_fair_value(order_depth.buy_orders, order_depth.sell_orders)
            if vw_mid_price:
                self.kelp_vw_mid_price_history.append(vw_mid_price)
                self.kelp_atr_history.append(vw_mid_price)
                if len(self.kelp_vw_mid_price_history) > 50:
                    self.kelp_vw_mid_price_history.pop(0)
                    self.kelp_atr_history.pop(0)

            atr_raw = self.calculate_atr(self.kelp_atr_history, self.atr_period)
            if atr_raw is None or vw_mid_price is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp, "kelp_profit": self.kelp_profit, "resin_profit": self.resin_profit})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data

            atr = max(atr_raw, 1)
            market_depth = len(order_depth.buy_orders) + len(order_depth.sell_orders)
            base_spread = max(1, atr // (1 if market_depth > 20 else 4))
            if len(self.kelp_vw_mid_price_history) >= 3:
                trend = self.kelp_vw_mid_price_history[-1] - self.kelp_vw_mid_price_history[-3]
                if abs(trend) > atr:
                    base_spread *= 2
            if state.timestamp - self.last_trade_timestamp > 100 and base_spread > 1:
                base_spread = 1
                logger.print(f"KELP: Liquidity boost, spread reduced to {base_spread}")

            trade_size = min(self.base_trade_size + atr // 2, limit // 2)
            bid_price = int(vw_mid_price - base_spread)
            ask_price = int(vw_mid_price + base_spread)
            if hidden_fair and abs(hidden_fair - vw_mid_price) > base-spread:
                bid_price = min(bid_price, hidden_fair - 1)
                ask_price = max(ask_price, hidden_fair + 1)

            self.update_profit(state, vw_mid_price=vw_mid_price)
            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > 0:
                result["KELP"].append(Order("KELP", bid_price, buy_volume))
                logger.print(f"KELP: Bid at {bid_price} for {buy_volume}, ATR: {atr}, Pos: {pos}")
                self.last_trade_timestamp = state.timestamp
            if sell_volume > 0 and ask_price < float("inf"):
                result["KELP"].append(Order("KELP", ask_price, -sell_volume))
                logger.print(f"KELP: Ask at {ask_price} for {sell_volume}, ATR: {atr}, Pos: {pos}")
                self.last_trade_timestamp = state.timestamp

        if "RAINFOREST_RESIN" in state.order_depths:
            order_depth = state.order_depths["RAINFOREST_RESIN"]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
            pos = self.position["RAINFOREST_RESIN"]
            limit = self.position_limits["RAINFOREST_RESIN"]

            mid_price = round((best_bid + best_ask) / 2) if best_bid and best_ask < float("inf") else None
            hidden_fair = self.calculate_hidden_fair_value(order_depth.buy_orders, order_depth.sell_orders)
            if mid_price:
                self.resin_mid_price_history.append(mid_price)
                self.resin_atr_history.append(mid_price)
                if len(self.resin_mid_price_history) > 50:
                    self.resin_mid_price_history.pop(0)
                    self.resin_atr_history.pop(0)

            mu = self.calculate_mean(self.resin_mid_price_history, self.mean_period) or 10000
            atr_raw = self.calculate_atr(self.resin_atr_history, self.atr_period)
            if atr_raw is None or mid_price is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp, "kelp_profit": self.kelp_profit, "resin_profit": self.resin_profit})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data

            sigma = max(atr_raw, 1)
            fair_price = round(mu + self.theta * (mu - mid_price))
            if hidden_fair and abs(hidden_fair - fair_price) > sigma:
                fair_price = round((fair_price + hidden_fair) / 2)
            spread = max(1, sigma // 5)

            adjustment = self.calculate_inventory_adjustment(pos, limit)
            bid_price = int(fair_price - spread * adjustment)
            ask_price = int(fair_price + spread * adjustment)

            self.update_profit(state, fair_price=fair_price)
            trade_size = min(self.base_trade_size + sigma // 2, limit // 2)
            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > 0:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, buy_volume))
                logger.print(f"RESIN: Bid at {bid_price} for {buy_volume}, Fair: {fair_price}, Pos: {pos}")
            if sell_volume > 0 and ask_price < float("inf"):
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -sell_volume))
                logger.print(f"RESIN: Ask at {ask_price} for {sell_volume}, Fair: {fair_price}, Pos: {pos}")

        trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp, "kelp_profit": self.kelp_profit, "resin_profit": self.resin_profit})
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def toJSON(self, result: Any) -> str:
        return json.dumps(result, cls=ProsperityEncoder)