import json
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Trader:
    def __init__(self):
        self.position = {"KELP": 0, "RAINFOREST_RESIN": 0}
        self.position_limits = {"KELP": 50, "RAINFOREST_RESIN": 50}
        self.kelp_vw_mid_price_history = []
        self.kelp_atr_history = []
        self.last_trade_timestamp = 0
        self.resin_mid_price_history = []
        self.resin_atr_history = []
        self.atr_period = 20  # Increased for smoother volatility
        self.mean_period = 30  # Increased for longer-term mean
        self.theta = 0.15  # Reduced for cautious reversion
        self.base_trade_size = 5  # Reduced to minimize risk
        self.kelp_profit = 0
        self.resin_profit = 0
        self.inventory_penalty = 0.1  # Stronger adjustment

    def calculate_atr(self, high_low_history, period):
        if len(high_low_history) < period:
            return None
        tr_values = []
        for i in range(1, len(high_low_history)):
            high = high_low_history[i]
            low = high_low_history[i-1]
            tr = abs(high - low)
            tr_values.append(tr)
        if len(tr_values) < period:
            return None
        return round(sum(tr_values[-period:]) / period)  # Full period average

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

    def calculate_trend(self, price_history):
        if len(price_history) < 3:
            return 0
        return price_history[-1] - price_history[-3]  # 2-step trend

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
            if vw_mid_price:
                self.kelp_vw_mid_price_history.append(vw_mid_price)
                self.kelp_atr_history.append(vw_mid_price)
                if len(self.kelp_vw_mid_price_history) > 50:
                    self.kelp_vw_mid_price_history.pop(0)
                    self.kelp_atr_history.pop(0)

            atr_raw = self.calculate_atr(self.kelp_atr_history, self.atr_period)
            trend = self.calculate_trend(self.kelp_vw_mid_price_history)
            if atr_raw is None or vw_mid_price is None:
                trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp, "kelp_profit": self.kelp_profit, "resin_profit": self.resin_profit})
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data

            atr = max(atr_raw, 1)
            market_depth = len(order_depth.buy_orders) + len(order_depth.sell_orders)
            base_spread = max(3, atr) if abs(trend) < atr else max(5, atr * 2)  # Wider if trending
            if state.timestamp - self.last_trade_timestamp > 300 and base_spread > 3:
                base_spread = 3
                logger.print(f"KELP: Liquidity boost, spread reduced to {base_spread}")

            trade_size = min(self.base_trade_size, limit // 5)  # Very conservative
            adjustment = self.calculate_inventory_adjustment(pos, limit)
            bid_price = int(vw_mid_price - base_spread * adjustment)
            ask_price = int(vw_mid_price + base_spread * adjustment)

            self.update_profit(state, vw_mid_price=vw_mid_price)
            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > best_ask and abs(trend) < atr:  # Trade with trend check
                result["KELP"].append(Order("KELP", bid_price, buy_volume))
                logger.print(f"KELP: Bid at {bid_price} for {buy_volume}, ATR: {atr}, Trend: {trend}, Pos: {pos}")
                self.last_trade_timestamp = state.timestamp
            if sell_volume > 0 and ask_price < best_bid and abs(trend) < atr:
                result["KELP"].append(Order("KELP", ask_price, -sell_volume))
                logger.print(f"KELP: Ask at {ask_price} for {sell_volume}, ATR: {atr}, Trend: {trend}, Pos: {pos}")
                self.last_trade_timestamp = state.timestamp

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
            base_spread = max(3, sigma)  # Wider spread
            adjustment = self.calculate_inventory_adjustment(pos, limit)
            bid_price = int(fair_price - base_spread * adjustment)
            ask_price = int(fair_price + base_spread * adjustment)

            self.update_profit(state, fair_price=fair_price)
            trade_size = min(self.base_trade_size, limit // 5)
            available_to_buy = limit - pos
            available_to_sell = limit + pos
            buy_volume = min(trade_size, available_to_buy)
            sell_volume = min(trade_size, available_to_sell)
            if buy_volume > 0 and bid_price > best_ask:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", bid_price, buy_volume))
                logger.print(f"RESIN: Bid at {bid_price} for {buy_volume}, Fair: {fair_price}, Pos: {pos}")
            if sell_volume > 0 and ask_price < best_bid:
                result["RAINFOREST_RESIN"].append(Order("RAINFOREST_RESIN", ask_price, -sell_volume))
                logger.print(f"RESIN: Ask at {ask_price} for {sell_volume}, Fair: {fair_price}, Pos: {pos}")

        trader_data = json.dumps({"last_trade_timestamp": self.last_trade_timestamp, "kelp_profit": self.kelp_profit, "resin_profit": self.resin_profit})
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def toJSON(self, result: Any) -> str:
        return json.dumps(result, cls=ProsperityEncoder)