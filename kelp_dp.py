# Final Enhanced IMC Prosperity Strategy – Full Length
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Trade
from collections import defaultdict, deque

# Maintain historical price data across calls
price_history = defaultdict(lambda: deque(maxlen=100))

# Position limits per product
POSITION_LIMITS = {
    "PEARLS": 100,
    "BANANAS": 100,
    # Add more products if available
}

# Fair price estimation from rolling trade history
def fair_price_from_trades(trades: List[Trade], product: str) -> float:
    for trade in trades:
        price_history[product].append(trade.price)
    if price_history[product]:
        return sum(price_history[product]) / len(price_history[product])
    return None

# Logging utility for debug
def log(msg: str):
    print("[LOG]", msg)

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            current_position = state.position.get(product, 0)
            trades = state.market_trades.get(product, [])
            orders: List[Order] = []

            # Logging state
            log(f"Processing {product} | Position: {current_position}")

            # Estimate fair price
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            fair_price = fair_price_from_trades(trades, product)
            if not fair_price and best_bid and best_ask:
                fair_price = (best_bid + best_ask) / 2
            elif not fair_price:
                log(f"Insufficient data for {product} – skipping")
                continue

            log(f"{product} Fair Price (adjusted): {fair_price:.2f}")

            # Bias based on inventory to prevent overloading
            inventory_bias = 0.05 * current_position
            adjusted_fair_price = fair_price - inventory_bias

            # Define bid/ask spread and order volumes
            spread = 1
            buy_price = int(adjusted_fair_price - spread)
            sell_price = int(adjusted_fair_price + spread)
            order_volume = min(20, max(5, int(abs(sell_price - buy_price))))

            log(f"{product} → Buy @ {buy_price} | Sell @ {sell_price} | Volume: {order_volume}")

            # Place buy orders if under position limit
            if current_position < POSITION_LIMITS[product]:
                orders.append(Order(product, buy_price, +order_volume))
            # Place sell orders if over negative limit
            if current_position > -POSITION_LIMITS[product]:
                orders.append(Order(product, sell_price, -order_volume))

            result[product] = orders

        return result
