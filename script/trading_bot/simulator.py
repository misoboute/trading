# trading_bot/engine/simulator.py

import asyncio
import websockets
import json
from pionex import PionexMarketDataFeed
from datetime import datetime, timedelta
import pandas as pd
from signals import SignalEngine

class SimulationExchange:
    def __init__(self, fee_rate: float = 0.0005, simulate_slippage: bool = False, data_source=None):
        self.fee_rate = fee_rate
        self.simulate_slippage = simulate_slippage
        self.balances = {}
        self.orders = {}
        self.order_id_counter = 1
        self.data_source = data_source
        self.top_of_book = { }

    async def place_order(self, side: str, symbol: str, qty: float, type: str = 'MARKET', price: float = None) -> int:
        bid, ask = await self._get_top_of_book(symbol)
        market_price = ask if side == 'BUY' else bid
        executed_price = price if type == 'LIMIT' and price else market_price

        if self.simulate_slippage:
            slippage_factor = 1.0005 if side == 'BUY' else 0.9995
            executed_price *= slippage_factor

        base_asset, quote_asset = symbol.split("_")
        cost = executed_price * qty
        fee = cost * self.fee_rate

        if side == 'BUY':
            if self.balances.get(quote_asset, 0) < cost + fee:
                raise Exception("Insufficient balance")
            self.balances[quote_asset] -= (cost + fee)
            self.balances[base_asset] = self.balances.get(base_asset, 0) + qty
        else:  # sell
            if self.balances.get(base_asset, 0) < qty:
                raise Exception("Insufficient balance")
            self.balances[base_asset] -= qty
            self.balances[quote_asset] = self.balances.get(quote_asset, 0) + (cost - fee)

        order_id = self.order_id_counter
        self.order_id_counter += 1
        self.orders[order_id] = {
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': executed_price,
            'status': 'FILLED'
        }
        print(f"Order placed: {side} {qty} {symbol} @ {executed_price:.8f} (Order ID: {order_id})")
        return order_id

    async def cancel_order(self, order_id: int) -> None:
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELED'

    async def get_order_status(self, order_id: int) -> dict:
        return self.orders.get(order_id, {})

    async def get_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)

    async def deposit(self, asset: str, amount: float) -> None:
        self.balances[asset] = self.balances.get(asset, 0.0) + amount

    async def get_total_balance(self) -> dict:
        return dict(self.balances)

    async def get_portfolio_value_in_quote(self, quote_asset: str) -> float:
        value = 0
        for asset, amount in self.balances.items():
            if asset == quote_asset:
                value += amount
                continue
            symbol = f"{asset}_{quote_asset}"
            bid, ask = await self._get_top_of_book(symbol)
            mid_price = (bid + ask) / 2
            value += amount * mid_price
        return value

    def _update_order_book(self, symbol: str, bids: list, asks: list, timestamp: int):
        self.top_of_book[symbol] = (float(bids[0][0]), float(asks[0][0]))

    async def _get_top_of_book(self, symbol: str) -> tuple:
        if symbol not in self.top_of_book:
            await self.data_source.subscribe_depth(
                symbol, limit=5, callback=lambda bids, asks, ts: self._update_order_book(symbol, bids, asks, ts))
            while symbol not in self.top_of_book:
                await self.data_source.process_messages()
                await asyncio.sleep(0.1)
        return self.top_of_book[symbol]

        
async def run_simulation(pair, init_cap_base_asset=0.17, init_cap_quote_asset=0.0017):
    async with PionexMarketDataFeed() as data_source:
        signal_engine = SignalEngine()
        exchange = SimulationExchange(data_source=data_source)
        await exchange.deposit('ETH', init_cap_base_asset)
        await exchange.deposit('BTC', init_cap_quote_asset)
        await data_source.subscribe_depth(symbol=pair, limit=5, callback=signal_engine.update_order_book)
        await data_source.subscribe_trades(symbol=pair, callback=signal_engine.update_trades)

        start_ts = datetime.utcnow()
        trade_log = []

        while True:
            await data_source.process_messages()
            signal, ob_momentum, stoch_rsi = signal_engine.generate_signal()

            last_bid, last_ask, timestamp = signal_engine.get_last_bid_ask() or (0.0, 0.0, 0)
            if not last_bid or not last_ask:
                await asyncio.sleep(0.1)
                continue

            portfolio_value = await exchange.get_portfolio_value_in_quote('BTC')
            if signal == 'BUY':
                target_quote = 0.2 * portfolio_value
                delta_quote = await exchange.get_balance('BTC') - target_quote
                if abs(delta_quote / target_quote) > 0.05:
                    print(f"Before buying portfolio Value: {portfolio_value:.8f} BTC")
                    qty = delta_quote / last_ask
                    await exchange.place_order('BUY', symbol=pair, qty=qty)
                    trade_log.append((datetime.utcnow(), 'BUY', last_ask, qty))

            elif signal == 'SELL':
                target_base = 0.2 * portfolio_value / last_bid
                delta_base = await exchange.get_balance('ETH') - target_base
                if abs(delta_base / target_base) > 0.05:
                    print(f"Before selling portfolio Value: {portfolio_value:.8f} BTC")
                    await exchange.place_order('SELL', symbol=pair, qty=delta_base)
                    trade_log.append((datetime.utcnow(), 'SELL', last_bid, delta_base))
                    

            if datetime.utcnow() - start_ts > timedelta(hours=24):
                break

            await asyncio.sleep(0.1)

        # Summary
        # final_value = quote_bal + base_bal * mid_price
        print(f"\n--- 24H SIMULATION COMPLETE ---")
        # print(f"Final Portfolio Value: {final_value:.8f} ({quote_bal:.8f} Q, {base_bal:.8f} B @ {mid_price:.8f})")
        for entry in trade_log:
            print(f"[{entry[0]}] {entry[1]} @ {entry[2]:.8f} | Qty: {entry[3]:.6f}")
