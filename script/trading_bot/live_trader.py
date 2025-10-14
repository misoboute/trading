# trading_bot/engine/live_trader.py

import asyncio
import websockets
import json
from datetime import datetime
import aiohttp
import pandas as pd
from signals import SignalEngine

API_KEY = 'YOUR_PIONEX_API_KEY'
API_SECRET = 'YOUR_PIONEX_SECRET_KEY'

# For simplicity, using market orders via REST
async def place_market_order(session, side, amount, pair):
    url = 'https://api.pionex.com/api/v1/order'
    payload = {
        "symbol": pair,
        "side": side.lower(),
        "type": "market",
        "quantity": amount,
    }
    headers = {
        "X-PIONEX-KEY": API_KEY,
        "Content-Type": "application/json",
    }
    async with session.post(url, json=payload, headers=headers) as resp:
        return await resp.json()

async def run_live_trader(pair):
    signal_engine = SignalEngine()
    base_bal = 0.0   # e.g., ETH
    quote_bal = 1000.0  # e.g., BTC equivalent

    uri = f"wss://ws.pionex.com/ws/{pair.lower()}@depth"
    print(f"Connecting to {uri} for live trading...")

    async with aiohttp.ClientSession() as session:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "event": "subscribe",
                "channel": "depth",
                "symbol": pair.lower()
            }))

            while True:
                message = await websocket.recv()
                data = json.loads(message)

                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if not bids or not asks:
                    continue

                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
                bid_volume = sum(float(bid[1]) for bid in bids[:5])
                ask_volume = sum(float(ask[1]) for ask in asks[:5])

                df = {'close': [mid_price]}
                signal, ob_momentum, stoch_rsi = signal_engine.generate_signal(
                    pd.DataFrame(df), bid_volume, ask_volume
                )

                portfolio_value = quote_bal + base_bal * mid_price

                if signal == 'BUY':
                    target_quote = 0.2 * portfolio_value
                    target_base = 0.8 * portfolio_value / mid_price
                    delta_quote = quote_bal - target_quote
                    if delta_quote > 0:
                        qty = delta_quote / mid_price
                        response = await place_market_order(session, 'BUY', qty, pair)
                        quote_bal -= delta_quote
                        base_bal += qty
                        print(f"[{datetime.utcnow()}] LIVE BUY SWITCH @ {mid_price:.8f} | Qty: {qty:.6f} | Resp: {response}")

                elif signal == 'SELL':
                    target_quote = 0.8 * portfolio_value
                    target_base = 0.2 * portfolio_value / mid_price
                    delta_base = base_bal - target_base
                    if delta_base > 0:
                        response = await place_market_order(session, 'SELL', delta_base, pair)
                        base_bal -= delta_base
                        quote_bal += delta_base * mid_price
                        print(f"[{datetime.utcnow()}] LIVE SELL SWITCH @ {mid_price:.8f} | Qty: {delta_base:.6f} | Resp: {response}")

                await asyncio.sleep(1)  # Control loop pace
