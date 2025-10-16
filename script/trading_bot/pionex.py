import asyncio
import gzip
import websockets
import json
import logging
from datetime import datetime, timezone

# Objects of this class can be passed to the SimulationExchange as the data source. We'll implement this later.
# The PionexMarketDataFeed class wraps the Pionex WebSocket API for market data.
class PionexMarketDataFeed:
    def __init__(self, public_websocket_url="wss://ws.pionex.com/wsPub"):
        self.public_websocket_url = public_websocket_url
        self.websocket = None
        self.polling_task = None
        self.depth_callbacks = {}
        self.trade_callbacks = {}
        self.message_queue = asyncio.Queue()

    async def __aenter__(self):
        logging.log(logging.INFO, "Entering PionexMarketDataFeed context")
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        logging.log(logging.INFO, "Exiting PionexMarketDataFeed context")
        await self.close()
        self.message_queue = asyncio.Queue()

    async def connect(self):
        logging.log(logging.INFO, "Connecting")
        self.websocket = await websockets.connect(self.public_websocket_url)
        self.polling_task = asyncio.create_task(self._poll_messages())
        logging.log(logging.INFO, "Connected; started polling messages")

    async def close(self):
        logging.log(logging.INFO, "Closing connection")
        if self.polling_task:
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def subscribe_trades(self, symbol: str, callback):
        if symbol in self.trade_callbacks:
            self.trade_callbacks[symbol].append(callback)
        else:
            await self.websocket.send(json.dumps({"op": "SUBSCRIBE", "topic": "TRADE", "symbol": symbol}))
            self.trade_callbacks[symbol] = [callback]

    async def unsubscribe_trades(self, symbol: str):
        await self.websocket.send(json.dumps({"op": "UNSUBSCRIBE", "topic": "TRADE", "symbol": symbol}))
        self.trade_callbacks.pop(symbol, None)

    async def subscribe_depth(self, symbol: str, limit: int, callback):
        if symbol in self.depth_callbacks:
            self.depth_callbacks[symbol].append(callback)
        else:
            await self.websocket.send(json.dumps({"op": "SUBSCRIBE", "topic": "DEPTH", "symbol": symbol, "limit": limit}))
            self.depth_callbacks[symbol] = [callback]

    async def unsubscribe_depth(self, symbol: str):
        await self.websocket.send(json.dumps({"op": "UNSUBSCRIBE", "topic": "DEPTH", "symbol": symbol}))
        self.depth_callbacks.pop(symbol, None)

    async def _poll_messages(self):
        while True:
            try:
                message = await self.websocket.recv()
                await self.message_queue.put(message)
            except Exception as e:
                logging.log(logging.ERROR, f"Polling error: {e}")
                break
            await asyncio.sleep(0.1)

    async def process_messages(self):
        if not self.polling_task:
            return
        if self.polling_task.done():
            if self.polling_task.cancelled():
                raise asyncio.CancelledError()
            if self.polling_task.exception():
                logging.log(logging.ERROR, "Polling task ended with an exception.")
                raise self.polling_task.exception()
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            data = json.loads(message)
            topic = data.get("topic")
            symbol = data.get("symbol")
            type = data.get("type")

            if data.get("op") == "PING":
                await self.websocket.send(json.dumps({"op": "PONG", "timestamp": data.get("timestamp")}))
            elif type in ["SUBSCRIBED", "UNSUBSCRIBED"]:
                logging.log(logging.INFO, f"Received {type} confirmation for {topic} on {symbol}")
            elif topic == "TRADE" and symbol in self.trade_callbacks:
                trades = [{
                    'time': datetime.fromtimestamp(trade["timestamp"] / 1000.0, tz=timezone.utc),
                    'price': float(trade["price"]),
                    'size': float(trade["size"]),
                    'side': trade["side"]} for trade in data.get("data", [])]
                for cb in self.trade_callbacks[symbol]:
                    cb(trades)
            elif topic == "DEPTH" and symbol in self.depth_callbacks:
                orders = data.get("data", {})
                bids = [(float(b[0]), float(b[1])) for b in orders['bids']]
                asks = [(float(a[0]), float(a[1])) for a in orders['asks']]
                for cb in self.depth_callbacks[symbol]:
                    cb(bids, asks, datetime.fromtimestamp(data["timestamp"] / 1000.0, tz=timezone.utc))
                continue
            else:
                logging.log(logging.WARN, f"Unhandled message: {data}") 
