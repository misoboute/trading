#!/usr/bin/env python3

# Run this script to listen to the Pionex spot market data for a single symbol name and record it to a file.

from datetime import datetime

import argparse
import json
import os
import sys
import asyncio
import websockets.asyncio.client as ws_client

class PionexMarketDataStream:
    def __init__(self, public_websocket_url="wss://ws.pionex.com/wsPub"):
        self.public_websocket_url = public_websocket_url
        self.symbols = []

    async def connect(self):
        print(f"Connecting to Pionex WebSocket at {self.public_websocket_url}...")
        self.websocket = await ws_client.connect(self.public_websocket_url)
        print("Connected to Pionex WebSocket.")

    def subscribe(self, symbol):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            subscription_message = {
                "op": "SUBSCRIBE",
                "topic": "TRADE",
                "symbol": symbol
            }
            asyncio.create_task(self.websocket.send(json.dumps(subscription_message)))

    async def receive_data(self):
        try:
            message = await self.websocket.recv()
            data = json.loads(message)
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Record Pionex spot market data for a single symbol.")
    parser.add_argument("symbols", type=str, default=['ETH_BTC'],
                        nargs='*', help="List of symbols to subscribe to (e.g., ETH_BTC BTC_USDT).")
    parser.add_argument("--output", type=str, default="pionex_data.json", help="Output fil+e to save the recorded data.")
    return parser.parse_args()

async def pionex_md_dump(symbols, output_file):
    pio_md = PionexMarketDataStream()
    await pio_md.connect()
    for symbol in symbols:
        pio_md.subscribe(symbol)
    print(f"Subscribed to symbols: {', '.join(symbols)}")
    try:
        while True:
            data = await pio_md.receive_data()
            if data:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(data) + '\n')
                print(f"Received data: {data}")
    except KeyboardInterrupt:
        print("Stopping data recording.")
    except Exception as e:
        print(f"An error occurred: {e}")
                

def main():
    args = parse_args()
    asyncio.run(pionex_md_dump(args.symbols, args.output))
    return 0

if __name__ == "__main__":
    sys.exit(main())