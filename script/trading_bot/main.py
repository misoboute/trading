#!/usr/bin/env python3
# trading_bot/main.py

import asyncio
import os
from datetime import datetime, timezone
from backtester import run_backtest
from simulator import run_simulation
from live_trader import run_live_trader
import pionex
from pionex_rec import PionexMarketDataRecorder

def identifierize_symbol(symbol: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in symbol)

async def record_market_data(symbol: str, filepath: str = None):
    if not filepath:
        timestamp_str = datetime.now(timezone.utc).strftime('%y%m%d.%H%M%S')
        safe_symbol = identifierize_symbol(symbol)
        filename = f"{safe_symbol}.{timestamp_str}.bin"
        filepath = os.path.join(os.getcwd(), filename)
    async with pionex.PionexMarketDataFeed() as feed, PionexMarketDataRecorder(symbol, filepath, feed) as recorder:
        try:
            while True:
                await feed.process_messages()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass 


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="ETHBTC Trading Bot")
    parser.add_argument('--mode', choices=['backtest', 'simulate', 'trade', 'record'], required=True, help="Run mode")
    parser.add_argument('--pair', default='ETH_BTC', help="Trading pair")
    parser.add_argument('--start', help="Start datetime for backtest")
    parser.add_argument('--end', help="End datetime for backtest")
    parser.add_argument('--file', help="Recording file for recording or backtest")
    parser.add_argument('--capital-quote-ccy', type=float, default=0.00174,
                        help="Initial capital in quote currency for backtest or simulation")
    parser.add_argument('--capital-base-ccy', type=float, default=0.171,
                        help="Initial capital in base currency for simulation")
    args = parser.parse_args()

    if args.mode == 'backtest':
        asyncio.run(run_backtest(args.pair, args.start, args.end, args.file, args.capital_base_ccy, args.capital_quote_ccy))
    elif args.mode == 'simulate':
        asyncio.run(run_simulation(args.pair, args.capital_base_ccy, args.capital_quote_ccy))
    elif args.mode == 'trade':
        asyncio.run(run_live_trader(args.pair))
    elif args.mode == 'record':
        asyncio.run(record_market_data(args.pair, args.file))
