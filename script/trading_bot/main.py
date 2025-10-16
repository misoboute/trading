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
import logging

def identifierize_symbol(symbol: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in symbol)

def configure_logging(args):
    if not args.log_file:
        timestamp_str = datetime.now(timezone.utc).strftime('%y%m%d.%H%M%S')
        safe_symbol = identifierize_symbol(args.pair)
        args.log_file = os.path.join(os.getcwd(), f"{safe_symbol}.{args.mode}.{timestamp_str}.log")
    if not os.path.exists(os.path.dirname(args.log_file)):
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f %z')

async def record_market_data(symbol: str, filepath: str = None):
    if not filepath:
        timestamp_str = datetime.now(timezone.utc).strftime('%y%m%d.%H%M%S')
        safe_symbol = identifierize_symbol(symbol)
        filename = f"{safe_symbol}.{timestamp_str}.bin"
        filepath = os.path.join(os.getcwd(), filename)
    async with pionex.PionexMarketDataFeed() as feed:
        async with PionexMarketDataRecorder(symbol, filepath, feed) as recorder:
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
    parser.add_argument('--log-file', help="Log file path")
    args = parser.parse_args()

    configure_logging(args)

    if args.mode == 'backtest':
        asyncio.run(run_backtest(args.pair, args.start, args.end, args.file, args.capital_base_ccy, args.capital_quote_ccy))
    elif args.mode == 'simulate':
        asyncio.run(run_simulation(args.pair, args.capital_base_ccy, args.capital_quote_ccy))
    elif args.mode == 'trade':
        asyncio.run(run_live_trader(args.pair))
    elif args.mode == 'record':
        asyncio.run(record_market_data(args.pair, args.file))
