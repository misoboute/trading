# trading_bot/engine/backtester.py

import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from signals import SignalEngine
from pionex_rec import MarketDataPlayback
from simulator import SimulationExchange
from datetime import datetime, timezone, timedelta

async def run_backtest(pair, start, end, file_path, init_cap_base_asset=0.17, init_cap_quote_asset=0.0017):
    async with MarketDataPlayback(file_path) as data_source:
        signal_engine = SignalEngine()
        exchange = SimulationExchange(data_source=data_source)
        await exchange.deposit('ETH', init_cap_base_asset)
        await exchange.deposit('BTC', init_cap_quote_asset)
        data_source.subscribe_depth(callback=signal_engine.update_order_book)
        data_source.subscribe_trades(callback=signal_engine.update_trades)

        start_ts = datetime.now(timezone.utc)
        trade_log = []

        await data_source.play()
        while False:
            signal, ob_momentum, stoch_rsi = signal_engine.generate_signal()

            last_bid, last_ask, timestamp = signal_engine.get_last_bid_ask() or (0.0, 0.0, 0)
            if not last_bid or not last_ask:
                await asyncio.sleep(0)  # yield control to event loop
                continue

            now = datetime.fromtimestamp(timestamp / 1000.0)
            if not start_ts:
                start_ts = now
            portfolio_value = await exchange.get_portfolio_value_in_quote('BTC')
            if signal == 'BUY':
                target_quote = 0.2 * portfolio_value
                delta_quote = await exchange.get_balance('BTC') - target_quote
                if abs(delta_quote / target_quote) > 0.05:
                    print(f"Before buying portfolio Value: {portfolio_value:.8f} BTC")
                    qty = delta_quote / last_ask
                    await exchange.place_order('BUY', symbol=pair, qty=qty)
                    trade_log.append((now, 'BUY', last_ask, qty))

            elif signal == 'SELL':
                target_base = 0.2 * portfolio_value / last_bid
                delta_base = await exchange.get_balance('ETH') - target_base
                if abs(delta_base / target_base) > 0.05:
                    print(f"Before selling portfolio Value: {portfolio_value:.8f} BTC")
                    await exchange.place_order('SELL', symbol=pair, qty=delta_base)
                    trade_log.append((now, 'SELL', last_bid, delta_base))
                    

            if now - start_ts > timedelta(hours=24):
                break

        # Summary
        # final_value = quote_bal + base_bal * mid_price
        print(f"\n--- 24H SIMULATION COMPLETE ---")
        # print(f"Final Portfolio Value: {final_value:.8f} ({quote_bal:.8f} Q, {base_bal:.8f} B @ {mid_price:.8f})")
        for entry in trade_log:
            print(f"[{entry[0]}] {entry[1]} @ {entry[2]:.8f} | Qty: {entry[3]:.6f}")

    # plt.figure(figsize=(12,6))
    # plt.plot(timestamps, portfolio, label='Portfolio Value')
    # plt.title(f"Backtest on {pair} | Capital: €{initial_capital}")
    # plt.xlabel("Time")
    # plt.ylabel("Portfolio (€)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
