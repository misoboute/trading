# trading_bot/engine/signals.py

import indicators
import numpy as np
import pandas as pd
import datetime

class IndicatorReliabilityCalculator:
    def __init__(self):
        self.mkt_rev_finder = indicators.MarketReversalFinder(40, 5)
        self.indicator_collectors = [
            IndicatorCollection(datetime.timedelta(seconds=15)),
            IndicatorCollection(datetime.timedelta(seconds=30)),
            IndicatorCollection(datetime.timedelta(seconds=60)),
            # IndicatorCollection(datetime.timedelta(seconds=120)),
        ]
        self.mkt_rev_finder.add_new_trade_callback(self._on_new_hindsight_good_trade)
        self.prime_buy_trades = []
        self.prime_sell_trades = []
        self.backtesters = []
        self.simul_tester = None
        self.indicator_range_min = { 'wobi': 0, 'rsi': 0, 'stoch_rsi_d': 0, 'stoch_rsi_k': 0, 'volume': 0 }
        self.indicator_range_max = { 'wobi': 100, 'rsi': 100, 'stoch_rsi_d': 100, 'stoch_rsi_k': 100, 'volume': 1e9 }
        self.last_sim_id = 0

    def update_order_book(self, bids, asks, time):
        if not bids or not asks:
            return
        self.mkt_rev_finder.update_top_of_book(bids[0][0], asks[0][0], time)
        for ic in self.indicator_collectors:
            ic.update_order_book(bids, asks, time)

    def update_trades(self, trades):
        if not trades:
            return
        self.last_trade = trades[0]
        for ic in self.indicator_collectors:
            ic.update_trades(trades)

    def _on_new_hindsight_good_trade(self, trade):
        for ind_col in self.indicator_collectors:
            ind_col.add_potential_trade_point(trade['best_time'], trade['side'], trade['best_price'])
        if all([len(i.get_prime_buy_indicator_stats()) > 0 and len(i.get_prime_sell_indicator_stats()) > 0 
                for i in self.indicator_collectors]):
            best_backtester = max(self.backtesters, key=lambda bt: bt.get_pnl()) if self.backtesters else None
            if best_backtester:
                print(f"Best backtester PnL: {best_backtester.get_pnl() * 10000:.2f}bps")
                print(f"Buy triggers: {best_backtester.get_buy_triggers()}")
                print(f"Sell triggers: {best_backtester.get_sell_triggers()}")
                print(f"Num candles: {best_backtester.get_num_candles()}")
                print(f"Num trades: {best_backtester.get_num_trades()}")
                if self.simul_tester:
                    bal_quote, bal_base = self.simul_tester.get_balances()
                    print(f"All balances in quote: {self.simul_tester.get_all_balances_in_quote():.8f} | "
                        f"Quote: {bal_quote:.8f}, Base: {bal_base:.8f}, PnL: {self.simul_tester.get_pnl() * 10000:.2f}bps")
                    self.simul_tester.clear_callbacks()
                else:
                    print("No simulation tester yet, creating new one")
                    bal_quote, bal_base = 1.0, 0.0
                self.last_sim_id += 1
                self.simul_tester = TradeBacktester(
                    best_backtester.get_buy_triggers(), best_backtester.get_sell_triggers(), 
                    bal_quote, bal_base, print_trades=True, sim_id=self.last_sim_id)
            self._recreate_backtesters()
            for ind_col in self.indicator_collectors:
                ind_col.reset_prime_indicator_stats()
        return

    def _clear_backtesters(self):
        for bt in self.backtesters:
            bt.clear_callbacks()
        self.backtesters.clear()

    def _recreate_backtesters(self):
        self._clear_backtesters()
        ind_min_max_by_side = {
            'buy': { 'wobi': (0, None), 'rsi': (0, None), 'stoch_rsi_d': (0, None), 'stoch_rsi_k': (0, None), 'volume': (0, None) },
            'sell': { 'wobi': (None, 100), 'rsi': (None, 100), 'stoch_rsi_d': (None, 100), 'stoch_rsi_k': (None, 100), 'volume': (None, 1e9) }
        }

        def make_trigger(ind_coll, ind_name, beta, stats, side):
            radius = beta * stats['stdev']
            ind_mean = stats['mean']
            lo = ind_min_max_by_side[side][ind_name][0]
            hi = ind_min_max_by_side[side][ind_name][1]
            if lo is None:
                lo = max(ind_mean - radius, self.indicator_range_min[ind_name])
            if hi is None:
                hi = min(ind_mean + radius, self.indicator_range_max[ind_name])
            return { 'ind_coll': ind_coll, 'ind_name': ind_name, 'lo': lo, 'hi': hi, 'beta': beta }

        buy_ind_names = ['wobi', 'rsi', 'stoch_rsi_d', 'stoch_rsi_k', 'volume']
        sell_ind_names = ['wobi', 'rsi', 'stoch_rsi_d', 'stoch_rsi_k', 'volume']
        for buy_ind_coll1 in self.indicator_collectors:
            for buy_ind_name1 in buy_ind_names:
                buy_ind1_stats = buy_ind_coll1.get_prime_buy_indicator_stats()[buy_ind_name1]
                if buy_ind1_stats['stdev'] is None or buy_ind1_stats['stdev'] == 0:
                    continue
                for buy_ind_coll2 in self.indicator_collectors:
                    for buy_ind_name2 in buy_ind_names:
                        buy_ind2_stats = buy_ind_coll2.get_prime_buy_indicator_stats()[buy_ind_name2]
                        if (buy_ind_coll1 == buy_ind_coll2) and (buy_ind_name1 == buy_ind_name2):
                            continue
                        if buy_ind2_stats['stdev'] is None or buy_ind2_stats['stdev'] == 0:
                            continue
                        for sell_ind_coll1 in self.indicator_collectors:
                            for sell_ind_name1 in sell_ind_names:
                                sell_ind1_stats = sell_ind_coll1.get_prime_sell_indicator_stats()[sell_ind_name1]
                                if sell_ind1_stats['stdev'] is None or sell_ind1_stats['stdev'] == 0:
                                    continue
                                for sell_ind_coll2 in self.indicator_collectors:
                                    for sell_ind_name2 in sell_ind_names:
                                        sell_ind2_stats = sell_ind_coll2.get_prime_sell_indicator_stats()[sell_ind_name2]
                                        if (sell_ind_coll1 == sell_ind_coll2) and (sell_ind_name1 == sell_ind_name2):
                                            continue
                                        if sell_ind2_stats['stdev'] is None or sell_ind2_stats['stdev'] == 0:
                                            continue
                                        # print(f"Indicator stats:", buy_ind1_stats, buy_ind2_stats, sell_ind1_stats, sell_ind2_stats)
                                        betas = [.0]
                                        for buy_ind1_beta in betas:
                                            buy_trig1 = make_trigger(buy_ind_coll1, buy_ind_name1, buy_ind1_beta, buy_ind1_stats, 'buy')
                                            for buy_ind2_beta in betas:
                                                buy_trig2 = make_trigger(buy_ind_coll2, buy_ind_name2, buy_ind2_beta, buy_ind2_stats, 'buy')
                                                for sell_ind1_beta in betas:
                                                    sell_trig1 = make_trigger(sell_ind_coll1, sell_ind_name1, sell_ind1_beta, sell_ind1_stats, 'sell')
                                                    for sell_ind2_beta in betas:
                                                        sell_trig2 = make_trigger(sell_ind_coll2, sell_ind_name2, sell_ind2_beta, sell_ind2_stats, 'sell')
                                                        buy_triggers = [ buy_trig1, buy_trig2 ]
                                                        sell_triggers = [ sell_trig1, sell_trig2 ]
                                                        bt = TradeBacktester(buy_triggers, sell_triggers)
                                                        self.backtesters.append(bt)
        print(f"Created {len(self.backtesters)} backtesters")

class IndicatorCollection:
    def __init__(self, candle_period: datetime.timedelta):
        self.candles = indicators.CandleMaker(candle_period)
        self.candles.add_new_candle_callback(self._on_new_candle)
        self.ob_imbalance = indicators.OrderBookImbalance()
        self.ob_imbalance_avg = indicators.IncrementalAverageCalculator()
        self.weighted_imbalance = indicators.OrderBookWeightedImbalance()
        self.weighted_imbalance_avg = indicators.IncrementalAverageCalculator()
        self.rsi_calc = indicators.IncrementalRSI(14)
        self.stoch_rsi_calc = indicators.IncrementalStochasticOscillator(14, 3)
        self.top_of_book = ()
        self.trade_volume = 0.0
        self.candles_list = []
        self.num_primes_for_stats = 3
        self.reset_prime_indicator_stats()
        self.new_candle_callbacks = {}
        self.last_cb_sub_id = 0

    def update_order_book(self, bids, asks, time):
        if not bids or not asks:
            return
        self.top_of_book = (bids[0][0], asks[0][0])
        self.ob_imbalance.update_order_book(bids, asks)
        self.weighted_imbalance.update_order_book(bids, asks)
        self.ob_imbalance_avg.push(self.ob_imbalance.get_ratios()[0])
        self.weighted_imbalance_avg.push(self.weighted_imbalance.get_ratios()[0])

    def update_trades(self, trades):
        if not trades:
            return
        self.last_trade = trades[0]
        for trade in trades[::-1]:
            self.candles.push(trade['time'], trade['price'])
            self.trade_volume += trade['size']

    def add_potential_trade_point(self, time, side, price):
        if not self.candles_list:
            return
        for candle in self.candles_list[-1::-1]:
            if candle['close_time'] <= time:
                if side == 'BUY':
                    candle['prime_buy'] = price
                    self.prime_buy_indicator_snapshots.append(candle)
                else:
                    candle['prime_sell'] = price
                    self.prime_sell_indicator_snapshots.append(candle)
                break
        for snaps, stats in [(self.prime_sell_indicator_snapshots, self.prime_sell_ind_stats),
            (self.prime_buy_indicator_snapshots, self.prime_buy_ind_stats)]:
            self._update_prime_indicator_stats(snaps, stats)

    def get_prime_buy_indicator_stats(self):
        return self.prime_buy_ind_stats

    def get_prime_sell_indicator_stats(self):
        return self.prime_sell_ind_stats

    def add_new_candle_callback(self, callback):
        self.last_cb_sub_id += 1
        self.new_candle_callbacks[self.last_cb_sub_id] = callback
        return self.last_cb_sub_id

    def remove_new_candle_callback(self, callback_id):
        if callback_id in self.new_candle_callbacks:
            del self.new_candle_callbacks[callback_id]

    def reset_prime_indicator_stats(self):
        self.prime_buy_indicator_snapshots = []
        self.prime_sell_indicator_snapshots = []
        self.prime_buy_ind_stats = {}
        self.prime_sell_ind_stats = {}

    def _update_prime_indicator_stats(self, snaps, stats):
        if not snaps or len(snaps) < self.num_primes_for_stats:
            return
        for indicator in ['wobi', 'rsi', 'stoch_rsi_d', 'stoch_rsi_k', 'volume']:
            ind_stats = stats[indicator] = {}
            snap_series = pd.Series([c[indicator] for c in snaps if c[indicator] is not None])
            ind_stats['mean'] = float(np.mean(snap_series)) if not snap_series.empty else None
            ind_stats['stdev'] = float(np.std(snap_series)) if not snap_series.empty else None
            ind_stats['min'] = float(np.min(snap_series)) if not snap_series.empty else None
            ind_stats['max'] = float(np.max(snap_series)) if not snap_series.empty else None
            ind_stats['range'] = ind_stats['max'] - ind_stats['min'] if ind_stats['max'] is not None and ind_stats['min'] is not None else None

    def get_ind_names(self):
        return ['wobi', 'rsi', 'stoch_rsi_d', 'stoch_rsi_k', 'volume']

    def get_enriched_candles(self):
        return pd.DataFrame(self.candles_list).set_index('close_time')

    def __repr__(self):
        return f"IndicatorCollection(candle_period={self.candles.get_period_length()})"

    def _on_new_candle(self, candle: dict):
        self.rsi_calc.push(candle['close'])
        rsi = self.rsi_calc.get_rsi()
        candle['rsi'] = rsi
        if rsi is not None:
            self.stoch_rsi_calc.push(rsi, rsi, rsi)
        candle['stoch_rsi_d'] = self.stoch_rsi_calc.get_percent_d()
        candle['stoch_rsi_k'] = self.stoch_rsi_calc.get_percent_k()
        candle['wobi'] = self.weighted_imbalance_avg.get_average()
        candle['obi'] = self.ob_imbalance_avg.get_average()
        candle['volume'] = self.trade_volume
        candle['best_bid'] = self.top_of_book[0]
        candle['best_ask'] = self.top_of_book[1]
        self.trade_volume = 0.0
        self.candles_list.append(candle)
        self.weighted_imbalance_avg.reset()
        self.ob_imbalance_avg.reset()
        for callback in self.new_candle_callbacks.values():
            callback(candle)

    def _on_new_hindsight_good_trade(self, trade):
        prime_trade = {k: trade[k] for k in ['start_time', 'end_time', 'best_time', 'side']}
        if trade['side'] == 'BUY':
            prime_trade['prime_price'] = trade['min_price']
            prime_trade['max_price'] = trade['max_price']
            self.prime_buy_trades.append(trade)
        else:
            prime_trade['prime_price'] = trade['max_price']
            prime_trade['min_price'] = trade['min_price']
            self.prime_sell_trades.append(trade)


class TradeBacktester:
    def __init__(self, buy_triggers, sell_triggers, init_bal_quote=1.0, init_bal_base=0.0, print_trades=False,
                 sim_id=255):
        # For ETH/BTC, base asset is ETH, quote asset is BTC, so buy means buy ETH with BTC, sell means sell ETH for BTC
        # price is how much you pay in quote asset (ETH) to buy 1 unit of base asset (BTC)
        self.buy_triggers = buy_triggers
        self.sell_triggers = sell_triggers
        self.print_trades = print_trades
        self.evaluated_buy_triggers = [False for _ in buy_triggers]
        self.evaluated_sell_triggers = [False for _ in sell_triggers]
        self.init_bal_quote = self.bal_quote = init_bal_quote
        self.init_bal_base = self.bal_base = init_bal_base
        self.num_trig_candles = 0
        self.num_trades = 0
        self.exchange_fee = 0.0005
        self.pnl = 0.0
        self.last_candle = None
        self.cb_subs = []
        self.sim_id = sim_id
        self._subscribe_to_ind_colls_for_side('buy', buy_triggers, self.evaluated_buy_triggers)
        self._subscribe_to_ind_colls_for_side('sell', sell_triggers, self.evaluated_sell_triggers)

    def clear_callbacks(self):
        for ind_coll, cb_id in self.cb_subs:
            ind_coll.remove_new_candle_callback(cb_id)
        self.cb_subs.clear()

    def get_buy_triggers(self):
        return self.buy_triggers

    def get_sell_triggers(self):
        return self.sell_triggers

    def get_pnl(self):
        return self.pnl

    def get_balances(self):
        return self.bal_quote, self.bal_base

    def get_all_balances_in_quote(self):
        if self.bal_base > 0:
            return self.bal_quote + (self.bal_base * self.last_candle['best_bid'] if self.last_candle else 0.0)
        return self.bal_quote

    def get_num_trades(self):
        return self.num_trades

    def get_num_candles(self):
        return self.num_trig_candles / (len(self.buy_triggers) + len(self.sell_triggers))

    def _do_trade(self, candle: dict, side: str):
        best_bid, best_ask = candle['best_bid'], candle['best_ask']
        if side == 'buy' and self.bal_quote > 0:
            self.bal_base = self.bal_quote / best_ask * (1 - self.exchange_fee)
            self.bal_quote = 0.0
            if self.print_trades:
                print(f"ID {self.sim_id} | Time {candle['close_time']}: BUY @ {best_ask:.8f} | Base: {self.bal_base:.8f}, Quote: {self.bal_quote:.8f}")
        elif side == 'sell' and self.bal_base > 0:
            self.bal_quote = self.bal_base * best_bid * (1 - self.exchange_fee)
            self.bal_base = 0.0
            if self.print_trades:
                print(f"ID {self.sim_id} | Time {candle['close_time']}: SELL @ {best_bid:.8f} | Base: {self.bal_base:.8f}, Quote: {self.bal_quote:.8f}")
        else:
            return
        all_bal_quote = self.bal_base * best_bid if self.bal_base > 0 else self.bal_quote
        all_bal_base = self.bal_quote / best_ask if self.bal_quote > 0 else self.bal_base
        self.pnl = all_bal_quote / self.init_bal_quote if self.init_bal_quote > 0 else (
            all_bal_base / self.init_bal_base if self.init_bal_base > 0 else 0.0)
        self.pnl -= 1.0
        self.num_trades += 1

    def _subscribe_to_ind_colls_for_side(self, side, trig_list, eval_trig_list):
        for i, trig in enumerate(trig_list):
            ind_coll = trig['ind_coll']
            new_trig_candle_cb = lambda c, i=i, trig=trig: self._eval_new_trig_candle(
                c, side, eval_trig_list, i, trig['ind_name'], trig['lo'], trig['hi'])
            self.cb_subs.append((ind_coll, ind_coll.add_new_candle_callback(new_trig_candle_cb)))

    def _eval_new_trig_candle(self, candle, side, eval_trig_list, trig_idx, ind_name, lo, hi):
        ind_val = candle.get(ind_name)
        if ind_val is None:
            return
        eval_trig_list[trig_idx] = ind_val >= lo and ind_val <= hi
        if all(eval_trig_list):
            self._do_trade(candle, side)
            for i in range(len(eval_trig_list)):
                eval_trig_list[i] = False
        self.num_trig_candles += 1
        self.last_candle = candle

class SignalEngine:
    def __init__(self):
        self.top_of_book = None
        self.last_trade = None
        self.mkt_rev_finder = indicators.MarketReversalFinder()
        self.mkt_rev_finder.add_new_trade_callback(self._on_new_hindsight_good_trade)
        self.ob_imbalance = indicators.OrderBookImbalance()
        self.weighted_imbalance = indicators.OrderBookWeightedImbalance()
        self.weighted_imbalance_sma = indicators.IncrementalSimpleMovingAverage(1)
        self.candles = indicators.CandleMaker(datetime.timedelta(minutes=1))
        self.candles.add_new_candle_callback(self._on_new_candle)
        self.rsi_calc = indicators.IncrementalRSI(14)
        self.stoch_rsi_calc = indicators.IncrementalStochasticOscillator(14, 3)
        self.signal_callbacks = []
        self.last_signal = None
        hindsight_sma_length = 2
        self.hindsight_trade_ind_sma = { 
            'BUY' : {
                'weighted_imbalance': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length),
                'rsi': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length),
                'stoch_rsi_d': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length),
                'stoch_rsi_k': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length) },
            'SELL' : {
                'weighted_imbalance': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length),
                'rsi': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length),
                'stoch_rsi_d': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length),
                'stoch_rsi_k': indicators.IncrementalSimpleMovingAverage(hindsight_sma_length) }
            }
        self.indicators_map = {'weighted_imbalance': None,
            'rsi': None,
            'stoch_rsi_d': None,
            'stoch_rsi_k': None
        }

    def add_signal_callback(self, callback):
        self.signal_callbacks.append(callback)

    def get_last_trade_price(self):
        self.last_trade.get('price') if self.last_trade else None

    def get_last_bid_ask(self):
        if self.top_of_book:
            return self.top_of_book
        return None

    def update_order_book(self, bids, asks, time):
        if not bids or not asks:
            return
        self.top_of_book = (float(bids[0][0]), float(asks[0][0]), time)
        self.weighted_imbalance.update_order_book(bids, asks)
        self.ob_imbalance.update_order_book(bids, asks)
        self.weighted_imbalance_sma.push(self.weighted_imbalance.get_ratios()[0])
        self.indicators_map['weighted_imbalance'] = self.weighted_imbalance_sma.get_sma()
        self.mkt_rev_finder.update_top_of_book(*self.top_of_book, self.indicators_map.copy())
        self.generate_signal()

    def update_trades(self, trades):
        if not trades:
            return
        self.last_trade = trades[0]
        for trade in trades[::-1]:
            self.candles.push(trade['time'], trade['price'])
        self.indicators_map['rsi'] = self.rsi_calc.get_rsi()
        self.indicators_map['stoch_rsi_d'] = self.stoch_rsi_calc.get_percent_d()
        self.indicators_map['stoch_rsi_k'] = self.stoch_rsi_calc.get_percent_k()
        self.generate_signal()

    def generate_signal_3(self):
        if any([i is None for i in self.indicators_map.values()]):
            return
        next_signal = 'SELL' if self.last_signal == 'BUY' else 'BUY' if self.last_signal == 'SELL' else None
        cur_wi = self.indicators_map['weighted_imbalance']
        cur_srk = self.indicators_map['stoch_rsi_k']
        if not next_signal:
            next_signal = 'SELL' if cur_wi < 0.5 else 'BUY'
        num_hindsight_trades = 6
        if len(self.mkt_rev_finder.get_trade_history()) < num_hindsight_trades:
            return
        hindsight = [
            t['indicators'] for t in self.mkt_rev_finder.get_trade_history()[-num_hindsight_trades:] if t['side'] == next_signal]
        if any([i is None for i in hindsight]):
            return
        if next_signal == 'BUY':
            cmp_weighted_imbalance = max([i['weighted_imbalance'] for i in hindsight])
            cmp_stoch_rsi_k = max([i['stoch_rsi_k'] for i in hindsight])
            do_generate = (cur_wi >= cmp_weighted_imbalance and cur_srk >= cmp_stoch_rsi_k)
        else:
            cmp_weighted_imbalance = min([i['weighted_imbalance'] for i in hindsight])
            cmp_stoch_rsi_k = min([i['stoch_rsi_k'] for i in hindsight])
            do_generate = (cur_wi <= cmp_weighted_imbalance and cur_srk <= cmp_stoch_rsi_k)
        if do_generate:
            price = self.top_of_book[0] if next_signal == 'SELL' else self.top_of_book[1]
            time = self.top_of_book[2].isoformat()
            print(f"[{time}] {next_signal} @ {price} | "
                f"WI: {cur_wi} vs {cmp_weighted_imbalance} | K: {cur_srk} vs {cmp_stoch_rsi_k}")
            self.last_signal = next_signal
            for cb in self.signal_callbacks:
                cb(self.last_signal)
        
    def generate_signal(self):
        if any([i is None for i in self.indicators_map.values()]):
            return
        next_signal = 'SELL' if self.last_signal == 'BUY' else 'BUY' if self.last_signal == 'SELL' else None
        if not next_signal:
            next_signal = 'SELL' if self.indicators_map['weighted_imbalance'] < 0.5 else 'BUY'
        hindsight_inds = self.hindsight_trade_ind_sma[next_signal]
        if any([i.get_sma() is None for i in hindsight_inds.values()]):
            return
        do_generate = False
        cur_wi = self.indicators_map['weighted_imbalance']
        cur_srsi_k = self.indicators_map['stoch_rsi_k']
        sma_wi = hindsight_inds['weighted_imbalance'].get_sma()
        sma_srsi_k = hindsight_inds['stoch_rsi_k'].get_sma()
        if next_signal == 'BUY' and cur_wi <= sma_wi and cur_srsi_k >= sma_srsi_k:
            do_generate = True
        elif next_signal == 'SELL' and cur_wi <= sma_wi and cur_srsi_k <= sma_srsi_k:
            do_generate = True
        if do_generate:
            price = self.top_of_book[0] if next_signal == 'SELL' else self.top_of_book[1]
            time = self.top_of_book[2].isoformat()
            print(f"[{time}] {next_signal} @ {price} | "
                f"WI: {cur_wi} vs {sma_wi} | K: {cur_srsi_k} vs {sma_srsi_k}")
            self.last_signal = next_signal
            for cb in self.signal_callbacks:
                cb(self.last_signal)

    def generate_signal_1(self):
        for side, ind_map in self.hindsight_trade_ind_sma.items():
            std_dev = 0.1
            tgt_ind_list = [(ind, sma_calc.get_sma()) for ind, sma_calc in ind_map.items()]
            if all(
                val is not None and self.indicators_map[ind] is not None and
                abs(val - self.indicators_map[ind]) < std_dev * max(abs(val), 1e-9)
                for ind, val in tgt_ind_list
            ) and self.last_signal != side:
                self.last_signal = side
                price = self.top_of_book[0] if side == 'SELL' else self.top_of_book[1]
                time = self.top_of_book[2].isoformat()
                print(f"Signal at {time}: //// {side} @ {price} \\\\\\\\ ")
                for cb in self.signal_callbacks:
                    cb(self.last_signal)

    def _on_new_candle(self, candle: dict):
        self.rsi_calc.push(candle['close'])
        rsi = self.rsi_calc.get_rsi()
        if not rsi:
            return
        self.stoch_rsi_calc.push(rsi, rsi, rsi)

    def _on_new_hindsight_good_trade(self, trade):
        side = trade['side']
        for ind_name, ind_value in trade['indicators'].items():
            if ind_value is not None:
                self.hindsight_trade_ind_sma[side][ind_name].push(ind_value)
