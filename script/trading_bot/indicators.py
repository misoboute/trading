from collections import deque
import datetime
import pandas as pd

class IncrementalAverageCalculator:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    def push(self, value: float):
        self.count += 1
        self.total += value

    def get_average(self):
        if self.count == 0:
            return None
        return self.total / self.count

    def reset(self):
        self.count = 0
        self.total = 0.0

class IncrementalSimpleMovingAverage:
    def __init__(self, period: int):
        """
        Incremental SMA calculator.

        Args:
            period (int): Number of periods for SMA.
        """
        self.period = period
        self.values = deque(maxlen=period)
        self.sum = 0.0
        self.sma = None

    def push(self, value: float):
        """
        Add a new value to update the SMA.
        """
        if len(self.values) == self.period:
            # Remove oldest value from sum when deque is full
            self.sum -= self.values[0]
        self.values.append(value)
        self.sum += value

        if len(self.values) == self.period:
            self.sma = self.sum / self.period
        else:
            self.sma = self.sum / len(self.values)

    def get_sma(self):
        """
        Get the current SMA value.
        """
        return self.sma


class IncrementalExponentialMovingAverage:
    def __init__(self, period: int, smoothing: float = 2.0):
        """
        Incremental EMA calculator.

        Args:
            period (int): Number of periods for EMA.
            smoothing (float): Smoothing factor, typically 2.0 for standard EMA.
        """
        self.period = period
        self.smoothing = smoothing
        self.multiplier = smoothing / (period + 1)
        self.ema = None

    def push(self, value: float):
        """
        Add a new value to update the EMA.
        """
        if self.ema is None:
            # First data point: use it as the initial EMA
            self.ema = value
        else:
            self.ema += self.multiplier * (value - self.ema)

    def get_ema(self):
        """
        Get the current EMA value.
        """
        return self.ema

class IncrementalRSI:
    def __init__(self, period: int):
        self.period = period
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.prev_price = None
        self.rsi = None

    def push(self, price: float):
        if self.prev_price is None:
            self.prev_price = price
            return

        change = price - self.prev_price
        self.prev_price = price

        self.gains.append(max(change, 0))
        self.losses.append(-min(change, 0))

        if len(self.gains) < self.period:
            return

        avg_gain = sum(self.gains) / self.period
        avg_loss = sum(self.losses) / self.period

        if avg_loss == 0:
            self.rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            self.rsi = 100 - (100 / (1 + rs))
        # print("RSI: ", self.rsi)
        
    def get_rsi(self):
        return self.rsi

class IncrementalStochasticOscillator:
    def __init__(self, period: int, smooth_k: int):
        self.period = period
        self.smooth_k = smooth_k
        self.highs = deque(maxlen=period)
        self.lows = deque(maxlen=period)
        self.percent_k_s = deque(maxlen=smooth_k)
        self.percent_k = None
        self.percent_d = None

    def push(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) < self.period:
            return

        lowest_low = min(self.lows)
        highest_high = max(self.highs)
        if highest_high > lowest_low:
            self.percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            self.percent_k_s.append(self.percent_k)

        if len(self.percent_k_s) < self.smooth_k:
            return

        self.percent_d = sum(self.percent_k_s) / self.smooth_k

    def get_percent_k(self):
        return self.percent_k

    def get_percent_d(self):
        return self.percent_d

# MarketReversalFinder finds the optimal sell/buy points in historical data. I'm going to use it in parallel with
# other indicators to try and find a correlation between the optimal trade points and the indicators.
# The class will look like this:
class MarketReversalFinder:
    # low_high_spread: The minimum amount, in bps, of price spread between a trade point and the opposite-side trade
    # point that follows immediately after. For example, if this is set to 20, if the algorithm decides that a buy
    # trade should have been made at time t1 (because, say, the price is at a local minimum), then the next trade must
    # be a sell trade at a point (t2) where the price is at least 20bps higher than at t1 and has started falling down
    # at least 20bps lower at t3, and so on.
    def __init__(self, low_high_min_spread_bps: float = 20, optimal_price_tolerance_bps: float = 3):
        if optimal_price_tolerance_bps * 2 >= low_high_min_spread_bps:
            raise RuntimeError(
                f'Optimal price tolerance ({optimal_price_tolerance_bps}) too big '
                f'w.r.t low/high spread ({low_high_min_spread_bps})')
        self.low_high_spread = low_high_min_spread_bps / 10000
        self.optimal_price_tolerance = optimal_price_tolerance_bps / 10000
        self.history = []
        self.bids = []
        self.asks = []
        self.last_optimal_trade = None
        self.new_trade_callbacks = []

    def add_new_trade_callback(self, callback):
        self.new_trade_callbacks.append(callback)

    # Takes the latest TOB and the internally kept history of buy/sell data. It uses bid prices for selling and ask
    # prices for buying. It prints the complete buy/sell history data whenever the history is changed.
    def update_top_of_book(self, bid: float, ask: float, timestamp: datetime.datetime, indicators=None):
        if not self.last_optimal_trade:
            self.last_optimal_trade = [timestamp, ask, 'BUY', indicators]
            return

        self.asks.append([timestamp, ask])
        self.bids.append([timestamp, bid])

        if self.last_optimal_trade[2] == 'BUY':
            if ask < self.last_optimal_trade[1]:
                self.last_optimal_trade = [ timestamp, ask, 'BUY', indicators ]
                return
            if bid >= self.last_optimal_trade[1] * (1 + self.low_high_spread):
                self._register_recent_trade(timestamp, bid, 'SELL', indicators)

        elif self.last_optimal_trade[2] == 'SELL':
            if bid > self.last_optimal_trade[1]:
                self.last_optimal_trade = [ timestamp, bid, 'SELL', indicators ]
                return
            if ask <= self.last_optimal_trade[1] * (1 - self.low_high_spread):
                self._register_recent_trade(timestamp, ask, 'BUY', indicators)

    def _register_recent_trade(self, timestamp: int, price: float, side: str, indicators):
        t = self.last_optimal_trade
        start_time = end_time = t[0]
        start_price = end_price = t[1]
        i_trade = 0
        if t[2] == 'BUY':
            for i, tr in enumerate(self.asks):
                if tr[0] == t[0] and tr[1] == t[1]:
                    i_trade = i
                    break
            max_ask = t[1] * (1 + self.optimal_price_tolerance)
            for i, tr in enumerate(self.asks[i_trade:]):
                if tr[1] > max_ask:
                    end_time, end_price = tr[0], tr[1]
                    break
            for i, tr in enumerate(self.asks[i_trade::-1]):
                if tr[1] > max_ask:
                    start_time, start_price = tr[0], tr[1]
                    break
            min_price = t[1]
            max_price = max(start_price, end_price, t[1])
            self.asks.clear()
        elif t[2] == 'SELL':
            for i, tr in enumerate(self.bids):
                if tr[0] == t[0] and tr[1] == t[1]:
                    i_trade = i
                    break
            min_bid = t[1] * (1 - self.optimal_price_tolerance)
            for i, tr in enumerate(self.bids[i_trade:]):
                if tr[1] < min_bid:
                    end_time, end_price = tr[0], tr[1]
                    break
            for i, tr in enumerate(self.bids[i_trade::-1]):
                if tr[1] < min_bid:
                    start_time, start_price = tr[0], tr[1]
                    break
            min_price = min(start_price, end_price, t[1])
            max_price = t[1]
            self.bids.clear()

        self.history.append({
            'start_time': start_time, 'end_time': end_time,
            'min_price': min_price, 'max_price': max_price,
            'best_time': t[0], 'best_price': t[1], 'side': t[2], 'indicators': t[3]
        })
        self.last_optimal_trade = [timestamp, price, side, indicators]
        for cb in self.new_trade_callbacks:
            cb(self.history[-1])
        # print(f"TRADE SIGNAL: {self.history[-1]}")

    def get_trade_history(self):
        return self.history


class CandleMaker:
    def __init__(self, period_length: datetime.timedelta):
        self.period_length = period_length
        self.current_period_start = None
        self.current_candle = None
        self.candles = []
        self.callbacks = []

    def add_new_candle_callback(self, callback):
        self.callbacks.append(callback)

    def push(self, time: datetime.datetime, value: float):
        period_start = self._floor_time(time)

        if self.current_period_start is None:
            self._start_new_candle(period_start, value)
            return

        if period_start != self.current_period_start:
            self._finalize_candle()
            self._start_new_candle(period_start, value)
        else:
            c = self.current_candle
            c['high'] = max(c['high'], value)
            c['low'] = min(c['low'], value)
            c['close'] = value
            c['sum'] += value
            c['count'] += 1

    def get_candles(self) -> pd.DataFrame:
        return pd.DataFrame(self.candles)

    def get_period_length(self):
        return self.period_length

    def _start_new_candle(self, period_start: datetime.datetime, value: float):
        self.current_period_start = period_start
        self.current_candle = {
            'open_time': period_start,
            'open': value,
            'high': value,
            'low': value,
            'close': value,
            'sum': value,
            'count': 1,
        }

    def _finalize_candle(self):
        if self.current_candle:
            self.current_candle['close_time'] = self.current_period_start + self.period_length
            self.candles.append(self.current_candle)
            for cb in self.callbacks:
                cb(self.current_candle)

    def _floor_time(self, time: datetime.datetime) -> datetime.datetime:
        period_secs = int(self.period_length.total_seconds())
        return datetime.datetime.fromtimestamp((int(time.timestamp()) // period_secs) * period_secs, tz=time.tzinfo)

class OrderBookWeightedImbalance:
    def __init__(self, tob_weight = 1):
        self.tob_cumulative_qty = 1.0 / tob_weight
        self.ratios = [50., 50.]  # default neutral

    def update_order_book(self, bids: list, asks: list):
        if not bids and not asks:
            self.ratios = [50., 50.]
            return
        if not bids:
            self.ratios = [0., 100.]
            return
        if not asks:
            self.ratios = [100., 0.]
            return

        mid = (bids[0][0] + asks[0][0]) / 2.0
        side_weights = []

        for order_list in (bids, asks):
            cumulative_qty = self.tob_cumulative_qty
            weighted_total_on_side = 0.0
            for price, qty in order_list:
                dist = abs(price - mid) or 1e-9
                weight = qty / (cumulative_qty * dist)
                weighted_total_on_side += weight
                cumulative_qty += qty
            side_weights.append(weighted_total_on_side)

        total_weight = sum(side_weights)
        self.ratios = [100 * w / total_weight for w in side_weights] if total_weight > 0 else [50., 50.]

    def get_ratios(self):
        return self.ratios


class OrderBookImbalance:
    def __init__(self):
        self.ratios = [50., 50.]  # default neutral

    def update_order_book(self, bids: list, asks: list):
        if not bids and not asks:
            self.ratios = [50., 50.]
            return
        if not bids:
            self.ratios = [0., 100.]
            return
        if not asks:
            self.ratios = [100., 0.]
            return

        total_bids, total_asks = sum([b[1] for b in bids]), sum(a[1] for a in asks)
        total_orders = total_bids + total_asks
        self.ratios = [100 * total_bids / total_orders, 100 * total_asks / total_orders] if total_orders > 0 else [50., 50.]

    def get_ratios(self):
        return self.ratios
