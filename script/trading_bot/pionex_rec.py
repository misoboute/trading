# trading_bot/md_recorder.py

import asyncio
import pionex
import datetime
import struct


FILE_HEADER_STRUCT = struct.Struct('<I H B B')  # File type specifier, file subtype specifier, major version, minor version
TRADES_HEADER_STRUCT = struct.Struct('<B B')  # record type (0), number of trades ()
TRADE_ENTRY_STRUCT = struct.Struct('<Q d d B')  # timestamp, price, size, side (0 = buy, 1 = sell)
DEPTH_HEADER_STRUCT = struct.Struct('<B d B B')  # record type (1), timestamp, number of bids, number of asks
DEPTH_ENTRY_STRUCT = struct.Struct('<d d')  # price, size
FILE_TYPE_SPECIFIER = 0xA1B2C3D4  # Magic number
FILE_SUBTYPE_SPECIFIER = 0x0001 # Pionex recording
RECORDING_VERSION_MAJOR = 1
RECORDING_VERSION_MINOR = 0
TRADES_RECORD_TYPE = 0
DEPTH_RECORD_TYPE = 1
MAX_DEPTH_LEVELS = 10


class PionexMarketDataRecorder:
    def __init__(self, pair: str, filepath: str, data_feed: pionex.PionexMarketDataFeed):
        self.pair = pair
        self.data_feed = data_feed
        self.filepath = filepath

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.stop()

    def handle_trades(self, trades):
        self.file.write(TRADES_HEADER_STRUCT.pack(TRADES_RECORD_TYPE, len(trades)))
        for trade in trades:
            ts = int(trade["time"].timestamp() * 1000)
            price = float(trade["price"])
            size = float(trade["size"])
            side = 0 if trade["side"] == "BUY" else 1
            self.file.write(TRADE_ENTRY_STRUCT.pack(ts, price, size, side))

    def handle_depth(self, bids, asks, time: datetime.datetime):
        bids = bids[:MAX_DEPTH_LEVELS]
        asks = asks[:MAX_DEPTH_LEVELS]
        self.file.write(DEPTH_HEADER_STRUCT.pack(DEPTH_RECORD_TYPE, int(time.timestamp() * 1000), len(bids), len(asks)))
        for price, size in (bids + asks):
            self.file.write(DEPTH_ENTRY_STRUCT.pack(float(price), float(size)))

    async def start(self):
        self.file = open(self.filepath, "wb")
        self.file.write(FILE_HEADER_STRUCT.pack(
            FILE_TYPE_SPECIFIER, FILE_SUBTYPE_SPECIFIER, RECORDING_VERSION_MAJOR, RECORDING_VERSION_MINOR))
        await self.data_feed.subscribe_trades(self.pair, self.handle_trades)
        await self.data_feed.subscribe_depth(self.pair, MAX_DEPTH_LEVELS, self.handle_depth)

    async def stop(self):
        if self.file:
            self.file.close()


class MarketDataPlayback:
    def __init__(self, filepath, start_time=None, end_time=None):
        self.filepath = filepath
        self.trade_callbacks = []
        self.depth_callbacks = []
        self.data_available = True
        self.start_time = start_time
        self.end_time = end_time

    async def __aenter__(self):
        self.file = open(self.filepath, "rb")
        self._verify_header()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def _verify_header(self):
        header = self.file.read(FILE_HEADER_STRUCT.size)
        if len(header) != FILE_HEADER_STRUCT.size:
            raise ValueError("Incomplete or missing file header")

        file_type, subtype, major, minor = FILE_HEADER_STRUCT.unpack(header)
        if file_type != FILE_TYPE_SPECIFIER:
            raise ValueError("Invalid file type specifier")
        if subtype != FILE_SUBTYPE_SPECIFIER:
            raise ValueError("Unsupported file subtype")
        if (major, minor) != (RECORDING_VERSION_MAJOR, RECORDING_VERSION_MINOR):
            raise ValueError("Unsupported version")

    def subscribe_trades(self, callback):
        self.trade_callbacks.append(callback)

    def subscribe_depth(self, callback):
        self.depth_callbacks.append(callback)

    async def play(self):
        while self.data_available:
            type_byte = self.file.peek(1)[:1]
            if not type_byte:
                break
            record_type = type_byte[0]

            if record_type == TRADES_RECORD_TYPE:
                self._play_trades()
            elif record_type == DEPTH_RECORD_TYPE:
                self._play_depth()
            else:
                raise ValueError(f"Unknown record type: {record_type}")

            await asyncio.sleep(0)  # yield control to event loop

    def _play_trades(self):
        header = self._read_exact_num_bytes_or_none(TRADES_HEADER_STRUCT.size)
        if not header:
            return
        
        _, count = TRADES_HEADER_STRUCT.unpack(header)
        trades = []
        for _ in range(count):
            entry = self._read_exact_num_bytes_or_none(TRADE_ENTRY_STRUCT.size)
            if not entry:
                return
            ts, price, size, side = TRADE_ENTRY_STRUCT.unpack(entry)
            time = datetime.datetime.fromtimestamp(ts / 1000.0, tz=datetime.timezone.utc)
            if self.start_time and time < self.start_time:
                continue
            if self.end_time and time > self.end_time:
                self.data_available = False
                return
            trades.append({
                "time": time,
                "price": price,
                "size": size,
                "side": "BUY" if side == 0 else "SELL"
            })
        for cb in self.trade_callbacks:
            cb(trades)

    def _play_depth(self):
        header = self._read_exact_num_bytes_or_none(DEPTH_HEADER_STRUCT.size)
        if not header:
            return
        _, timestamp, bid_count, ask_count = DEPTH_HEADER_STRUCT.unpack(header)
        levels = bid_count + ask_count
        entries = []
        for _ in range(levels):
            entry = self._read_exact_num_bytes_or_none(DEPTH_ENTRY_STRUCT.size)
            if not entry:
                return
            price, size = DEPTH_ENTRY_STRUCT.unpack(entry)
            entries.append((price, size))
        bids = entries[:bid_count]
        asks = entries[bid_count:]
        for cb in self.depth_callbacks:
            time = datetime.datetime.fromtimestamp(timestamp / 1000.0, tz=datetime.timezone.utc)
            if self.start_time and time < self.start_time:
                continue
            if self.end_time and time > self.end_time:
                self.data_available = False
                return
            cb(bids, asks, time)

    def _read_exact_num_bytes_or_none(self, size: int):
        bytes = self.file.read(size)
        if len(bytes) < size:
            partial_bytes = self.file.read(size - len(bytes))
            bytes += partial_bytes
            while partial_bytes and len(bytes) < size:
                partial_bytes = self.file.read(size - len(bytes))
                bytes += partial_bytes
            if len(bytes) < size:
                self.data_available = False
                return None
        return bytes