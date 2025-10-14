import indicators
import signals
from pionex_rec import MarketDataPlayback
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import datetime
import numpy as np
import pandas as pd
import asyncio
import mplfinance as mpf

def plot_market_analysis(candles_df, indicators, optimal_trades, config):
    """
    Plot candles with grouped indicators and optimal trade shading.

    Parameters:
    - candles_df: DataFrame with columns ['time', 'open', 'high', 'low', 'close']
    - indicators: dict of {name: pd.Series}, each Series indexed by time
    - optimal_trades: list of dicts with keys:
        ['start_time', 'end_time', 'min_price', 'max_price', 'best_time', 'best_price', 'side']
    - config: dict like:
        {
            "price_panel": ["EMA_50", "EMA_200", "Bollinger_Upper", "Bollinger_Lower"],
            "oscillator_panel": ["RSI_14", "StochRSI"],
            "ratio_panel": ["Weighted_Imbalance"]
        }
    """

    # --- Step 1: Set up figure and grid ---
    n_panels = 1  # candles always present
    if config.get("oscillator_panel"): n_panels += 1
    if config.get("ratio_panel"): n_panels += 1

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(n_panels, 1, height_ratios=[3] + [1]*(n_panels-1), hspace=0)

    ax_main = fig.add_subplot(gs[0])

    # --- Step 2: Plot candles ---
    times = pd.to_datetime(candles_df['time'])
    ax_main.plot(times, candles_df['close'], color="black", lw=0.8, label="Close")  # simple close-line for skeleton
    # Optional: Replace with mplfinance-style candlesticks if desired

    # --- Step 3: Price-based indicators ---
    for name in config.get("price_panel", []):
        if name in indicators:
            ax_main.plot(indicators[name].index, indicators[name].values, lw=1, label=name)

    # --- Step 4: Optimal trade shading ---
    for trade in optimal_trades:
        start = mdates.date2num(trade['start_time'])
        end = mdates.date2num(trade['end_time'])

        if trade['side'] == 'BUY':
            y0 = trade['best_price']
            y1 = trade['max_price']
            color = (0.2, 0.8, 0.2, 0.3)
        else:  # SELL
            y0 = trade['min_price']
            y1 = trade['best_price']
            color = (0.8, 0.2, 0.2, 0.3)

        rect = Rectangle(
            (start, y0), end - start, y1 - y0,
            facecolor=color, edgecolor=None
        )
        ax_main.add_patch(rect)

    ax_main.set_ylabel("Price")
    ax_main.legend(loc="upper left")

    # --- Step 5: Oscillator panel ---
    panel_index = 1
    if config.get("oscillator_panel"):
        ax_osc = fig.add_subplot(gs[panel_index], sharex=ax_main)
        for name in config["oscillator_panel"]:
            if name in indicators:
                ax_osc.plot(indicators[name].index, indicators[name].values, lw=1, label=name)
        ax_osc.set_ylabel("Oscillator (0-100)")
        ax_osc.set_ylim(0, 100)
        ax_osc.legend(loc="upper left")
        panel_index += 1

    # --- Step 6: Ratio panel ---
    if config.get("ratio_panel"):
        ax_ratio = fig.add_subplot(gs[panel_index], sharex=ax_main)
        for name in config["ratio_panel"]:
            if name in indicators:
                ax_ratio.plot(indicators[name].index, indicators[name].values, lw=1, label=name)
        ax_ratio.set_ylabel("Ratio (0-1)")
        ax_ratio.set_ylim(0, 1)
        ax_ratio.legend(loc="upper left")

    # --- Step 7: Formatting ---
    ax_main.xaxis_date()
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    plt.show()

class RecordingAnalyser:
    def __init__(self):
        self.ind_rel_calc = signals.IndicatorReliabilityCalculator()
        self.indicator_collector = signals.IndicatorCollection(datetime.timedelta(seconds=90))

    async def read_file(self, filepath, start_time=None, end_time=None):
        async with MarketDataPlayback(filepath, start_time=start_time, end_time=end_time) as data_source:
            data_source.subscribe_depth(callback=self._order_book_record)
            data_source.subscribe_trades(callback=self._trades_record)
            await data_source.play()

    def show_plots(self, config={}):
        matplotlib.use("Agg")
        # can_df, reliability_buy, reliability_sell = self.indicator_collector.process()
        candles, reliability_buy, reliability_sell = self.ind_rel_calc.process()
            
        print("buy reliability:")
        print(reliability_buy.head(10))
        print("sell reliability:")
        print(reliability_sell.head(10))
        for num, can_df in enumerate(candles):
            can_df = can_df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            add_plots = [
                mpf.make_addplot(can_df['prime_buy'], type='scatter', marker='^', color='green'),
                mpf.make_addplot(can_df['prime_sell'], type='scatter', marker='v', color='red'),
                # mpf.make_addplot(can_df['rsi'], panel=2, color='purple', ylim=(0,100.)),
                # mpf.make_addplot(can_df['stoch_rsi_d'], panel=2, color='orange'),
                mpf.make_addplot(can_df['stoch_rsi_k'], panel=2, color='yellow'),
                mpf.make_addplot(can_df['wobi'], panel=2, color='green'),
                mpf.make_addplot(can_df['obi'], panel=2, color='blue'),
            ]

            fig, axes = mpf.plot(
                can_df,
                type="candle",
                style="yahoo",
                volume=True,
                addplot=add_plots,
                # panel_ratios=(3,1),  # main chart 3x taller than RSI panel
                returnfig=True
            )

            ax_main = axes[0]  # main candlestick axes
            ax_volume = axes[2]  # volume axes
            ax_rsi = axes[3]  # RSI panel axes
            ax_main.set_title("Market Analysis Plot")
            ax_main.set_ylabel("Price")
            ax_volume.set_ylabel("Volume")
            ax_rsi.set_ylabel("RSI / StochRSI")
            ax_rsi.set_ylim(0, 100)
            ax_main.legend(loc='upper left')
            ax_main.grid(True)
            ax_rsi.grid(True)
            ax_rsi.axhline(70, color='red', linestyle='--', lw=0.7)
            ax_rsi.axhline(30, color='green', linestyle='--', lw=0.7)
            # Format x-axis for minute-level data
            ax_main.xaxis.set_major_locator(mdates.MinuteLocator(interval=10000))
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            fig.autofmt_xdate()
            plt.savefig(f"market_analysis_plot_{num}.png", bbox_inches='tight', dpi=300)

    def _order_book_record(self, bids, asks, time):
        if not bids or not asks:
            return
        self.ind_rel_calc.update_order_book(bids, asks, time)
        self.indicator_collector.update_order_book(bids, asks, time)

    def _trades_record(self, trades):
        if not trades:
            return
        self.ind_rel_calc.update_trades(trades)
        self.indicator_collector.update_trades(trades)


# --- Example usage ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot market analysis data")
    parser.add_argument('file', help="Recording file")
    args = parser.parse_args()
    analyser = RecordingAnalyser()
    start_time=datetime.datetime(2025, 8, 9, 19, 20, 0, tzinfo=datetime.timezone.utc)
    end_time=start_time + datetime.timedelta(hours=12)
    asyncio.run(analyser.read_file(args.file, start_time=start_time, end_time=end_time))
    # analyser.show_plots()
    # # Dummy candles
    # times = pd.date_range(datetime.datetime(2025, 8, 10, 3, 29), periods=50, freq="T")
    # df_candles = pd.DataFrame({
    #     "time": times,
    #     "open": 50 + pd.Series(range(50)),
    #     "high": 101 + pd.Series(range(50)),
    #     "low": 99 + pd.Series(range(50)),
    #     "close": 50 + pd.Series(range(50)) + pd.Series([(-1)**i for i in range(50)])*0.5
    # })

    # # Dummy indicators
    # indicators = {
    #     "EMA_50": pd.Series(100 + pd.Series(range(50))*0.5, index=times),
    #     "RSI_14": pd.Series((pd.Series(range(50)) % 100), index=times),
    #     "Weighted_Imbalance": pd.Series((pd.Series(range(50)) % 10)/10, index=times)
    # }

    # # Dummy trades
    # trades = [
    #     {
    #         "start_time": times[10].to_pydatetime(),
    #         "end_time": times[15].to_pydatetime(),
    #         "min_price": 105,
    #         "max_price": 110,
    #         "best_time": times[11].to_pydatetime(),
    #         "best_price": 106,
    #         "side": "BUY"
    #     },
    #     {
    #         "start_time": times[25].to_pydatetime(),
    #         "end_time": times[30].to_pydatetime(),
    #         "min_price": 120,
    #         "max_price": 125,
    #         "best_time": times[28].to_pydatetime(),
    #         "best_price": 125,
    #         "side": "SELL"
    #     }
    # ]

    # config = {
    #     "price_panel": ["EMA_50"],
    #     "oscillator_panel": ["RSI_14"],
    #     "ratio_panel": ["Weighted_Imbalance"]
    # }

    # plot_market_analysis(df_candles, indicators, trades, config)
