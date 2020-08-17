import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def normalize_indicators(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def calc_sma(prices, window):
    return prices.rolling(window=window).mean()[window - 1:]


def calc_bb(prices, window):
    sma = calc_sma(prices, window)
    two_std = 2 * sma.std()
    lower_band = sma - two_std
    higher_band = sma + two_std
    return pd.concat([lower_band, prices, sma, higher_band], axis=1,
                     keys=['Lower Band', 'Price', 'SMA', 'Higher Band'])


def calc_bbv(prices, window):
    sma = calc_sma(prices, window)
    two_std = 2 * sma.std()
    return (prices - sma) / two_std


def gen_bb_chart(prices, window):
    bbv = calc_bbv(prices, window)
    bb = calc_bb(prices, window)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10, 7)
    myFmt = mdates.DateFormatter('%b %Y')
    ax1.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_formatter(myFmt)
    for col in bb.columns:
        ax1.plot(bb[col], label=col)
    ax1.grid(True)
    ax1.legend(loc='best')
    ax2.plot(bbv)
    ax1.set_title('Bollinger Bands')
    ax2.set_title('Bollinger Band Values')
    ax2.axhline(y=1)
    ax2.axhline(y=-1)
    ax2.grid(True)
    plt.savefig(fname='BB')


def calc_ema(prices, window):
    return prices.ewm(span=window, adjust=False).mean()


def calc_macd(prices):
    twelve_day = calc_ema(prices, 12)
    twenty_six = calc_ema(prices, 26)
    macd_line = normalize_indicators(twelve_day - twenty_six)
    signal_line = normalize_indicators(calc_ema(macd_line, 9))
    macd = pd.concat([macd_line, signal_line, macd_line - signal_line],
                     axis=1,
                     keys=['MACD Line', 'Signal Line', 'Difference'])
    components = pd.concat([twelve_day, twenty_six, prices], axis=1,
                           keys=['12-Day EMA', '26-Day EMA', 'Price'])
    return components, macd


def gen_macd_chart(prices):
    components, macd = calc_macd(prices)
    myFmt = mdates.DateFormatter('%b %Y')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(10, 7)
    for col in components.columns:
        if col == 'Price':
            ax1.plot(components[col], 'k', alpha=0.2, label=col)
        else:
            ax1.plot(components[col], label=col)
    ax1.set_title('MACD Components')
    ax1.grid(True)
    ax1.legend(loc='best')
    ax1.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_formatter(myFmt)
    ax2.plot(macd['Signal Line'], 'r', label='Signal Line')
    ax2.plot(macd['MACD Line'], 'y', label='MACD Line')
    ax2.set_title('MACD: Signal Line and MACD Line')
    ax2.grid(True)
    ax2.legend(loc='best')
    ax3.bar(x=macd.index, height=macd['Difference'], color='g')
    ax3.set_title('MACD: Histogram')
    ax3.grid(True)
    plt.savefig(fname='MACD')


def calc_vpt(prices, volume):
    vpt = pd.Series(0, index=volume.index)

    for ix in range(1, len(volume)):
        vol = volume.iloc[ix]
        prev_vpt = vpt.iloc[ix - 1]
        today_close = prices.iloc[ix]
        ytd_close = prices.iloc[ix - 1]
        pct_change = (today_close - ytd_close) / ytd_close
        vpt.iloc[ix] = prev_vpt + (vol * pct_change)

    vpt.iloc[0] = None
    return vpt


def gen_vpt_chart(prices, volume):
    vpt = calc_vpt(prices, volume)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10, 7)
    myFmt = mdates.DateFormatter('%b %Y')
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.plot(prices)
    ax1.set_title('Adjusted Closing Prices')
    ax1.grid(True)
    ax2.xaxis.set_major_formatter(myFmt)
    ax2.plot(vpt, 'r')
    ax2.grid(True)
    ax2.set_title('Volume Price Trend')
    plt.savefig(fname='VPT')


def calc_rsi(prices):
    movements = prices.diff()
    up_days = movements.copy()
    down_days = movements.copy()
    up_days[movements < 0] = 0
    down_days[movements > 0] = 0
    avg_up = up_days.copy()
    avg_down = abs(down_days).copy()
    avg_up[:14] = None
    avg_down[:14] = None
    avg_up[14] = up_days[1:15].mean()
    avg_down[14] = abs(down_days[1:15].mean())
    for ix in range(15, len(avg_up)):
        avg_up.iloc[ix] = (avg_up.iloc[ix] + avg_up.iloc[ix - 1] * 13) / 14
        avg_down.iloc[ix] = (avg_down.iloc[ix] +
                             avg_down.iloc[ix - 1] * 13) / 14
    rs = avg_up / avg_down
    return 100 - (100 / (1 + rs))


def gen_rsi_chart(prices):
    rsi = calc_rsi(prices)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10, 7)
    myFmt = mdates.DateFormatter('%b %Y')
    ax1.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_formatter(myFmt)
    ax1.plot(prices)
    ax1.set_title('Adjusted Closing Prices')
    ax1.grid(True)
    ax2.plot(rsi, 'r')
    ax2.grid(True)
    ax2.axhline(y=30)
    ax2.axhline(y=70)
    ax2.set_title('Relative Strength Indicator')
    plt.savefig(fname='RSI')


def calc_atr(high, low, close):
    h_l = high - low
    h_pc = high.iloc[1:] - close.values[:-1]
    l_pc = low.iloc[1:] - close.values[:-1]
    components = pd.concat([h_l, abs(h_pc), abs(l_pc)], axis=1)
    tr = components.max(axis=1)
    atr = tr.copy()
    atr[:14] = None
    atr[13] = tr[:14].mean()
    for ix in range(14, len(atr)):
        atr.iloc[ix] = (atr.iloc[ix] + atr.iloc[ix - 1] * 13) / 14
    return atr


def gen_atr_chart(high, low, close):
    atr = calc_atr(high, low, close)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(10, 7)
    myFmt = mdates.DateFormatter('%b %Y')
    ax1.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_formatter(myFmt)
    ax1.plot(close)
    ax1.set_title('Adjusted Closing Prices')
    ax1.grid(True)
    ax2.plot(atr, 'r')
    ax2.grid(True)
    ax2.set_title('Average True Range')
    plt.savefig(fname='ATR')


def author():
    return 'mwong83'  # replace tb34 with your Georgia Tech username.
