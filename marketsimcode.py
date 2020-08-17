import datetime as dt

import pandas as pd

from util import get_data


def compute_portvals(trades, symbol, start_val=100000,
                     commission=9.95, impact=0.005):

    start_date = trades.index[0]
    end_date = trades.index[-1]
    spy = get_data(['SPY'], pd.date_range(start=start_date, end=end_date))

    prices = get_data([symbol], spy.index, addSPY=False)
    prices['Cash'] = 1.0
    trades['Cash'] = prices[symbol] * -trades[symbol]

    sell = (trades['Cash'] > 0)
    buy = (trades['Cash'] < 0)
    trades['Cash'].loc[sell] = trades['Cash'][sell] * \
        (1 - impact) - commission
    trades['Cash'].loc[buy] = trades['Cash'][buy] * \
        (1 + impact) - commission

    holdings = trades.copy()
    holdings.iloc[0, -1] += start_val
    for ix in range(1, len(holdings)):
        holdings.iloc[ix] += holdings.iloc[ix - 1]
    values = holdings * prices
    port_vals = values.sum(axis=1)

    return port_vals
