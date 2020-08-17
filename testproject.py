import datetime as dt
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment1 as e1
import experiment2 as e2
import ManualStrategy as mstrat
import marketsimcode as mks
import StrategyLearner as sl


def normalize(df):
    return (df / df.iloc[0])


def cumulative_ret(port_vals):
    return (port_vals.ix[-1] / port_vals.ix[0]) - 1


def avg_dr(port_vals):
    return port_vals.diff()[1:].mean()


def sd_dr(port_vals):
    return port_vals.diff()[1:].std()


def compute_stats(port_vals):
    return [('CR', cumulative_ret(port_vals)),
            ('AVG. DR', avg_dr(port_vals)),
            ('STDEV OF DR', sd_dr(port_vals)),
            ]


def author(self):
    return 'mwong83'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    symbol = 'JPM'
    i_sd = dt.datetime(2008, 1, 1)
    i_ed = dt.datetime(2009, 12, 31)
    o_sd = dt.datetime(2010, 1, 1)
    o_ed = dt.datetime(2011, 12, 31)
    ms = mstrat.ManualStrategy()
    bs = mstrat.BenchmarkStrategy()

    i_ms_trades = ms.testPolicy(symbol=symbol, sd=i_sd, ed=i_ed,
                                impact=0.005, commission=9.95, sv=100000)
    i_bs_trades = bs.testPolicy(symbol=symbol, sd=i_sd, ed=i_ed,
                                impact=0.005, commission=9.95, sv=100000)
    i_ms_pv = mks.compute_portvals(i_ms_trades, symbol)
    i_bs_pv = mks.compute_portvals(i_bs_trades, symbol)

    long_dates = i_ms_trades[i_ms_trades[symbol] > 0]
    short_dates = i_ms_trades[i_ms_trades[symbol] < 0]
    plt.figure(figsize=(10, 7))
    for date in long_dates.index:
        plt.axvline(date, linewidth=2, color='b')

    for date in short_dates.index:
        plt.axvline(date, linewidth=2, color='k')

    plt.plot(normalize(i_ms_pv), 'r', label='Manual Strategy')
    plt.plot(normalize(i_bs_pv), 'g', label='Benchmark Strategy')
    plt.legend(loc='best')
    plt.title('In-Sample Manual Strat. vs. Benchmark')
    plt.savefig(fname='i_ms_bs')
    plt.close()
    i_ms_stats = compute_stats(normalize(i_ms_pv))
    # print('IN-SAMPLE MANUAL STRAT STATS')
    # for name, stat in i_ms_stats:
    #     print(name, ':', stat)
    # i_bs_stats = compute_stats(normalize(i_bs_pv))
    # print('\n')
    # print('IN-SAMPLE BENCHMARK STRAT STATS')
    # for name, stat in i_bs_stats:
    #     print(name, ':', stat)
    # print('\n')
    o_ms_trades = ms.testPolicy(symbol=symbol, sd=o_sd, ed=o_ed,
                                impact=0.005, commission=9.95, sv=100000)
    o_bs_trades = bs.testPolicy(symbol=symbol, sd=o_sd, ed=o_ed,
                                impact=0.005, commission=9.95, sv=100000)
    o_ms_pv = mks.compute_portvals(o_ms_trades, symbol)
    o_bs_pv = mks.compute_portvals(o_bs_trades, symbol)

    long_dates = o_ms_trades[o_ms_trades[symbol] > 0]
    short_dates = o_ms_trades[o_ms_trades[symbol] < 0]

    plt.figure(figsize=(10, 7))
    for date in long_dates.index:
        plt.axvline(date, linewidth=2, color='b')

    for date in short_dates.index:
        plt.axvline(date, linewidth=2, color='k')

    plt.plot(normalize(o_ms_pv), 'r', label='Manual Strategy')
    plt.plot(normalize(o_bs_pv), 'g', label='Benchmark Strategy')
    plt.legend(loc='best')
    plt.title('Out-of-Sample Manual Strat. vs. Benchmark')
    plt.savefig(fname='o_ms_bs')
    o_ms_stats = compute_stats(normalize(o_ms_pv))
    # print('OUT-SAMPLE MANUAL STRAT STATS')
    # for name, stat in o_ms_stats:
    #     print(name, ':', stat)
    # o_bs_stats = compute_stats(normalize(o_bs_pv))
    # print('\n')
    # print('OUT-SAMPLE BENCHMARK STRAT STATS')
    # for name, stat in o_bs_stats:
    #     print(name, ':', stat)

    learner = sl.StrategyLearner()
    learner.addEvidence()
    learner.testPolicy()
    e1.run_exp1()
    e2.run_exp2()
