import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

import ManualStrategy as mstrat
import marketsimcode as mks
import StrategyLearner as slearn


def normalize(df):
    return (df / df.iloc[0])


def author():
    return 'mwong83'  # replace tb34 with your Georgia Tech username.


def run_exp1():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    bm = mstrat.BenchmarkStrategy()
    bm_trades = bm.testPolicy(symbol=symbol, sd=sd, ed=ed,
                              impact=0.005, commission=9.95
                              )
    bm_pv = mks.compute_portvals(bm_trades, symbol)

    ms = mstrat.ManualStrategy()
    ms_trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed,
                              impact=0.005, commission=9.95
                              )
    ms_pv = mks.compute_portvals(ms_trades, symbol)

    sl = slearn.StrategyLearner(impact=0.005, commission=9.95)
    sl.addEvidence(symbol=symbol, sd=sd, ed=ed)
    sl_trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed)
    sl_pv = mks.compute_portvals(sl_trades, symbol)
    plt.figure(figsize=(10, 7))
    plt.plot(normalize(bm_pv), 'r', label='Bechmark Strategy')
    plt.plot(normalize(ms_pv), 'g', label='Manual Strategy')
    plt.plot(normalize(sl_pv), 'b', label='Strategy Learner')
    plt.title('Experiment 1')
    plt.legend(loc='best')
    plt.savefig(fname='experiment_1')
