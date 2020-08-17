import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

import ManualStrategy as mstrat
import marketsimcode as mks
import StrategyLearner as slearn


def author():
    return 'mwong83'  # replace tb34 with your Georgia Tech username.


def normalize(df):
    return (df / df.iloc[0])


def run_exp2():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    sl_1 = slearn.StrategyLearner(impact=0.005, commission=0)
    sl_1.addEvidence(symbol=symbol, sd=sd, ed=ed)
    sl_1_trades = sl_1.testPolicy(symbol=symbol, sd=sd, ed=ed)
    sl_1_pv = mks.compute_portvals(
        sl_1_trades, symbol, commission=0, impact=0.005)

    sl_2 = slearn.StrategyLearner(impact=0.25, commission=0)
    sl_2.addEvidence(symbol=symbol, sd=sd, ed=ed)
    sl_2_trades = sl_2.testPolicy(symbol=symbol, sd=sd, ed=ed)
    sl_2_pv = mks.compute_portvals(
        sl_2_trades, symbol, commission=0, impact=0.25)

    sl_3 = slearn.StrategyLearner(impact=0.50, commission=0)
    sl_3.addEvidence(symbol=symbol, sd=sd, ed=ed)
    sl_3_trades = sl_3.testPolicy(symbol=symbol, sd=sd, ed=ed)
    sl_3_pv = mks.compute_portvals(
        sl_3_trades, symbol, commission=0, impact=0.50)

    plt.figure(figsize=(10, 7))
    plt.plot(normalize(sl_1_pv), 'b', label='0.005 Impact')
    plt.plot(normalize(sl_2_pv), 'r', label='0.25 Impact')
    plt.plot(normalize(sl_3_pv), 'g', label='0.50 Impact')
    plt.title('Experiment 2: Increased Impact vs. Performance')
    plt.legend(loc='best')
    plt.savefig(fname='experiment_2a')
    plt.close()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.bar(x=['0.05', '0.25', '0.50'], height=[
        len(sl_1_trades[sl_1_trades[symbol] != 0]),  len(
            sl_2_trades[sl_2_trades[symbol] != 0]), len(
            sl_3_trades[sl_3_trades[symbol] != 0])]
           )
    ax.set_title('Experiment 2: Number of Trades')
    plt.savefig(fname='experiment_2b')
