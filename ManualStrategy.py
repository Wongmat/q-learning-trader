import datetime as dt
from abc import ABC, abstractmethod

import pandas as pd

import indicators as helpers
from util import get_data


def normalize(df):
    return (df / df[0])


class Strategy(ABC):
    @staticmethod
    def author():
        return 'mwong83'  # replace tb34 with your Georgia Tech username.

    def __init__(self):
        self.shares = 0
        self.holdings = 0
        self.enter_price = 0

    def record_buy(self, date, amt_shares=None):
        if not amt_shares:
            amt_shares = 1000 - self.holdings
        if amt_shares > 0:
            self.holdings += amt_shares
            self.trades.loc[self.trades.index ==
                            date, self.symbol] = amt_shares

    def record_sell(self, date, amt_shares=None):
        if not amt_shares:
            amt_shares = 1000 + self.holdings
        self.holdings -= amt_shares
        self.trades.loc[self.trades.index == date, self.symbol] = -amt_shares

    @abstractmethod
    def run_policy():
        pass

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1),
                   ed=dt.datetime(2011, 12, 31),
                   sv=100000, impact=0, commission=0):
        self.symbol = symbol
        self.start_val = sv
        spy = get_data(['SPY'], pd.date_range(start=sd, end=ed))
        self.adj_close = get_data([symbol], spy.index, addSPY=False)
        self.trades = self.adj_close.copy()
        self.trades[self.symbol] = 0
        self.impact = impact
        self.commission = commission
        return self.run_policy()


class BenchmarkStrategy(Strategy):
    def run_policy(self):
        today = self.adj_close.index[0]
        self.record_buy(today,  amt_shares=abs(self.holdings))
        return self.trades


class ManualStrategy(Strategy):
    def run_policy(self):
        def decide_if_enter(
                today, today_price, today_bbv, ytd_bbv, today_macd, today_rsi):
            b_comp_1 = today_bbv >= -1 and ytd_bbv < -1
            b_comp_2 = today_rsi <= 31
            b_comp_3 = today_macd < 0
            b_signal = b_comp_2 or b_comp_3 if b_comp_1 else b_comp_2 and b_comp_3

            s_comp_1 = today_bbv <= 1 and ytd_bbv > 1
            s_comp_2 = today_rsi >= 69
            s_comp_3 = today_macd > 0
            s_signal = s_comp_2 or s_comp_3 if s_comp_1 else s_comp_2 and s_comp_3
            if b_signal:
                self.record_buy(today)

            if s_signal:
                self.record_sell(today)

            if self.holdings != 0:
                self.enter_price = today_price

        def decide_if_exit(today, today_price,
                           today_bbv, ytd_bbv, today_macd, today_rsi):

            if abs(today_bbv) < 0.05:
                commission_rate = self.commission / 1000
                adj_sell_price = today_price * \
                    (1 - self.impact) - commission_rate
                adj_buy_price = today_price * \
                    (1 + self.impact) + commission_rate

                if self.enter_price < adj_sell_price:
                    self.record_sell(today)

                elif self.enter_price > adj_buy_price:
                    self.record_buy(today)

        self.bbv = helpers.calc_bbv(self.adj_close[self.symbol], 100)
        components, macd = helpers.calc_macd(self.adj_close[self.symbol])
        self.macd = macd['MACD Line']
        self.rsi = helpers.calc_rsi(self.adj_close[self.symbol])

        for ix in range(1, len(self.adj_close)):
            today = self.adj_close.index[ix]
            ytd = self.adj_close.index[ix - 1]
            today_price = self.adj_close.loc[today, self.symbol]

            today_bbv = self.bbv.loc[today]
            ytd_bbv = self.bbv.loc[ytd]

            today_rsi = self.rsi.loc[today]

            today_macd = self.macd.loc[today]

            if (self.holdings == 0):
                decide_if_enter(today=today, today_price=today_price,
                                today_bbv=today_bbv, ytd_bbv=ytd_bbv,
                                today_macd=today_macd, today_rsi=today_rsi)
            else:
                decide_if_exit(today=today, today_price=today_price,
                               today_bbv=today_bbv, ytd_bbv=ytd_bbv,
                               today_macd=today_macd, today_rsi=today_rsi)

        return self.trades


def main():
    symbol = 'XOM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    ms = ManualStrategy()
    bs = BenchmarkStrategy()

    ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    bs.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)


if __name__ == '__main__':
    main()
