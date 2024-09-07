''' 
This pairs trading strategy is for trading 2 stocks (in this example AAPL and MSFT). Pairs trading is done to exploit the differences in price of highly correlated stocks. 
If the price of the stocks exceed or are under mean + 2 STD of acceptable fluctuations, then we will buy the stock that is higher and short the stock that is lower in price. 
$9,993.26
Equity
-$2.00
Fees
$967.47
Holdings
$-2.00
Net Profit
5.251%
PSR
-0.07 %
Return
$-6.84
Unrealized
$1,008.74
Volume


'''

from sklearn import linear_model
import numpy as np
import pandas as pd
from scipy import stats
from math import floor
from datetime import timedelta

from AlgorithmImports import *

class PairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        
        self.SetStartDate(2022,1,1)
        self.SetEndDate(2023,9,30)
        self.SetCash(10000)

        self.numdays = 250  # set the length of training period
        tickers = ["MSFT","AAPL"]
        self.symbols = []
        self.hist_window = []
        self.threshold = 2

        for i in tickers:
            self.symbols.append(self.add_equity(i).Symbol)
        for i in self.symbols:
            i.hist_window = RollingWindow[TradeBar](self.numdays) 


    def OnData(self, data):
        if not (data.ContainsKey("MSFT") and data.ContainsKey("AAPL")):
            return
                
        for symbol in self.symbols:
            symbol.hist_window.Add(data[symbol])
        
    
        price_x = pd.Series([float(i.Close) for i in self.symbols[0].hist_window], 
                             index = [i.Time for i in self.symbols[0].hist_window])
                             
        price_y = pd.Series([float(i.Close) for i in self.symbols[1].hist_window], 
                             index = [i.Time for i in self.symbols[1].hist_window])

        if len(price_x) < self.numdays:
            return

        spread = self.regr(np.log(price_x), np.log(price_y))
        mean = np.mean(spread)
        std = np.std(spread)
        ratio = floor(self.Portfolio[self.symbols[1]].Price / self.Portfolio[self.symbols[0]].Price)
        # quantity = float(self.CalculateOrderQuantity(self.symbols[0],0.4)) 
        
        if spread[-1] < mean + self.threshold * std:
            if not self.Portfolio[self.symbols[0]].Quantity > 0 and not self.Portfolio[self.symbols[0]].Quantity < 0:
                self.Sell(self.symbols[1], 2) 
                self.Buy(self.symbols[0],  max(ratio * 1, 2))
        
        elif spread[-1] > mean - self.threshold * std:
            if not self.Portfolio[self.symbols[0]].Quantity < 0 and not self.Portfolio[self.symbols[0]].Quantity > 0:
                self.Sell(self.symbols[0], 2)
                self.Buy(self.symbols[1], max(ratio * 1, 2)) 

        else:
            self.Liquidate()

    
    def regr(self,x,y):
        regr = linear_model.LinearRegression()
        x_constant = np.column_stack([np.ones(len(x)), x])
        regr.fit(x_constant, y)
        beta = regr.coef_[0]
        alpha = regr.intercept_
        spread = y - x*beta - alpha
        return spread
