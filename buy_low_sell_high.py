'''
Stats: 
$26,556.00
Equity
-$21.00
Fees
$26,495.46
Holdings
$17,746.53
Net Profit
1.543%
PSR
431.12 %
Return
$3,808.47
Unrealized
$256,634.52
Volume

If we have bought and the price is below 90% its average value (we believe that there will be more losses in the future and we should cut our losses) or that it is 130% its average 
value and that the stock will go down soon so we should sell now and retain the profit. 

'''


# region imports
from AlgorithmImports import *
# endregion

class SmoothRedOrangeParrot(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2006, 1, 4)
        self.set_end_date(2023,6,6)
        self.set_cash(5000)
        self.ticker = self.add_equity("SPY", Resolution.DAILY)
        self.newma = self.SMA(self.ticker.Symbol,100)
        self.rsi = self.RSI(self.ticker.Symbol,14,Resolution.DAILY)
        self.SetWarmup(100)

    def on_data(self, data: Slice):
        if self.IsWarmingUp:
            return 
        rsi_value = self.rsi.Current.Value
        if not self.Portfolio.Invested:
            if self.ticker.Price>self.newma.Current.Value:
                self.SetHoldings(self.ticker.Symbol,1)
                #self.StopMarketOrder(self.ticker.Symbol,-1, self.Securities[self.ticker.Symbol].Close*0.90)
        if self.Portfolio.Invested:
            if self.ticker.Price<(self.Portfolio[self.ticker.Symbol].AveragePrice)*0.9:
                self.Liquidate()
        if self.Portfolio.Invested:
            if self.ticker.Price>(self.Portfolio[self.ticker.Symbol].AveragePrice)*1.3:
                self.Liquidate()
        self.plot("SPY","MA30",self.newma.Current.Value)
        self.Plot("SPY","SPY",self.ticker.Price)
    def OnOrderEvent(self,orderEvent):
        order =self.Transactions.GetOrderById(orderEvent.OrderId)
        self.Log("{0}: {1}: {2}: ".format(self.Time,order.Type, orderEvent))
