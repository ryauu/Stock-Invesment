import datetime
class model_stock():
    def __init__(self, data):
        #time
        now = datetime.datetime.now()
        self.ts = int(now.timestamp())
        #OHLC
        self.priceOpen = data["priceOpen"]
        self.priceClose = data["priceClose"]
        self.priceDayHigh= data["priceHigh"]
        self.priceDayLow= data["priceLow"]
        #Volume
        self.totalVolume = data["totalVolume"]
    def printed(self):
        t = datetime.datetime.fromtimestamp(self.ts)

        change = self.priceClose - self.priceReference
        pct = (change / self.priceReference) * 100

        arrow = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"

        return f"{t.strftime('%H:%M:%S')} | {arrow} {self.priceClose:.2f} ({pct:+.2f}%)"