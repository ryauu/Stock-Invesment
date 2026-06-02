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
        #Change
        self.netChange = data["netChange"]
        self.pctChange = data["pctChange"]
        #Reference
        self.priceReference= data["priceReference"]
        self.priceFloor= data["priceFloor"]
        self.priceCeiling = data['priceCeiling']
        #Volume
        self.totalVolume = data["totalVolume"]
        self.totalBuyVolume = data["totalBuyVolume"]
        self.totalSellVolume = data["totalSellVolume"]

    def printed(self):
        t = datetime.datetime.fromtimestamp(self.ts)

        change = self.priceClose - self.priceReference
        pct = (change / self.priceReference) * 100

        arrow = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"

        return f"{t.strftime('%H:%M:%S')} | {arrow} {self.priceClose:.2f} ({pct:+.2f}%)"