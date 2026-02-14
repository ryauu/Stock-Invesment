import datetime
class VIXStock():
    def __init__(self, quote):
        now = datetime.datetime.now()
        self.ts = now.strftime("%Y-%m-%d, %H:%M:%S")
        self.price = quote["price"]
        self.priceChange= quote["priceChange"]
        self.priceDayHigh= quote["priceDayHigh"]
        self.priceDayLow= quote["priceDayLow"]