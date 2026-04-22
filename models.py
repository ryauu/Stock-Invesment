import datetime
class VIXStock():
    def __init__(self, quote):
        now = datetime.datetime.now()
        self.ts = int(now.timestamp())
        self.price = quote["priceClose"]
        self.priceReference= quote["priceReference"]
        self.priceDayHigh= quote["priceHigh"]
        self.priceDayLow= quote["priceLow"]
    def printed(self):
        t = datetime.datetime.fromtimestamp(self.ts)

        change = self.price - self.priceReference
        pct = (change / self.priceReference) * 100

        arrow = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"

        return f"{t.strftime('%H:%M:%S')} | {arrow} {self.price:.2f} ({pct:+.2f}%)"