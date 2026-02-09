import requests, datetime, sqlite3, time,os
import pandas as pd
#API
URL = "https://assets.msn.com/service/Finance/QuoteSummary?apikey=0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM&activityId=698745e9-cc23-43d0-a1d7-1474cb056147&ocid=finance-utils-peregrine&cm=vi-vn&it=web&scn=ANON&ids=bxcyrw&intents=Quotes,QuoteDetails&wrapodata=false"
#Check ERROR
def FetchQuote():
    try:
        check = requests.get(URL,timeout=10)#check file
        check.raise_for_status()
        data= check.json()#Transfer Data to JSON
    except requests.RequestException as error:
        raise RuntimeError(f"API request failed: {error}")
    except ValueError as error:
        raise RuntimeError(f"Invalid JSON response: {error}")
    if not isinstance(data,list) or len(data) == 0:
        raise RuntimeError("API returned empty or invalid data")
    item = data[0]
    if "quote" not in item:
        raise RuntimeError("Missing 'quote' in API response")
    
    return item["quote"]
#Data
class VIXstock():
    def __init__(self,quote):#INPUT
        now = datetime.datetime.now()
        self.ts = now.strftime("%Y-%m-%d %H:%M:%S")
        self.price = quote["price"]
        self.priceChange = quote["priceChange"]
        self.priceDayHigh = quote["priceDayHigh"]
        self.priceDayLow = quote["priceDayLow"]
    def to_dataframe(self):#DATAFRAME
        df = pd.DataFrame({
            "price"        :[self.price],
            "priceChange"  :[self.priceChange],
            "priceDayHigh" :[self.priceDayHigh],
            "priceDayLow"  :[self.priceDayLow],
            "created_at"   : [self.ts]
        })
        with sqlite3.connect("stock_invesment.db") as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS VIX(
                    price REAL,
                    priceChange REAL,
                    priceDayHigh REAL,
                    priceDayLow REAL,
                    created_at TEXT
                )
            """)
            cur.execute("""
                INSERT INTO VIX (price, priceChange, priceDayHigh, priceDayLow, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (self.price, self.priceChange, self.priceDayHigh, self.priceDayLow, self.ts))
            cur.execute("""
                DELETE FROM vix
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM vix
                    GROUP BY price,priceChange,priceDayHigh,priceDayLow,date(created_at)
                )
            """)    
        return df
def main():
    first_run = True
    line_count=0
    max_count = 15
    while True:
        try:
            if first_run:
                os.system("cls"if os.name == "nt" else "clear")
                first_run=False

            if max_count == line_count:
                os.system("cls"if os.name == "nt" else "clear")
                line_count= 0
            line_count += 1

            quote = FetchQuote()
            x = VIXstock(quote)
            df = x.to_dataframe()
            print(f"Saved at {x.ts}")
            
        except Exception as e:
            print("Error:", e)

        time.sleep(60)
if __name__ == "__main__":
    main()