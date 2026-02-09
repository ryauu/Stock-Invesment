import sqlite3,requests, datetime,time , os
import pandas as pd
#Nguồn
URL= "https://assets.msn.com/service/Finance/QuoteSummary?apikey=0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM&activityId=698745e9-cc23-43d0-a1d7-1474cb056147&ocid=finance-utils-peregrine&cm=vi-vn&it=web&scn=ANON&ids=bxcyrw&intents=Quotes,QuoteDetails&wrapodata=false"
#Kiểm tra lỗi dữ liệu
def CheckError():
    try:
        check= requests.get(URL, timeout=10)# kiểm tra lỗi nếu >10 giây sẽ ngắt
        check.raise_for_status()
        data= check.json()# Chuyển dữ liệu thành JSON
    except requests.RequestException as error:
        raise RuntimeError(f"Yêu cầu API lỗi{error}")
    except ValueError as error:
        raise RuntimeError(f"Lỗi ko đọc được file{error}")
    if len(data) == 0 or not isinstance(data,list):
        raise RuntimeError("API trống hoặc dữ liệu không hợp lệ")
    item = data[0]
    if "quote" not in item:
        raise RuntimeError("Lỗi dữ liệu")
    
    return item["quote"]
class VIXStock():
    def __init__(self, quote):
        now = datetime.datetime.now()
        self.ts = now.strftime("%Y-%m-%d, %H:%M:%S")
        self.price = quote["price"]
        self.priceChange= quote["priceChange"]
        self.priceDayHigh= quote["priceDayHigh"]
        self.priceDayLow= quote["priceDayLow"]
    def to_Frame(self):
        df = pd.DataFrame({
            "price"         :[self.price],
            "priceChange"   :[self.priceChange],
            "priceDayHigh"  :[self.priceDayHigh],
            "priceDayLow"   :[self.priceDayLow],
            "created_at"  :[self.ts],
        })
        with sqlite3.connect("stock_invesment.db") as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS VIX(
                    price Real,
                    priceChange real,
                    priceDayHigh real,
                    priceDayLow real,
                    created_at text
                )
            """)
            cur.execute("""
                INSERT INTO VIX(price,priceChange,priceDayHigh,priceDayLow,created_at)
                VALUES (?,?,?,?,?)
            """, (self.price,self.priceChange,self.priceDayHigh,self.priceDayLow,self.ts)
            )
            cur.execute("""
            DELETE FROM VIX
            WHERE rowid not in(
                SELECT min(rowid)
                from VIX
                group by price,priceChange,priceDayHigh,priceDayLow,date(created_at)
            )
            """)
        return df
def main():
    first_run = True
    line_counts=0
    while True:
        os.system("cls" if os.name=="nt" else "clear")
        try:
            lineMax= int(input("nhập số dòng muốn terminal hiển thị [≥1]:"))
            if lineMax >= 1:
                break
            else:
                print("Phải nhập số ≥ 1:")
                time.sleep(2)
                os.system("cls" if os.name=="nt" else "clear")
                continue
        except ValueError:
            print("Nhập sai")
            time.sleep(2)
            print("hãy nhập lại")
            time.sleep(2)
            os.system("cls" if os.name=="nt" else "clear")
            continue
        except EOFError:
            print("Hãy nhập số mong muốn:")
            time.sleep(3)
            os.system("cls" if os.name=="nt" else "clear")
            return
    while True:
        try:
            quote = CheckError()
            x =VIXStock(quote)
            if first_run:
                os.system("cls" if os.name=="nt" else "clear")
                first_run=False
            if lineMax == line_counts:
                os.system("cls" if os.name=="nt" else "clear")
                line_counts=0
            line_counts += 1
            df = x.to_Frame()
            print(f"Giá hiện tại: {quote['price']}, thời gian tạo: {x.ts}")
        except Exception as e:
            print("Error:", e)

        time.sleep(60)
if __name__ == "__main__":
    main()
