import sqlite3,requests, datetime,time , os
import pandas as pd
#Nguồn
URL= "https://assets.msn.com/service/Finance/QuoteSummary?apikey=0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM&activityId=698745e9-cc23-43d0-a1d7-1474cb056147&ocid=finance-utils-peregrine&cm=vi-vn&it=web&scn=ANON&ids=bxcyrw&intents=Quotes,QuoteDetails&wrapodata=false"
def URL_check():
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
#Tạo Obbject VIX
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
    #Webhook
    def send_info(self):
        text = {
            "embeds":[
                {
                    "title":"📈VIX Alert",
                    "color": int("26ff3c",16),
                    "fields":[
                        {"name":"Price:","value":str(self.price),"inline":True},
                        {"name":"priceChange:","value":str(self.priceChange),"inline":True},
                        {"name":"priceDayHigh:","value":str(self.priceDayHigh),"inline":True},
                        {"name":"priceDayLow:","value":str(self.priceDayLow),"inline":True},
                    ],
                    "footer":{
                        "text": self.ts
                    } 
                }
            ]
        }
        webhook_URL = os.getenv("WEBHOOK")#Link WEBHOOK
        return requests.post(webhook_URL,json=text)
        
#Hàm Input
def info():
    while True:   
        os.system("cls" if os.name=="nt" else "clear")
        try:
            global lineMax
            lineMax = int(input("Nhập số dòng muốn terminal hiển thị [≥1]:"))
            global ky_vong
            ky_vong = float(input("Nhập giá mục tiêu(Target price) mà bạn mong muốn:"))
            ky_vong = round(ky_vong,2)
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

#Hàm Run
def run():
    first_run = True
    line_counts=0
    alert = True
    while True:
        try:
            quote = URL_check()
            x =VIXStock(quote)
            if first_run:
                os.system("cls" if os.name=="nt" else "clear")
                first_run=False
            if lineMax == line_counts:
                os.system("cls" if os.name=="nt" else "clear")
                line_counts=0
            line_counts += 1
            x.to_Frame()
            if round(x.price,2) >= ky_vong and alert:
                x.send_info()
                alert = False
            if round(x.price,2)< ky_vong:
                alert=True
            print(f"Giá hiện tại: {quote['price']}, thời gian tạo: {x.ts}")
        except Exception as e:
            print("Error:", e)
        time.sleep(60)

#Chạy chương trình
if __name__ == "__main__": 
    while True:
        info()
        try:
            run()
        except KeyboardInterrupt:
            os.system("cls" if os.name=="nt" else "clear")
            print("Chương trình sẽ khơi động lại sau 3 giây")
            time.sleep(3)
            continue
