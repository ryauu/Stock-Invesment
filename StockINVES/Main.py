from collector import fetch_quote
from models import VIXStock
from database import sql
from notifier import send
import os,time
#Hàm Run
def run():
    while True:
        quote = fetch_quote()
        stock = VIXStock(quote)
        sql(stock)
        send(stock)
        time.sleep(60)
#Chạy chương trình
if __name__ == "__main__": 
    while True:
        try:
            run()
        except KeyboardInterrupt:
            os.system("cls" if os.name=="nt" else "clear")
            print("Chương trình sẽ khơi động lại sau 3 giây")
            time.sleep(3)
            continue
