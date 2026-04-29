from collector import VIX
from models import VIXStock
from PostgreSQL import save
from notifier import send,info
import time
from realtime import get_next_action
from OHLC import *
#Hàm Run
def run():
    last_price=None
    last_save_time = 0
    last_ohlc_time = 0

    while True:
        should_run, sleep_time = get_next_action()
        if not should_run:
            time.sleep(sleep_time)
            continue

        try:
            data = VIX()
        except Exception as e:
            print("API lỗi:", e)
            time.sleep(5)
            continue
        stock = VIXStock(data)

        now = time.time() 
        save(stock)
        if stock.price != last_price or now - last_save_time >= 30:
            try:
                send(stock)
            except Exception as e:
                print("Webhook lỗi:", e)
            print(stock.printed())

            last_price = stock.price
            last_save_time = now
        
        if now - last_ohlc_time >= 60:
            rows = fetch_raw()
            candles = ohlc_func(rows)
            last_5 = {m: candles[m] for m in sorted(candles)[-5:]}
            print_ohlc(last_5)
            last_ohlc_time = now
        time.sleep(1)
#Chạy chương trình
if __name__ == "__main__":
    run()