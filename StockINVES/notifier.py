import requests,os,time
def send(stock):
    text = {
        "embeds":[
            {
                "title":"📈VIX Alert",
                "color": int("26ff3c",16),
                "fields":[
                    {"name":"Price:","value":str(stock.price),"inline":True},
                    {"name":"priceChange:","value":str(stock.priceChange),"inline":True},
                    {"name":"priceDayHigh:","value":str(stock.priceDayHigh),"inline":True},
                    {"name":"priceDayLow:","value":str(stock.priceDayLow),"inline":True},
                ],
                "footer":{
                    "text": stock.ts
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