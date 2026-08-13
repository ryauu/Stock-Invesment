import requests,os,time
from datetime import datetime

def send(stock, ticker:str, prediction=None):
    webhook_URL = os.getenv("WEBHOOK")
    if not webhook_URL:
        print("Thiếu WEBHOOK URL")
        return

    # time
    time_str = datetime.fromtimestamp(stock.ts).strftime("%d/%m %H:%M:%S")

    # tính toán
    change = stock.priceClose - stock.priceReference
    pct = (change / stock.priceReference) * 100

    # màu
    if stock.priceClose > stock.priceReference:
        color = int("26ff3c", 16)  # xanh
    elif stock.priceClose < stock.priceReference:
        color = int("ff3c3c", 16)  # đỏ
    else:
        color = int("f1c40f", 16)  # vàng

    #  signal
    if pct > 1:
        signal = "🚀 BREAKOUT"
    elif pct < -1:
        signal = "🔻 DUMP"
    else:
        signal = "📊 NORMAL"

    fields = [
        {
            "name": "📊 Change",
            "value": f"{change:+.2f} ({pct:+.2f}%)",
            "inline": False
        },
        {
            "name": "📈 High",
            "value": f"{stock.priceDayHigh:.2f}",
            "inline": True
        },
        {
            "name": "📉 Low",
            "value": f"{stock.priceDayLow:.2f}",
            "inline": True
        },
        {
            "name": "⚖️ Ref",
            "value": f"{stock.priceReference:.2f}",
            "inline": True
        }
    ]

    if prediction:
        rank = prediction.get("rank", "?")
        mean_5d = prediction.get("mean_5d")
        win_rate = prediction.get("win_rate")
        is_top = prediction.get("is_top_pick")
        
        ai_text = f"**Rank:** #{rank}\n"
        if mean_5d is not None and win_rate is not None:
            ai_text += f"**Win Rate:** {win_rate:.0f}% (Kỳ vọng: {mean_5d*100:+.2f}%)\n"
        if is_top:
            ai_text += "🌟 **Khuyến nghị MUA (Top Pick)**"
            
        fields.append({
            "name": "🤖 AI Insights (RankNet)",
            "value": ai_text,
            "inline": False
        })

    # payload
    text = {
        "content": f"{signal} | {ticker.upper()} {stock.priceClose:.2f} ({pct:+.2f}%)",
        "embeds": [
            {
                "title": f"{signal} | {ticker.upper()} {stock.priceClose:.2f}",
                "color": color,
                "fields": fields,
                "footer": {
                    "text": f"🕒 {time_str}"
                }
            }
        ]
    }

    try:
        res = requests.post(webhook_URL, json=text, timeout=5)
        if res.status_code != 204:
            print("Webhook lỗi:", res.status_code, res.text)
        return res
    except requests.RequestException as e:
        print("Lỗi gửi webhook:", e)
        
#Hàm Input
def info():
    while True:   
        os.system("cls" if os.name=="nt" else "clear")
        try:
            lineMax = int(input("Nhập số dòng muốn terminal hiển thị [≥1]:"))
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