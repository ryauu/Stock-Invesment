import os
from dotenv import load_dotenv
load_dotenv()

from engine.collector import fetch_data
from Core.models import model_stock
from Database.PostgreSQL import PostgresManager
from engine.notifier import send
from engine.realtime import get_next_action

import time
import json
from rich.console import Console
from datetime import datetime
console = Console()

AI_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_model_cache", "ai_predictions.json")

def load_ai_cache():
    if os.path.exists(AI_CACHE_PATH):
        try:
            with open(AI_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Lỗi đọc AI Cache: {e}[/red]")
    return None

class SessionAggregator:
    def __init__(self):
        self.daily_data = {}
        self.session_saved = False
        self.ai_triggered = False

    def update(self, stock, ticker):
        if ticker not in self.daily_data:
            self.daily_data[ticker] = {
                "open": stock.priceOpen,
                "high": stock.priceDayHigh,
                "low": stock.priceDayLow,
                "close": stock.priceClose,
                "volume": stock.totalVolume,
                "ts": stock.ts
            }
        else:
            d = self.daily_data[ticker]
            d["high"] = max(d["high"], stock.priceDayHigh)
            d["low"] = min(d["low"], stock.priceDayLow)
            d["close"] = stock.priceClose
            d["volume"] = stock.totalVolume
            d["ts"] = stock.ts

    def get_1d_stock(self, ticker, today_dt=None):
        if ticker not in self.daily_data:
            return None
        d = self.daily_data[ticker]
        mock_data = {
            "priceOpen": d["open"],
            "priceHigh": d["high"],
            "priceLow": d["low"],
            "priceClose": d["close"],
            "totalVolume": d["volume"],
            "priceReference": d["open"]
        }
        stk = model_stock(mock_data)
        
        if today_dt:
            # Ép thời gian về đúng 07:00:00 của ngày hôm đó theo yêu cầu
            stk.ts = today_dt.replace(hour=7, minute=0, second=0, microsecond=0)
        else:
            stk.ts = d["ts"]
            
        return stk

    def reset(self):
        self.daily_data = {}
        self.session_saved = False
        self.ai_triggered = False


def run():

    db = PostgresManager()
    last_save_time = 0
    aggregator = SessionAggregator()

    from Database.db_utils import ALL_SYMBOLS
    stocks = ALL_SYMBOLS
    
    # Mã cổ phiếu CHỈ ĐỊNH để in ra Terminal và gửi lên Discord (tránh spam)
    # Tuy nhiên vẫn ngầm thu thập đủ tất cả các mã trong 'stocks' để AI có thể dự đoán.
    TARGET_TICKER = "vix"

    last_prices = {ticker: None for ticker in stocks}

    for ticker in stocks:
        db.create_tables(ticker)

    ai_cache = load_ai_cache()
    ai_cache_date = ai_cache.get("date") if ai_cache else None

    while True:
        # trong while True, TRƯỚC dòng `if not should_run:`
        now_dt = datetime.now()
        today_str = str(now_dt.date())
        current_minutes = now_dt.hour * 60 + now_dt.minute

        # 1. Reset aggregator vào đầu ngày (trước 9:00)
        if current_minutes < 9 * 60 and aggregator.session_saved:
            aggregator.reset()
            console.print("[cyan]Đã reset dữ liệu phiên (đón ngày mới).[/cyan]")

        # 2. Xử lý cuối phiên (sau 15:00)
        if current_minutes >= 15 * 60:
            if not aggregator.session_saved:
                console.print("[bold green]Cuối phiên giao dịch (15:00)! Bắt đầu chốt nến 1D và lưu DB...[/]")
                for ticker in stocks:
                    stk_1d = aggregator.get_1d_stock(ticker, now_dt)
                    if stk_1d:
                        db.save(stk_1d, ticker)
                        console.print(f"  -> Đã lưu nến 1D cho {ticker}")
                aggregator.session_saved = True
            
            if not aggregator.ai_triggered:
                console.print("[bold yellow]Kích hoạt chạy AI Pipeline ngầm...[/]")
                import threading
                import subprocess
                def run_ai_pipeline():
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    export_script = os.path.join(base_dir, "Database", "export_data.py")
                    predict_script = os.path.join(base_dir, "path", "ranknet", "predict_live.py")
                    try:
                        console.print("[bold yellow]Đang tổng hợp CSV (export_data.py)...[/]")
                        subprocess.run(["python", export_script], check=True)
                        console.print("[bold yellow]Đang chạy AI RankNet (predict_live.py)...[/]")
                        subprocess.run(["python", predict_script], check=True)
                        console.print("[bold green]Đã hoàn thành AI Pipeline! Hãy xem AI Cache mới được tạo ra.[/]")
                    except Exception as e:
                        console.print(f"[bold red]Lỗi AI Pipeline: {e}[/]")

                threading.Thread(target=run_ai_pipeline, daemon=True).start()
                aggregator.ai_triggered = True

        # Load lại AI cache nếu qua ngày mới
        if ai_cache_date != today_str:
            new_cache = load_ai_cache()
            if new_cache and new_cache.get("date") != ai_cache_date:
                ai_cache = new_cache
                ai_cache_date = ai_cache.get("date")
                console.print(f"[cyan]Đã cập nhật AI Cache cho ngày {ai_cache_date}[/cyan]")

        should_run, sleep_time = get_next_action()
        if not should_run:
            console.print(":pause_button: [dark_orange]Trading Suspended[/dark_orange]")
            time.sleep(sleep_time)
            continue

        now = time.time()
        for ticker in stocks:
            prediction = None
            if ai_cache and "rankings" in ai_cache and ticker in ai_cache["rankings"]:
                prediction = ai_cache["rankings"][ticker]
                if ai_cache.get("top_pick") and ai_cache["top_pick"].get("ticker") == ticker:
                    prediction["is_top_pick"] = True

            signal = None
            try:
                # =========================
                # COLLECT & AGGREGATE REALTIME DATA
                # =========================
                raw_data = fetch_data(ticker)
                stock = model_stock(raw_data)
                stock.ticker = ticker
                
                # Gom tick vào RAM thay vì lưu thẳng DB
                aggregator.update(stock, ticker)
                
                # db.save(stock, ticker) KHÔNG LƯU Ở ĐÂY NỮA
                
                # =========================
                # DISCORD NOTIFICATION & TERMINAL (Chỉ gửi mã chỉ định)
                # =========================
                if ticker == TARGET_TICKER:
                    if (stock.priceClose != last_prices[ticker] or now - last_save_time >= 30):
                        send(stock, ticker, prediction=prediction)
                        console.print(f"[{ticker.upper()}] {stock.printed()} | Vol: {stock.totalVolume:,}")
                        last_prices[ticker] = stock.priceClose
                        last_save_time = now # Chỉ reset timer khi đã gửi tin nhắn của mã chính

            except Exception as e:
                console.print(f"[red]Lỗi xử lý {ticker}: {e}[/red]")

        time.sleep(2)

if __name__ == "__main__":
    run()
