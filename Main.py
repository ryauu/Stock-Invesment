from engine.collector import fetch_data
from Core.models import model_stock
from Database.PostgreSQL import PostgresManager
from engine.notifier import send
from engine.realtime import get_next_action

from engine.predictor import Predictor
from engine.feature_engineering import (
    generate_features,
    validate_feature_rows
)

import time
from rich.console import Console
from datetime import datetime
console = Console()


def run():

    db = PostgresManager()
    predictor = Predictor()
    last_predict_day = None
    last_save_time = 0

    stocks = ["vix","vnindex"]

    last_prices = {
        "vix": None,
        "vnindex": None
    }

    for ticker in stocks:
        db.create_tables(ticker)

    while True:
        # trong while True, TRƯỚC dòng `if not should_run:`
        now_dt = datetime.now()
        today = now_dt.date()
        if now_dt.hour >= 15 and last_predict_day != today:
            try:
                df = db.get_vix_enriched_rows(limit=300)
                feature_df = generate_features(df)
                validate_feature_rows(feature_df, required_rows=1)   # validate TRƯỚC predict, xem mục 4
                prediction = predictor.predict(feature_df)
                print(f"[AI] Probability Up = {prediction:.4f}")

                latest = db.get_latest_rows("vix", limit=1)
                if not latest.empty:
                    row = latest.iloc[0]
                    stock_for_alert = model_stock({
                        "priceOpen": row["open"],
                        "priceHigh": row["high"],
                        "priceLow": row["low"],
                        "priceClose": row["close"],
                        "totalVolume": row["volume"],
                    })
                    send(stock_for_alert, "vix", prediction=prediction)
                else:
                    console.print("[yellow]Chưa có dữ liệu VIX trong bảng 'vix' để đính kèm thông báo[/yellow]")

                last_predict_day = today
            except Exception as e:
                console.print(f"[yellow]Prediction error: {e}")

        should_run, sleep_time = get_next_action()
        if not should_run:
            console.print(":pause_button: [dark_orange]Trading Suspended[/dark_orange]")
            time.sleep(sleep_time)
            continue

        now = time.time()
        for ticker in stocks:
            prediction = None
            signal = None
            try:
                # =========================
                # SAVE REALTIME DATA
                # =========================
                raw_data = fetch_data(ticker)
                stock = model_stock(raw_data)
                stock.ticker = ticker
                db.save(stock, ticker)
                # =========================
                # DISCORD NOTIFICATION
                # =========================
                if (stock.priceClose != last_prices[ticker] or now - last_save_time >= 30):
                    send(stock, ticker, prediction=prediction)
                    console.print(f"[{ticker.upper()}] {stock.printed()}")
                    last_prices[ticker] = (stock.priceClose)

            except Exception as e:
                console.print(f"[red]Lỗi xử lý "f"{ticker}: "f"{e}[/red]")

        last_save_time = now
        time.sleep(2)

if __name__ == "__main__":
    run()
