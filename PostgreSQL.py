import psycopg2
from datetime import datetime

def get_conn():
    return psycopg2.connect(
        dbname="postgres",   # sau này đổi thành stock_db
        user="postgres",
        password="0979281257",
        host="localhost",
        port="5432"
    )

def save(stock):
    with get_conn() as conn:
        with conn.cursor() as cur:
            ts = stock.ts
            if not isinstance(ts, datetime):
                ts = datetime.fromtimestamp(float(ts))

            cur.execute("""
                INSERT INTO vix(price, pricereference, pricedayhigh, pricedaylow, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                stock.price,
                stock.priceReference,
                stock.priceDayHigh,
                stock.priceDayLow,
                ts
            ))

def remove():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM vix
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM vix
                    GROUP BY price, pricereference, pricedayhigh, pricedaylow, DATE(created_at)
                )
            """)