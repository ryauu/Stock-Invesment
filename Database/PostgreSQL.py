import psycopg2,os
from datetime import datetime
from psycopg2 import OperationalError,Error

def get_conn():
    try:
        conn = psycopg2.connect(
            dbname = os.getenv("dbname"),   
            user = os.getenv("user"),
            password=os.getenv("password"),
            host = os.getenv("host"),
            port = os.getenv("port")
        )
        return conn
    except OperationalError as e:
        # Bắt các lỗi do sai mật khẩu, sai DB, mất mạng, DB sập...
        print(f"❌ Lỗi kết nối Database (OperationalError): {e}")
        return None
        
    except Error as e:
        # Lưới thứ 2: Bắt các lỗi sâu hơn của psycopg2 nếu có
        print(f"❌ Lỗi hệ thống Psycopg2: {e}")
        return None
        
    except Exception as e:
        # Lưới cuối cùng: Bắt lỗi của Python
        print(f"❌ Lỗi môi trường Python: {e}")
        return None
    
def create_tables(ticker:str):
    conn = get_conn()
    if conn is None:
        return
    table_name = ticker.lower()
    try:
        with conn:
            with conn.cursor()as cur:
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name}(
                id SERIAL PRIMARY KEY,
                price_open NUMERIC,
                price_high NUMERIC,
                price_low NUMERIC,
                price_close NUMERIC,
                net_change NUMERIC,
                pct_change NUMERIC,
                price_reference NUMERIC,
                price_floor NUMERIC,
                price_ceiling NUMERIC,
                total_volume BIGINT,
                total_buy_volume BIGINT,
                total_sell_volume BIGINT,
                created_at TIMESTAMP
                );
        """)
    
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra/tạo bảng: {e}")
    finally:
        conn.close() 

def save(stock,ticker:str):
    conn = get_conn()
    if conn is None: return
    try:
        with conn:
            with conn.cursor() as cur:
                ts = stock.ts
                if not isinstance(ts, datetime):
                    ts = datetime.fromtimestamp(float(ts))

                table_name = ticker.lower()

                cur.execute(f"""
                    INSERT INTO {table_name}(price_open, price_high, price_low, price_close, net_change, pct_change, price_reference, price_floor, price_ceiling, total_volume, total_buy_volume, total_sell_volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s)
                """, (
                    stock.priceOpen,
                    stock.priceDayHigh,
                    stock.priceDayLow,
                    stock.priceClose,
                    stock.netChange,
                    stock.pctChange,
                    stock.priceReference,
                    stock.priceFloor,
                    stock.priceCeiling,
                    stock.totalVolume,
                    stock.totalBuyVolume,
                    stock.totalSellVolume,
                    ts
                ))

    finally:
        conn.close()

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