import psycopg2,os
from datetime import datetime
from psycopg2 import OperationalError,Error
from dotenv import load_dotenv
from rich.console import Console
load_dotenv()

console = Console()
class PostgresManager:
    def __init__(self):
        self.conn = None
        try:
            self.conn = psycopg2.connect(
                dbname = os.getenv("dbname"),   
                user = os.getenv("user"),
                password=os.getenv("password"),
                host = os.getenv("host"),
                port = os.getenv("port")
            )

        except OperationalError as e:
            # Bắt các lỗi do sai mật khẩu, sai DB, mất mạng, DB sập...
            console.print(f"❌ [bold red]Lỗi kết nối Database (OperationalError): {e}")
            
        except Error as e:
            #Bắt các lỗi sâu hơn của psycopg2 nếu có
            console.print(f"❌ [bold red]Lỗi hệ thống Psycopg2: {e}")
            
        except Exception as e:
            #Bắt lỗi của Python
            console.print(f"❌ [bold red]Lỗi môi trường Python: {e}")
            
    def create_tables(self, ticker:str):
        conn = self.conn
        if conn is None:
            console.print(f"[bold red]Không có kết nối DB. Bỏ qua tạo bảng cho {ticker}")
            return
        
        table_name = ticker.lower()
        try:
            with conn:
                with conn.cursor()as cur:
                    cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}(
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMP,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                );
                """)
                    
        except Exception as e:
            console.print(f"❌ [bold red]Lỗi khi kiểm tra/tạo bảng: {e}")

    def save(self,stock,ticker:str):
        conn = self.conn
        if conn is None: return
        try:
            with conn:
                with conn.cursor() as cur:
                    ts = stock.ts
                    if not isinstance(ts, datetime):
                        ts = datetime.fromtimestamp(float(ts))

                    table_name = ticker.lower()

                    cur.execute(f"""
                        INSERT INTO {table_name}(
                        open, high, low,
                        close, volume, time)
                        VALUES (
                        %s, %s, %s,
                        %s, %s, %s,
                        )
                    """, (
                        stock.priceOpen, stock.priceDayHigh, stock.priceDayLow,
                        stock.priceClose,stock.totalVolume,ts
                    ))
        except Exception as e:
            console.print(f"❌ [bold red]Lỗi: {e}")
