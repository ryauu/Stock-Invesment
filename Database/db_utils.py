"""
db_utils.py
Module dùng chung cho việc kết nối PostgreSQL và load dữ liệu thô.
Dùng lại bởi cả combine_long_format_sql.py và export_data_features.py
để tránh lặp code get_conn() / load_raw() ở nhiều nơi.
"""
import os
import pandas as pd
import psycopg2
from psycopg2 import OperationalError, Error
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# Danh sách các mã cổ phiếu (KHÔNG bao gồm vnindex)
STOCK_LIST = ['bsi', 'cts', 'agr', 'vds', 'apg', 'tvs','fts', 'hcm', 'ors', 'ssi', 'vix']
INDEX_SYMBOL = 'vnindex'
ALL_SYMBOLS = STOCK_LIST + [INDEX_SYMBOL]


def get_conn():
    """Mở 1 kết nối Postgres mới. Trả về None nếu lỗi."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("dbname"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port")
        )
    except OperationalError as e:
        console.print(f"❌ [bold red]Lỗi kết nối Database (OperationalError): {e}[/]")
    except Error as e:
        console.print(f"❌ [bold red]Lỗi hệ thống Psycopg2: {e}[/]")
    except Exception as e:
        console.print(f"❌ [bold red]Lỗi môi trường Python: {e}[/]")
    return conn


def load_raw(table_name: str) -> pd.DataFrame:
    """Lấy dữ liệu thô (time, open, high, low, close, volume) của 1 bảng, sắp theo thời gian."""
    conn = get_conn()
    if conn is None:
        console.print(f"⚠️  [yellow]Không có kết nối DB, bỏ qua bảng '{table_name}'.[/]")
        return pd.DataFrame()
    try:
        with conn:
            df = pd.read_sql(f"""
                SELECT time, open, high, low, close, volume
                FROM {table_name}
                ORDER BY time ASC
            """, conn)
    except Exception as e:
        console.print(f"❌ [bold red]Lỗi khi query bảng '{table_name}': {e}[/]")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def resample_to_1d(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu tick -> OHLCV theo ngày (1D), đổi tên cột theo hậu tố."""
    if df.empty:
        return df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    # Bỏ những ngày không có dữ liệu (cuối tuần, nghỉ lễ)
    daily.dropna(subset=["open", "high", "low", "close"], inplace=True)

    daily.columns = [f"{col}_{suffix}" for col in daily.columns]
    return daily

def compute_liquidity_ranking() -> pd.DataFrame:
    rows = []
    for symbol in STOCK_LIST:
        raw = load_raw(symbol)
        if raw.empty:
            console.print(f"⚠️  Bảng '{symbol}' rỗng, bỏ qua.")
            continue
 
        raw = raw.copy()
        raw['time'] = pd.to_datetime(raw['time'])
        raw['dollar_volume'] = raw['close'] * raw['volume']
 
        rows.append({
            'symbol': symbol.upper(),
            'first_trade_date': raw['time'].min().date(),
            'last_trade_date': raw['time'].max().date(),
            'n_trading_days': len(raw),
            'avg_dollar_volume': raw['dollar_volume'].mean(),
            'median_dollar_volume': raw['dollar_volume'].median(),
            'avg_volume': raw['volume'].mean(),
        })
 
    df = pd.DataFrame(rows)
    df = df.sort_values('avg_dollar_volume', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['keep_top5'] = df['rank'] <= 5
    return df
 