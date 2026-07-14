import pandas as pd
import pandas_ta as ta
import numpy as np
import psycopg2, os
import warnings
from psycopg2 import OperationalError, Error
from dotenv import load_dotenv
from rich.console import Console

# Bỏ qua cảnh báo kết nối DB của Pandas
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()
console = Console()


def get_conn():
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


def load_raw(table_name: str):
    """Lấy dữ liệu thô (từng tick) của 1 bảng."""
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        with conn:
            df = pd.read_sql(f"""
                SELECT time, open, high, low, close, volume
                FROM {table_name}
                ORDER BY time ASC
            """, conn)
    finally:
        conn.close()
    return df


def resample_to_1d(df: pd.DataFrame, suffix: str):
    """Chuẩn hóa dữ liệu tick -> OHLCV theo ngày (1D)."""
    if df.empty:
        return df

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

    # Đổi tên cột theo hậu tố (vix / vnindex)
    daily.columns = [f"{col}_{suffix}" for col in daily.columns]
    return daily

def load_data():
    """Full join 2 bảng vix & vnindex SAU KHI đã chuẩn hóa về khung 1D."""
    raw_vix = load_raw("vix")
    raw_vnindex = load_raw("vnindex")

    if raw_vix.empty or raw_vnindex.empty:
        return pd.DataFrame()

    daily_vix = resample_to_1d(raw_vix, "vix")
    daily_vnindex = resample_to_1d(raw_vnindex, "vnindex")

    # Full outer join theo NGÀY (index sau resample là timestamp 00:00 mỗi ngày)
    df = daily_vix.join(daily_vnindex, how="outer")
    df.reset_index(inplace=True)   # đưa index 'time' về thành cột
    return df


# ---------------------------------------------------------
# 1. TẢI DỮ LIỆU TỪ POSTGRESQL (FULL JOIN 2 BẢNG VIX & VNINDEX)
# ---------------------------------------------------------
console.print("[bold cyan]Đang tải dữ liệu từ PostgreSQL (full join vix & vnindex)...[/]")
df_vix = load_data()

if df_vix.empty:
    console.print("[bold red]❌ Lỗi: Dữ liệu tải từ Database đang TRỐNG. Vui lòng kiểm tra lại bảng 'vix' và 'vnindex' trong PostgreSQL![/]")
    exit()

# Loại bỏ các dòng không khớp thời gian giữa 2 bảng (full join sinh ra NaN)
before_rows = len(df_vix)
df_vix.dropna(inplace=True)
after_rows = len(df_vix)
console.print(f"[bold yellow]👉 Full join xong: {before_rows} dòng -> sau dropna còn {after_rows} dòng khớp thời gian.[/]")

if df_vix.empty:
    console.print("[bold red]❌ Lỗi: Sau khi dropna, không còn dòng nào khớp thời gian giữa 2 bảng![/]")
    exit()

# Time-based Features
df_vix['time'] = pd.to_datetime(df_vix['time'])
df_vix.set_index("time", inplace=True)
df_vix.sort_index(inplace=True)

# ---------------------------------------------------------
# 2. TẠO ĐẶC TRƯNG TỪ VN-INDEX (LIÊN THỊ TRƯỜNG)
# ---------------------------------------------------------
console.print("[bold cyan]Đang tính toán đặc trưng VN-Index...[/]")
df_vix['vnindex_close_lag_1'] = df_vix['close_vnindex'].shift(1)
df_vix['vnindex_close_lag_2'] = df_vix['close_vnindex'].shift(2)
df_vix['vnindex_close_lag_3'] = df_vix['close_vnindex'].shift(3)
df_vix['vnindex_return_1d_lag_1'] = df_vix['close_vnindex'].pct_change().shift(1)
df_vix['vix_return_1d_lag_1'] = df_vix['close_vix'].pct_change().shift(1)
df_vix['vnindex_ma5_lag_1'] = df_vix['close_vnindex'].rolling(window=5).mean().shift(1)
df_vix['vix_vol_ma5_lag_1'] = df_vix['volume_vix'].rolling(window=5).mean().shift(1)

# ---------------------------------------------------------
# 3. THÊM CÁC CHỈ BÁO KỸ THUẬT
# (Sử dụng dữ liệu của VIX để tính toán)
# ---------------------------------------------------------
console.print("[bold cyan]Đang tính toán chỉ báo kỹ thuật cho VIX...[/]")
# Chỉ báo cơ bản
df_vix["SMA_10"] = ta.sma(length=10, close=df_vix["close_vix"])
df_vix["SMA_20"] = ta.sma(length=20, close=df_vix["close_vix"])
df_vix["RSI_14"] = ta.rsi(length=14, close=df_vix["close_vix"])

# Khoảng cách tương đối so với SMA
df_vix["SMA_10_dist"] = (df_vix["close_vix"] - df_vix["SMA_10"]) / df_vix["SMA_10"]
df_vix["SMA_20_dist"] = (df_vix["close_vix"] - df_vix["SMA_20"]) / df_vix["SMA_20"]

# Bollinger Bands
bb = ta.bbands(length=20, std=2, close=df_vix["close_vix"])
if bb is not None:
    df_vix["BB_lower"] = bb.iloc[:, 0]
    df_vix["BB_upper"] = bb.iloc[:, 2]
    df_vix["BB_width"] = (df_vix["BB_upper"] - df_vix["BB_lower"]) / df_vix["SMA_20"]
    df_vix["BB_position"] = (df_vix["close_vix"] - df_vix["BB_lower"]) / (df_vix["BB_upper"] - df_vix["BB_lower"])

# MACD
macd = ta.macd(fast=12, slow=26, signal=9, close=df_vix["close_vix"])
if macd is not None:
    df_vix["MACD"] = macd.iloc[:, 0]
    df_vix["MACD_Signal"] = macd.iloc[:, 2]
    df_vix["MACD_Hist"] = macd.iloc[:, 1]

# Biến động (Volatility)
df_vix["ATR_14"] = ta.atr(high=df_vix["high_vix"], low=df_vix["low_vix"], close=df_vix["close_vix"], length=14)
df_vix["ATR_14_ratio"] = df_vix["ATR_14"] / df_vix["close_vix"]
df_vix["Rolling_Std_10"] = df_vix["close_vix"].rolling(window=10).std()

# Khối lượng
df_vix["OBV"] = ta.obv(close=df_vix["close_vix"], volume=df_vix["volume_vix"])

# Động lượng & Xu hướng
adx = ta.adx(high=df_vix["high_vix"], low=df_vix["low_vix"], close=df_vix["close_vix"], length=14)
if adx is not None:
    df_vix["ADX_14"] = adx.iloc[:, 0]

stoch = ta.stoch(high=df_vix["high_vix"], low=df_vix["low_vix"], close=df_vix["close_vix"])
if stoch is not None:
    df_vix["Stoch_k"] = stoch.iloc[:, 0]
    df_vix["Stoch_d"] = stoch.iloc[:, 1]

# Đặc trưng chu kỳ (Lấy từ index datetime)
df_vix["day_of_week"] = df_vix.index.dayofweek
df_vix["month"] = df_vix.index.month

# Return nhiều khung thời gian của VIX
df_vix["return_5d"] = df_vix["close_vix"].pct_change(5)
df_vix["return_10d"] = df_vix["close_vix"].pct_change(10)
df_vix["return_20d"] = df_vix["close_vix"].pct_change(20)

# ---------------------------------------------------------
# 4. TẠO BIẾN MỤC TIÊU (TARGET VARIABLES) CHO MACHINE LEARNING
# ---------------------------------------------------------
console.print("[bold cyan]Đang tạo biến Target...[/]")
# --- Mục tiêu T+1 (Dự báo ngày mai) ---
df_vix["target_close_T1"] = df_vix["close_vix"].shift(-1)
df_vix["target_return_T1"] = df_vix["target_close_T1"] / df_vix["close_vix"] - 1
df_vix["target_trend_T1"] = np.where(df_vix["target_return_T1"] > 0, 1, 0)

# --- Mục tiêu T+3 (Dự báo xu hướng 3 ngày tới - Tránh tín hiệu nhiễu) ---
df_vix["target_close_T3"] = df_vix["close_vix"].shift(-3)
df_vix["log_return_T3"] = np.log(df_vix["target_close_T3"] / df_vix["close_vix"])
df_vix["target_trend_T3"] = np.where(df_vix["log_return_T3"] > 0.02, 1, 0)
# Tìm giá cao nhất trong 5 phiên tiếp theo
df_vix["max_close_next_5_days"] = df_vix["close_vix"].shift(-1).rolling(window=5, min_periods=1).max()
df_vix["log_return_max"] = np.log(df_vix["max_close_next_5_days"] / df_vix["close_vix"])
df_vix["target_trend_T5"] = np.where(df_vix["log_return_max"] > 0.02, 1, 0)
# Chỉ tìm giá cao nhất từ T+3 đến T+5
df_vix["max_close_next_3_days_after_T2"] = df_vix["close_vix"].shift(-3).rolling(window=3, min_periods=1).max()
df_vix["log_return_max_real"] = np.log(df_vix["max_close_next_3_days_after_T2"] / df_vix["close_vix"])
df_vix["target_trend_T5_Real"] = np.where(df_vix["log_return_max_real"] > 0.02, 1, 0)

# ---------------------------------------------------------
# 5. LÀM SẠCH VÀ XUẤT DỮ LIỆU
# ---------------------------------------------------------
# Loại bỏ các dòng NaN (do tính toán các chỉ báo như MACD cần 26 ngày và shift dữ liệu)
df_vix.dropna(inplace=True)

# Khôi phục index thời gian về thành một cột bình thường
df_vix.reset_index(inplace=True)

# Thêm cột auto_id
df_vix.insert(0, 'auto_id', range(1, 1 + len(df_vix)))

csv_output = "vix_features_1D_updated.csv"
df_vix.to_csv(csv_output, index=False)

console.print(f"[bold green]✅ HOÀN TẤT:[/] Đã xuất dữ liệu tổng hợp ra file: [bold]{csv_output}[/]")
console.print(f"[bold yellow]👉 Dataset có {len(df_vix)} dòng và {len(df_vix.columns)} cột.[/]")