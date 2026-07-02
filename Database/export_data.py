import pandas as pd
import pandas_ta as ta
import numpy as np
import psycopg2, os
from psycopg2 import OperationalError, Error
from dotenv import load_dotenv
from rich.console import Console
from sklearn.preprocessing import MinMaxScaler

load_dotenv()
console = Console()

def get_conn():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname = os.getenv("dbname"),   
            user = os.getenv("user"),
            password=os.getenv("password"),
            host = os.getenv("host"),
            port = os.getenv("port")
        )

    except OperationalError as e:
        console.print(f"❌ [bold red]Lỗi kết nối Database (OperationalError): {e}[/]")
    except Error as e:
        console.print(f"❌ [bold red]Lỗi hệ thống Psycopg2: {e}[/]")
    except Exception as e:
        console.print(f"❌ [bold red]Lỗi môi trường Python: {e}[/]")
    return conn

def load_data():
    with get_conn() as conn:
        df = pd.read_sql("""
        SELECT open, high, low, close, volume, time
        FROM vix
        ORDER BY time ASC
        """, conn )
    return df

# ---------------------------------------------------------
# XỬ LÝ DỮ LIỆU TỪ CƠ SỞ DỮ LIỆU (KHUNG 1D)
# ---------------------------------------------------------
console.print("[bold cyan]Đang tải dữ liệu từ PostgreSQL...[/]")
df = load_data()

# Time-based Features
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)

# Nén dữ liệu thành khung 1 Ngày (1D)
df_1d = df.resample("1D").agg({
    "open": "first",    
    "high": "max",      
    "low": "min",       
    "close": "last",    
    "volume": "last"    
})

# Loại bỏ các ngày nghỉ
df_1d.dropna(inplace=True)

# ---------------------------------------------------------
# THÊM CÁC CHỈ BÁO KỸ THUẬT (TECHNICAL INDICATORS)
# ---------------------------------------------------------
df_1d["SMA_10"] = ta.sma(length=10, close=df_1d["close"])
df_1d["SMA_20"] = ta.sma(length=20, close=df_1d["close"]) # Thêm SMA 20
df_1d["RSI_14"] = ta.rsi(length=14, close=df_1d["close"])

# Thêm Bollinger Bands
# ta.bbands trả về DataFrame gồm 3 cột: Lower Band, Mid Band (SMA), Upper Band
bb = ta.bbands(length=20, std=2, close=df_1d["close"])
df_1d["BB_lower"] = bb.iloc[:, 0] # Cột đầu tiên là Lower
df_1d["BB_upper"] = bb.iloc[:, 2] # Cột thứ 3 là Upper

# Thêm MACD
# ta.macd trả về DataFrame gồm 3 cột: MACD, Histogram, Signal
macd = ta.macd(fast=12, slow=26, signal=9, close=df_1d["close"])
df_1d["MACD"] = macd.iloc[:, 0]         # Cột đầu tiên là MACD line
df_1d["MACD_Signal"] = macd.iloc[:, 2]  # Cột thứ 3 là Signal line

# Lag Features (Dữ liệu quá khứ)
df_1d["lag_1"] = df_1d["close"].shift(1)
df_1d["lag_5"] = df_1d["close"].shift(5)

# Làm sạch dữ liệu (Loại bỏ các dòng NaN sinh ra do tính toán chỉ báo)
df_1d.dropna(inplace=True)
df_1d.reset_index(inplace=True)

# ---------------------------------------------------------
# NHIỆM VỤ 1: XUẤT FILE BẢNG .CSV
# ---------------------------------------------------------
csv_output = "vix_features_1D_updated.csv"
df_1d.to_csv(csv_output, index=False)
console.print(f"[bold green]✅ NHIỆM VỤ 1:[/] Đã lưu dữ liệu dạng bảng ra file: [bold]{csv_output}[/]")

# ---------------------------------------------------------
# NHIỆM VỤ 2: CHUẨN HÓA, TẠO SEQUENCE & XUẤT 1 FILE TỔNG HỢP (.NPZ)
# ---------------------------------------------------------
console.print("[bold cyan]Đang xử lý dữ liệu Sequence (Sliding Window) cho Model...[/]")

# Loại bỏ cột 'time' để AI xử lý mảng số học
df_ai = df_1d.drop(columns=['time'])

data = df_ai.values 
target_col_index = df_ai.columns.get_loc('close') 

# Chuẩn hóa dữ liệu (Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(dataset, target_index, time_steps=10):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i : (i + time_steps)])
        y.append(dataset[i + time_steps, target_index])
    return np.array(X), np.array(y)

# Lấy 10 ngày quá khứ để dự đoán 1 ngày tiếp theo
time_steps = 10
X, y = create_sequences(scaled_data, target_col_index, time_steps)

# Lưu TẤT CẢ vào 1 file tổng hợp duy nhất (.npz)
npz_output = "vix_lstm_dataset.npz"
np.savez(npz_output, X_data=X, y_data=y)

console.print(f"[bold green]✅ NHIỆM VỤ 2:[/] Đã gom mảng X và y thành 1 file tổng hợp: [bold]{npz_output}[/]")
console.print(f"   - Hình dáng X (Input): {X.shape}")
console.print(f"   - Hình dáng y (Output): {y.shape}")

# ---------------------------------------------------------
# KIỂM TRA DỮ LIỆU
# ---------------------------------------------------------
print("\n--- CÁC CỘT DỮ LIỆU ĐÃ TẠO ---")
print(df_1d.columns.tolist())