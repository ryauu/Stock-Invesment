import pandas as pd
import pandas_ta as ta
import psycopg2,os
from psycopg2 import OperationalError,Error
from dotenv import load_dotenv
from rich.console import Console
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
        # Bắt các lỗi do sai mật khẩu, sai DB, mất mạng, DB sập...
        console.print(f"❌ [bold red]Lỗi kết nối Database (OperationalError): {e}")
        
    except Error as e:
        #Bắt các lỗi sâu hơn của psycopg2 nếu có
        console.print(f"❌ [bold red]Lỗi hệ thống Psycopg2: {e}")
        
    except Exception as e:
        #Bắt lỗi của Python
        console.print(f"❌ [bold red]Lỗi môi trường Python: {e}")
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
# XỬ LÝ DỮ LIỆU ĐƯA VÀO AI (KHUNG 1D)
# ---------------------------------------------------------

print("Đang tải dữ liệu từ PostgreSQL...")
df = load_data()

# Time-based Features
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)

# Nén dữ liệu thành khung 1 Ngày (1D)
# Vì script realtime lưu liên tục, ta cần tổng hợp lại điểm đầu/cuối của mỗi ngày
df_1d = df.resample("1D").agg({
    "open": "first",    # Lấy giá mở cửa phiên sáng
    "high": "max",      # Lấy mức giá cao nhất đạt được trong ngày
    "low": "min",       # Lấy mức giá thấp nhất trong ngày
    "close": "last",    # Lấy giá chốt phiên (đóng cửa)
    "volume": "last"    # Khối lượng giao dịch thường cộng dồn, lấy giá trị cuối ngày
})

# Loại bỏ các ngày nghỉ (Thứ 7, CN, Lễ) bị Pandas tự động sinh ra với giá trị NaN
df_1d.dropna(inplace=True)

# Technical Indicators (Chỉ báo kỹ thuật)
# Áp dụng các chỉ báo trên giá đóng cửa của khung 1D
df_1d["SMA_10"] = ta.sma(length=10, close=df_1d["close"])
df_1d["RSI_14"] = ta.rsi(length=14, close=df_1d["close"])

# 3. Lag Features (Dữ liệu quá khứ)
# Trích xuất giá đóng cửa của 1 ngày và 5 ngày (tương đương 1 tuần giao dịch) trước đó
df_1d["lag_1"] = df_1d["close"].shift(1)
df_1d["lag_5"] = df_1d["close"].shift(5)

# 4. Làm sạch dữ liệu
# Các dòng đầu tiên sẽ bị NaN vì chưa đủ số ngày gom dữ liệu để tính SMA/RSI/Lag
df_1d.dropna(inplace=True)
df_1d.reset_index(inplace=True)

# ---------------------------------------------------------
# LƯU DỮ LIỆU ĐỂ HUẤN LUYỆN AI
# ---------------------------------------------------------
output_file = "vix_features_1D.csv"
df_1d.to_csv(output_file, index=False)

print(f"\n✅ Đã xử lý xong! Dữ liệu 1D sẵn sàng cho AI được lưu tại: {output_file}")
print("\n--- 5 DÒNG DỮ LIỆU ĐẦU TIÊN ---")
# In ra để kiểm tra nhanh tính toàn vẹn của dữ liệu dạng bảng (tabular data)
print(df_1d[['time',"open","high","low","close","volume","SMA_10","RSI_14","lag_1","lag_5"]].head())