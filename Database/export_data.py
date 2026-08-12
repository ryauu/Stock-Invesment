"""
run_all.py

CHỈ CẦN CHẠY FILE NÀY. Không cần file nào khác ngoài db_utils.py.

    python run_all.py

Sẽ tự động:
  1. Đọc thẳng 12 bảng (11 mã + vnindex) từ PostgreSQL.
  2. Gộp thành long-format theo master calendar (lịch của VN-Index),
     forward-fill giá, fill 0 volume, đánh dấu is_trading.
     -> Xuất: merged_long_format.csv

  3. Với 11 mã (không tính vnindex), tính chỉ báo kỹ thuật (SMA, RSI,
     MACD, Bollinger, ATR, OBV, ADX, Stoch...) + các biến target
     (T+1, T+3, T+5), join thêm đặc trưng liên thị trường từ VN-Index.
     -> Xuất: all_stocks_features_1D.csv  (dùng để train model)

Yêu cầu: db_utils.py nằm cùng thư mục, và có sẵn .env
(dbname / user / password / host / port).
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings
from pathlib import Path

from db_utils import console, STOCK_LIST, INDEX_SYMBOL, ALL_SYMBOLS, load_raw, resample_to_1d

warnings.filterwarnings('ignore', category=UserWarning)

# Xác định thư mục StockINVES (thư mục cha của thư mục Database chứa file này)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_LONG_FORMAT = BASE_DIR / "merged_long_format.csv"
OUTPUT_FEATURES = BASE_DIR / "all_stocks_features_1D.csv"

PRICE_COLS = ['open', 'high', 'low', 'close', 'volume']
SYMBOL_SPECIFIC_COLS = ['return_1d_lag_1', 'vol_ma5_lag_1']


# ======================================================================
# PHẦN 1: GỘP OHLCV THÔ THEO MASTER CALENDAR -> merged_long_format.csv
# ======================================================================

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').drop_duplicates(subset='time', keep='last').reset_index(drop=True)
    return df


def build_long_format() -> pd.DataFrame:
    console.print(f"Đang tải '{INDEX_SYMBOL}' để làm master calendar...")
    vnindex_raw = load_raw(INDEX_SYMBOL)
    if vnindex_raw.empty:
        raise RuntimeError(f"Không load được bảng '{INDEX_SYMBOL}'. Kiểm tra kết nối DB.")
    master_calendar = pd.DatetimeIndex(sorted(clean_raw(vnindex_raw)['time'].unique()))
    console.print(f"👉 Master calendar: {len(master_calendar)} ngày, "
                  f"{master_calendar.min().date()} -> {master_calendar.max().date()}")

    all_frames = []
    for symbol in ALL_SYMBOLS:
        raw = load_raw(symbol)
        if raw.empty:
            console.print(f"⚠️  Bảng '{symbol}' rỗng, bỏ qua.")
            continue

        raw = clean_raw(raw).set_index('time')
        reindexed = raw.reindex(master_calendar)
        is_trading = reindexed['close'].notna().astype(int)

        reindexed[['open', 'high', 'low', 'close']] = reindexed[['open', 'high', 'low', 'close']].ffill()
        reindexed['volume'] = reindexed['volume'].fillna(0)
        reindexed['is_trading'] = is_trading
        reindexed['symbol'] = symbol.upper()
        reindexed.index.name = 'time'
        reindexed = reindexed.reset_index()

        all_frames.append(reindexed[['time', 'symbol', 'open', 'high', 'low',
                                      'close', 'volume', 'is_trading']])
        console.print(f"  {symbol.upper():8s}: {len(raw):5d} -> {len(reindexed):5d} rows "
                       f"({(is_trading == 0).sum()} ngày được fill)")

    long_df = pd.concat(all_frames, ignore_index=True)
    long_df = long_df.sort_values(['symbol', 'time']).reset_index(drop=True)
    return long_df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'is_trading']]


# ======================================================================
# PHẦN 2: CHỈ BÁO KỸ THUẬT + TARGET CHO TỪNG MÃ -> all_stocks_features_1D.csv
# ======================================================================

def add_vnindex_features(df: pd.DataFrame) -> pd.DataFrame:
    df['vnindex_return_1d'] = df['close_vnindex'].pct_change()
    df['vnindex_body_ratio'] = (df['close_vnindex'] - df['open_vnindex']) / (df['open_vnindex'] + 1e-8)
    df['vnindex_hl_ratio'] = (df['high_vnindex'] - df['low_vnindex']) / (df['low_vnindex'] + 1e-8)
    
    df['vnindex_return_1d_lag_1'] = df['close_vnindex'].pct_change().shift(1)
    df['vnindex_return_1d_lag_2'] = df['close_vnindex'].pct_change().shift(2)
    df['vnindex_return_1d_lag_3'] = df['close_vnindex'].pct_change().shift(3)
    
    close_vn_lag1 = df['close_vnindex'].shift(1)
    ma5_vn_lag1 = df['close_vnindex'].rolling(window=5).mean().shift(1)
    df['vnindex_ma5_dist_lag_1'] = (close_vn_lag1 - ma5_vn_lag1) / (ma5_vn_lag1 + 1e-8)
    
    vol_ma5 = df['volume_vnindex'].rolling(window=5).mean()
    df['vnindex_vol_ratio_ma5'] = (df['volume_vnindex'] - vol_ma5) / (vol_ma5 + 1e-8)
    df['vnindex_vol_roc_1d'] = df['volume_vnindex'].pct_change()
    return df


def add_technical_indicators(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    close = df[f"close_{suffix}"]
    open_price = df[f"open_{suffix}"]
    high = df[f"high_{suffix}"]
    low = df[f"low_{suffix}"]
    volume = df[f"volume_{suffix}"]

    df["Return_Close"] = close.pct_change()
    df["Body_Ratio"] = (close - open_price) / (open_price + 1e-8)
    df["High_Low_Ratio"] = (high - low) / (low + 1e-8)

    df[f'{suffix}_return_1d_lag_1'] = close.pct_change().shift(1)
    df[f'{suffix}_vol_ma5_lag_1'] = volume.rolling(window=5).mean().shift(1)

    df["Volume_Surge"] = volume / (df[f'{suffix}_vol_ma5_lag_1'] + 1e-8)
    df["Volume_ROC"] = volume.pct_change()

    sma_10 = ta.sma(length=10, close=close)
    sma_20 = ta.sma(length=20, close=close)
    df["RSI_14"] = ta.rsi(length=14, close=close)
    df["SMA_10_dist"] = (close - sma_10) / (sma_10 + 1e-8)
    df["SMA_20_dist"] = (close - sma_20) / (sma_20 + 1e-8)
    df["SMA_10_20_ratio"] = (sma_10 - sma_20) / (sma_20 + 1e-8)

    bb = ta.bbands(length=20, std=2, close=close)
    if bb is not None:
        bb_lower = bb.iloc[:, 0]
        bb_upper = bb.iloc[:, 2]
        bb_width = bb_upper - bb_lower
        df["BB_width_ratio"] = bb_width / (sma_20 + 1e-8)
        df["BB_position"] = (close - bb_lower) / (bb_width + 1e-8)

    macd = ta.macd(fast=12, slow=26, signal=9, close=close)
    if macd is not None:
        macd_line = macd.iloc[:, 0]
        macd_hist = macd.iloc[:, 1]
        macd_signal = macd.iloc[:, 2]
        
        df["MACD_Ratio"] = macd_line / (close + 1e-8)
        df["MACD_Signal_Ratio"] = macd_signal / (close + 1e-8)
        df["MACD_Hist_Ratio"] = macd_hist / (close + 1e-8)
        df["MACD_Hist_Diff"] = df["MACD_Hist_Ratio"].diff()

    atr_14 = ta.atr(high=high, low=low, close=close, length=14)
    df["ATR_14_ratio"] = atr_14 / (close + 1e-8)
    rolling_std_10 = close.rolling(window=10).std()
    df["Rolling_Std_10_Ratio"] = rolling_std_10 / (sma_10 + 1e-8)
    obv = ta.obv(close=close, volume=volume)
    obv_sma20 = obv.rolling(window=20).mean()
    df["OBV_Ratio"] = (obv - obv_sma20) / (obv_sma20.abs() + 1e-8)
    df["OBV_ROC"] = obv.diff() / (obv.shift(1).abs() + 1e-8)

    adx = ta.adx(high=high, low=low, close=close, length=14)
    if adx is not None:
        df["ADX_14"] = adx.iloc[:, 0]

    stoch = ta.stoch(high=high, low=low, close=close)
    if stoch is not None:
        df["Stoch_k"] = stoch.iloc[:, 0]
        df["Stoch_d"] = stoch.iloc[:, 1]

    df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    return df


def add_targets(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    close = df[f"close_{suffix}"]

    df["target_close_T1"] = close.shift(-1)
    df["target_return_T1"] = df["target_close_T1"] / close - 1
    df["target_trend_T1"] = np.where(df["target_return_T1"] > 0, 1, 0)

    df["target_close_T3"] = close.shift(-3)
    df["log_return_T3"] = np.log(df["target_close_T3"] / close)
    df["target_trend_T3"] = np.where(df["log_return_T3"] > 0.02, 1, 0)

    df["max_close_next_5_days"] = close.iloc[::-1].rolling(window=5, min_periods=1).max().iloc[::-1].shift(-1)
    df["log_return_max"] = np.log(df["max_close_next_5_days"] / close)
    df["target_trend_T5"] = np.where(df["log_return_max"] > 0.02, 1, 0)

    df["max_close_next_3_days_after_T2"] = close.iloc[::-1].rolling(window=3, min_periods=1).max().iloc[::-1].shift(-3)
    df["log_return_max_real"] = np.log(df["max_close_next_3_days_after_T2"] / close)
    df["target_trend_T5_Real"] = np.where(df["log_return_max_real"] > 0.02, 1, 0)
    return df


def build_features_for_symbol(symbol: str, vnindex_daily: pd.DataFrame) -> pd.DataFrame:
    raw = load_raw(symbol)
    if raw.empty:
        console.print(f"⚠️  Bảng '{symbol}' rỗng, bỏ qua.")
        return pd.DataFrame()

    daily = resample_to_1d(raw, symbol)
    df = daily.join(vnindex_daily, how="outer")
    before = len(df)
    
    # Tạo biến is_trading trước khi ffill (biết được ngày nào có giao dịch thật)
    df['is_trading'] = df[f'close_{symbol}'].notna().astype(int)
    
    # ffill các cột giá và fillna(0) cột volume
    price_cols = [f'open_{symbol}', f'high_{symbol}', f'low_{symbol}', f'close_{symbol}']
    df[price_cols] = df[price_cols].ffill()
    df[f'volume_{symbol}'] = df[f'volume_{symbol}'].fillna(0)
    
    # Drop các dòng VN-Index bị thiếu (giữ nguyên master calendar)
    vnindex_cols = [c for c in df.columns if 'vnindex' in c]
    df.dropna(subset=vnindex_cols, inplace=True)
    
    # Drop các dòng trước khi cổ phiếu niêm yết (giá ffill vẫn là NaN)
    df.dropna(subset=[f'close_{symbol}'], inplace=True)
    
    console.print(f"  {symbol.upper():8s}: full join {before} -> {len(df)} dòng khớp VN-Index")
    if df.empty:
        return pd.DataFrame()

    df.sort_index(inplace=True)
    df = add_vnindex_features(df)
    df = add_technical_indicators(df, symbol)
    df = add_targets(df, symbol)
    df.dropna(inplace=True)

    df.reset_index(inplace=True)
    df.insert(1, 'symbol', symbol.upper())

    # Đổi cột giá theo-mã (open_agr...) về generic (open...) để gộp nhiều
    # mã lại không bị NaN chéo cột.
    rename_map = {f"{c}_{symbol}": c for c in PRICE_COLS}
    rename_map.update({f"{symbol}_{c}": c for c in SYMBOL_SPECIFIC_COLS})
    df.rename(columns=rename_map, inplace=True)
    
    cols_to_drop = ['vol_ma5_lag_1', 'open_vnindex', 'high_vnindex', 'low_vnindex', 'close_vnindex', 'volume_vnindex']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df


def build_all_features() -> pd.DataFrame:
    console.print(f"Đang tải '{INDEX_SYMBOL}' để tính đặc trưng liên thị trường...")
    vnindex_raw = load_raw(INDEX_SYMBOL)
    vnindex_daily = resample_to_1d(vnindex_raw, "vnindex")

    all_frames = []
    for symbol in STOCK_LIST:
        feat_df = build_features_for_symbol(symbol, vnindex_daily)
        if not feat_df.empty:
            all_frames.append(feat_df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.insert(0, 'auto_id', range(1, 1 + len(combined)))
    return combined


# ======================================================================
# CHẠY TOÀN BỘ
# ======================================================================

if __name__ == "__main__":
    console.print("[bold magenta]===== BƯỚC 1/2: Gộp OHLCV thô =====[/]")
    long_df = build_long_format()
    long_df.to_csv(OUTPUT_LONG_FORMAT, index=False)
    console.print(f"[bold green]✅ Đã lưu:[/] {OUTPUT_LONG_FORMAT.resolve()} ({len(long_df)} dòng)\n")

    console.print("[bold magenta]===== BƯỚC 2/2: Tính chỉ báo kỹ thuật + target =====[/]")
    features_df = build_all_features()
    features_df.to_csv(OUTPUT_FEATURES, index=False)
    console.print(f"[bold green]✅ Đã lưu:[/] {OUTPUT_FEATURES.resolve()} "
                  f"({len(features_df)} dòng, {features_df['symbol'].nunique()} mã)\n")

    console.print("[bold cyan]===== HOÀN TẤT =====[/]")