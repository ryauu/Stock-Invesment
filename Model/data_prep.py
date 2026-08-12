# -*- coding: utf-8 -*-
"""
Chuẩn bị dữ liệu cho Global Model (LSTM đa mã cổ phiếu).

Xử lý 4 vấn đề đã nêu:
1. Chống rò rỉ dữ liệu -> chia Train/Val/Test theo MỐC THỜI GIAN cố định
   (không random split), áp dụng chung cho toàn bộ các mã.
2. Scaling Paradox -> chỉ dùng các cột tỷ lệ / đã chuẩn hoá theo giá
   (return, *_dist, *_ratio, BB_position, RSI, Stoch, ADX...), đồng thời
   tự tạo thêm vài cột tỷ lệ cho MACD, Rolling_Std (chia cho close) để
   loại bỏ ảnh hưởng thị giá tuyệt đối. Scaler (StandardScaler) chỉ được
   fit trên tập Train rồi áp dụng cho Val/Test (tránh leakage).
3. Sequence length không đều -> lọc bỏ mã không đủ số ngày để tạo ít
   nhất 1 cửa sổ trong mỗi giai đoạn Train/Val/Test.
4. Mất "tính nết" mã cổ biệt -> thêm cột symbol_id (categorical) để
   đưa vào Embedding layer trong model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 30          # số ngày nhìn lại (lookback)
TRAIN_END = "2023-12-31"  # train: <= mốc này
VAL_END = "2024-12-31"    # validation: sau TRAIN_END đến mốc này
# test: sau VAL_END đến hết dữ liệu (2025-01-01 .. 2026-06-30)

# Các cột KHÔNG dùng làm input vì mang thị giá/khối lượng tuyệt đối
# (đơn vị khác nhau giữa các mã -> gây lệch trọng số mô hình)
DROP_ABS_COLS = [
    "open", "high", "low", "close", "volume",
    "open_vnindex", "high_vnindex", "low_vnindex", "close_vnindex", "volume_vnindex",
    "vnindex_close_lag_1", "vnindex_close_lag_2", "vnindex_close_lag_3", "vnindex_ma5_lag_1",
    "SMA_10", "SMA_20", "BB_lower", "BB_upper", "OBV", "vol_ma5_lag_1",
    "ATR_14",  # đã có ATR_14_ratio thay thế
    "Rolling_Std_10",  # sẽ thay bằng Rolling_Std_10_ratio
    "MACD", "MACD_Signal", "MACD_Hist",  # sẽ thay bằng bản chia cho close
]

TARGET_COLS_ALL = [
    "target_close_T1", "target_return_T1", "target_trend_T1",
    "target_close_T3", "log_return_T3", "target_trend_T3",
    "max_close_next_5_days", "log_return_max", "target_trend_T5",
    "max_close_next_3_days_after_T2", "log_return_max_real", "target_trend_T5_Real",
]

# Target dùng để train trong bài này: dự đoán 1 ngày tới
TARGET_RETURN = "target_return_T1"
TARGET_TREND = "target_trend_T1"


def load_raw(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)
    return df


def engineer_ratio_features(df):
    """Tạo thêm các cột tỷ lệ (chuẩn hoá theo giá) để thay cho cột tuyệt đối."""
    df = df.copy()
    eps = 1e-9
    df["MACD_ratio"] = df["MACD"] / (df["close"] + eps)
    df["MACD_Signal_ratio"] = df["MACD_Signal"] / (df["close"] + eps)
    df["MACD_Hist_ratio"] = df["MACD_Hist"] / (df["close"] + eps)
    df["Rolling_Std_10_ratio"] = df["Rolling_Std_10"] / (df["close"] + eps)
    return df


def get_feature_columns(df):
    """Danh sách cột feature cuối cùng: bỏ id/time/symbol, bỏ target, bỏ cột tuyệt đối."""
    exclude = set(["auto_id", "time", "symbol"]) | set(TARGET_COLS_ALL) | set(DROP_ABS_COLS)
    feat_cols = [c for c in df.columns if c not in exclude]
    return feat_cols


def time_split_masks(df):
    train_mask = df["time"] <= TRAIN_END
    val_mask = (df["time"] > TRAIN_END) & (df["time"] <= VAL_END)
    test_mask = df["time"] > VAL_END
    return train_mask, val_mask, test_mask


def build_windows_per_symbol(df_symbol, feat_cols, symbol_id, window=WINDOW_SIZE):
    """
    Sinh các cửa sổ trượt (X: window ngày feature, y_return, y_trend) CHỈ
    trong nội bộ 1 mã (không nối chắp giữa các mã).
    Nhãn ở đúng dòng cuối cửa sổ (đã được tính sẵn = biến động sang ngày t+1).
    """
    vals = df_symbol[feat_cols].values.astype(np.float32)
    y_ret = df_symbol[TARGET_RETURN].values.astype(np.float32)
    y_trend = df_symbol[TARGET_TREND].values.astype(np.float32)
    times = df_symbol["time"].values

    X, yr, yt, ts, sid = [], [], [], [], []
    n = len(df_symbol)
    for end in range(window - 1, n):
        start = end - window + 1
        X.append(vals[start:end + 1])
        yr.append(y_ret[end])
        yt.append(y_trend[end])
        ts.append(times[end])
        sid.append(symbol_id)
    if not X:
        return None
    return (np.stack(X), np.array(yr), np.array(yt), np.array(ts), np.array(sid))


def prepare_dataset(csv_path, window=WINDOW_SIZE, min_samples_per_split=5, verbose=True):
    df = load_raw(csv_path)
    df = engineer_ratio_features(df)
    # loại bỏ các dòng thiếu target (ví dụ vài dòng cuối chuỗi chưa có nhãn tương lai)
    df = df.dropna(subset=[TARGET_RETURN, TARGET_TREND]).reset_index(drop=True)

    feat_cols = get_feature_columns(df)

    symbols = sorted(df["symbol"].unique())
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    all_X, all_yr, all_yt, all_ts, all_sid, all_split = [], [], [], [], [], []

    dropped_symbols = []
    for sym in symbols:
        g = df[df["symbol"] == sym].sort_values("time").reset_index(drop=True)
        # ĐIỂM MẤU CHỐT: groupby('symbol') trước, tạo sliding window RIÊNG cho từng mã
        train_mask, val_mask, test_mask = time_split_masks(g)

        # Kiểm tra đủ dữ liệu để tạo >=1 cửa sổ ở mỗi giai đoạn (vấn đề #3)
        enough = True
        for m, name in [(train_mask, "train"), (val_mask, "val"), (test_mask, "test")]:
            if m.sum() < window + min_samples_per_split:
                enough = False
        if not enough:
            dropped_symbols.append(sym)
            continue

        res = build_windows_per_symbol(g, feat_cols, symbol_to_id[sym], window=window)
        if res is None:
            dropped_symbols.append(sym)
            continue
        X, yr, yt, ts, sid = res

        # gán nhãn split cho từng cửa sổ dựa trên mốc thời gian của NGÀY DỰ BÁO (ts)
        ts_dt = pd.to_datetime(ts)
        split_arr = np.full(len(ts_dt), "train", dtype=object)
        split_arr[(ts_dt > pd.Timestamp(TRAIN_END)) & (ts_dt <= pd.Timestamp(VAL_END))] = "val"
        split_arr[ts_dt > pd.Timestamp(VAL_END)] = "test"

        all_X.append(X)
        all_yr.append(yr)
        all_yt.append(yt)
        all_ts.append(ts)
        all_sid.append(sid)
        all_split.append(split_arr)

    if dropped_symbols and verbose:
        print(f"[Cảnh báo] Loại {len(dropped_symbols)} mã do không đủ dữ liệu cho cả 3 giai đoạn: {dropped_symbols}")

    X = np.concatenate(all_X, axis=0)
    yr = np.concatenate(all_yr, axis=0)
    yt = np.concatenate(all_yt, axis=0)
    ts = np.concatenate(all_ts, axis=0)
    sid = np.concatenate(all_sid, axis=0)
    split = np.concatenate(all_split, axis=0)

    if verbose:
        print(f"Tổng số mẫu (cửa sổ {window} ngày): {len(X)}")
        for s in ["train", "val", "test"]:
            print(f"  - {s}: {(split == s).sum()} mẫu")
        print(f"Số features/ngày: {X.shape[-1]} -> {feat_cols}")
        print(f"Số mã dùng để train: {len(symbol_to_id)} -> {list(symbol_to_id.keys())}")

    # --- Scaling: FIT CHỈ TRÊN TRAIN, áp dụng cho val/test (chống leakage) ---
    train_idx = split == "train"
    val_idx = split == "val"
    test_idx = split == "test"

    n_feat = X.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X[train_idx].reshape(-1, n_feat))

    def scale(arr):
        shp = arr.shape
        flat = arr.reshape(-1, n_feat)
        flat = scaler.transform(flat)
        return flat.reshape(shp).astype(np.float32)

    X_scaled = scale(X)

    data = {
        "X_train": X_scaled[train_idx], "sid_train": sid[train_idx],
        "yr_train": yr[train_idx], "yt_train": yt[train_idx],
        "X_val": X_scaled[val_idx], "sid_val": sid[val_idx],
        "yr_val": yr[val_idx], "yt_val": yt[val_idx],
        "X_test": X_scaled[test_idx], "sid_test": sid[test_idx],
        "yr_test": yr[test_idx], "yt_test": yt[test_idx],
        "ts_test": ts[test_idx],
        "feat_cols": feat_cols,
        "symbol_to_id": symbol_to_id,
        "scaler": scaler,
        "window": window,
    }
    return data


if __name__ == "__main__":
    d = prepare_dataset("/mnt/user-data/uploads/all_stocks_features_1D.csv")