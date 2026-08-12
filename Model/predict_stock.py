# -*- coding: utf-8 -*-
"""
Dùng model Global đã train để dự đoán cho MỘT mã cụ thể.
Lấy window (mặc định 30 ngày) gần nhất của đúng mã đó, scale bằng scaler
đã fit lúc train (KHÔNG fit lại), rồi đưa vào model.
"""
import os
# Tắt cảnh báo của TensorFlow (0 = tất cả, 1 = ẩn INFO, 2 = ẩn INFO & WARNING, 3 = ẩn tất cả)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Tắt thông báo liên quan đến tối ưu hóa CPU oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
import joblib

# Tensorflow VÀ Keras phải được import SAU khi đã cấu hình os.environ ở trên
from tensorflow import keras

from data_prep import load_raw, engineer_ratio_features, get_feature_columns

# ... (Giữ nguyên phần code bên dưới của bạn) ...


def predict_latest(symbol, csv_path="all_stocks_features_1D.csv",
                    model_path="global_lstm_model.keras",
                    meta_path="model_metadata.joblib"):
    meta = joblib.load(meta_path)
    scaler = meta["scaler"]
    feat_cols = meta["feat_cols"]
    symbol_to_id = meta["symbol_to_id"]
    window = meta["window"]

    if symbol not in symbol_to_id:
        raise ValueError(
            f"Mã '{symbol}' không nằm trong tập mã đã dùng để train "
            f"({list(symbol_to_id.keys())}). Model chỉ dự đoán được cho các mã này."
        )

    df = load_raw(csv_path)
    df = engineer_ratio_features(df)
    g = df[df["symbol"] == symbol].sort_values("time").reset_index(drop=True)
    if len(g) < window:
        raise ValueError(f"Mã {symbol} không đủ {window} ngày dữ liệu để dự đoán.")

    last_window = g.iloc[-window:][feat_cols].values.astype(np.float32)
    last_window_scaled = scaler.transform(last_window).astype(np.float32)
    X = last_window_scaled[np.newaxis, ...]           # (1, window, n_features)
    sid = np.array([[symbol_to_id[symbol]]])          # (1, 1)

    model = keras.models.load_model(model_path)
    trend_prob, ret_pred = model.predict({"sequence_input": X, "symbol_input": sid}, verbose=0)

    last_date = g["time"].iloc[-1]
    print(f"Mã: {symbol} | Dữ liệu gần nhất: {last_date.date()}")
    print(f"Xác suất TĂNG (target_trend_T1) ngày kế tiếp: {trend_prob[0][0]:.4f}")
    print(f"Dự đoán target_return_T1 (ngày kế tiếp): {ret_pred[0][0]*100:.3f}%")
    return trend_prob[0][0], ret_pred[0][0]


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "VIX"
    predict_latest(sym)