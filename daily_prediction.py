"""
daily_prediction.py

Dự đoán return 5 ngày tới (TARGET_HORIZON=5) cho 11 mã baseline, dùng ĐÚNG
config checkpoint N=22 đã kiểm định trong quá trình ablation:
  - 13 feature ratio-based gốc + vnindex_return_1d_lag_1 (14 feature)
  - LSTM 3x64 units, sequence 20, L2=1e-4
  - Universe: FTS, HCM, ORS, SSI, VIX, BSI, CTS, AGR, VDS, APG, TVS

CHU KỲ: train 1 lần/ngày, cache model+scaler theo ngày (model_YYYYMMDD.keras,
scaler_YYYYMMDD.pkl). Chạy nhiều lần trong cùng 1 ngày sẽ dùng lại model đã
train, không train lại.

⚠️ QUAN TRỌNG — ĐỌC TRƯỚC KHI DÙNG:
Checkpoint N=22 có DSR=0.7174, CHƯA vượt ngưỡng 0.95 đã tự đặt ra trong suốt
quá trình pre-registration (28 lần thử nghiệm). Nghĩa là CHƯA đủ bằng chứng
thống kê để khẳng định model có edge thật, khác với việc chỉ "nhìn đẹp" do
đã thử nhiều cấu hình. Con số % dự đoán dưới đây là MỘT ĐIỂM DỰ ĐOÁN, không
phải khẳng định chắc chắn — script tự in kèm ngữ cảnh (độ lệch chuẩn return
lịch sử của từng mã) để đối chiếu quy mô dự đoán với biến động tự nhiên.
KHÔNG PHẢI LỜI KHUYÊN ĐẦU TƯ.

Chạy: python daily_prediction.py
Yêu cầu: merged_long_format.csv (dữ liệu mới nhất, đã cập nhật đến hôm qua),
tensorflow, scikit-learn cài sẵn.
"""
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ====================== CONFIG — ĐÚNG CHECKPOINT N=22 ======================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_START = '2016-07-05'
TARGET_HORIZON = 5
SEQUENCE_LENGTH = 20
WINSORIZE_QUANTILES = (0.01, 0.99)
VAL_SPLIT = 0.2

LSTM_UNITS = 64
LSTM_LAYERS = 3
DROPOUT = 0.2
L2_REG = 1e-4
LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 10

BASELINE_TICKERS = ['FTS', 'HCM', 'ORS', 'SSI', 'VIX', 'BSI', 'CTS', 'AGR', 'VDS', 'APG', 'TVS']
INDEX_SYMBOL = 'VNINDEX'

FEATURE_COLS = [
    'return_1d', 'return_5d_lag', 'return_10d_lag',
    'close_ma5_ratio', 'close_ma10_ratio', 'close_ma20_ratio',
    'ma5_ma20_ratio', 'volume_ratio',
    'volatility_5d', 'volatility_10d',
    'high_low_ratio', 'close_open_ratio',
    'rsi_14', 'vnindex_return_1d_lag_1',
]

MODEL_DIR = 'daily_model_cache'
os.makedirs(MODEL_DIR, exist_ok=True)


# ====================== DỮ LIỆU + FEATURE (giống hệt test_ratio.py) ======================
def load_data(filepath='merged_long_format.csv'):
    df = pd.read_csv(filepath)
    rename_map = {}
    if 'time' in df.columns:
        rename_map['time'] = 'date'
    if 'symbol' in df.columns:
        rename_map['symbol'] = 'ticker'
    df.rename(columns=rename_map, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].str.upper()
    df = df[(df['is_trading'] == 1) & (df['volume'] > 0)]
    df = df[df['date'] >= DATA_START]
    return df.sort_values(['ticker', 'date']).reset_index(drop=True)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 1.0 - (1.0 / (1.0 + rs))


def add_vnindex_context(df):
    vni = df[df['ticker'] == INDEX_SYMBOL][['date', 'close']].copy()
    vni = vni.sort_values('date')
    vni['vnindex_return_1d'] = vni['close'].pct_change()
    vni['vnindex_return_1d_lag_1'] = vni['vnindex_return_1d'].shift(1)
    return df.merge(vni[['date', 'vnindex_return_1d_lag_1']], on='date', how='left')


def add_features(df):
    df = df.copy()
    df['return_1d'] = df.groupby('ticker')['close'].pct_change()
    df['return_5d_lag'] = df.groupby('ticker')['close'].pct_change(5)
    df['return_10d_lag'] = df.groupby('ticker')['close'].pct_change(10)
    ma5 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    ma10 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
    ma20 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
    df['close_ma5_ratio'] = df['close'] / ma5
    df['close_ma10_ratio'] = df['close'] / ma10
    df['close_ma20_ratio'] = df['close'] / ma20
    df['ma5_ma20_ratio'] = ma5 / ma20
    df['volatility_5d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(5).std())
    df['volatility_10d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(10).std())
    volume_ma5 = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
    df['volume_ratio'] = df['volume'] / volume_ma5
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['rsi_14'] = df.groupby('ticker')['close'].transform(lambda x: compute_rsi(x, 14))
    df['target_raw'] = df.groupby('ticker')['close'].shift(-TARGET_HORIZON) / df['close'] - 1
    return df


def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(DROPOUT))
    for _ in range(LSTM_LAYERS - 2):
        model.add(LSTM(LSTM_UNITS, return_sequences=True))
        model.add(Dropout(DROPOUT))
    model.add(LSTM(LSTM_UNITS, return_sequences=False))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


def prepare_training_data(df_features, tickers, sequence_len=SEQUENCE_LENGTH):
    X_list, y_list, date_list = [], [], []
    for tic in tickers:
        sub = df_features[df_features['ticker'] == tic].sort_values('date')
        sub = sub.dropna(subset=FEATURE_COLS + ['target'])
        arr_f = sub[FEATURE_COLS].values
        arr_t = sub['target'].values
        arr_d = sub['date'].values
        for i in range(sequence_len, len(sub)):
            X_list.append(arr_f[i - sequence_len:i])
            y_list.append(arr_t[i])
            date_list.append(arr_d[i])
    return np.array(X_list), np.array(y_list).reshape(-1, 1), np.array(date_list)


def train_today_model(df_features, tickers, today_str):
    """Train model dùng TOÀN BỘ dữ liệu có đến ngày gần nhất — không giữ lại
    khoảng test riêng, vì đây là model sản xuất để dự đoán tương lai thật,
    không phải backtest."""
    train_df = df_features.copy()
    low = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[0])
    high = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[1])
    train_df['target'] = train_df['target_raw'].clip(low, high)

    X, y, sample_dates = prepare_training_data(train_df, tickers)
    print(f"  Tổng training samples: {len(X)}")

    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler.fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)

    unique_dates = np.sort(np.unique(sample_dates))
    split_date = unique_dates[int(len(unique_dates) * (1 - VAL_SPLIT))]
    train_mask = sample_dates < split_date
    val_mask = sample_dates >= split_date

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_val, y_val = X_scaled[val_mask], y[val_mask]

    model = build_lstm((SEQUENCE_LENGTH, X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE,
                                min_delta=1e-4, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                         callbacks=[early_stop], verbose=0)
    print(f"  Dừng ở epoch {len(history.history['loss'])}, val_loss={history.history['val_loss'][-1]:.6f}")

    model_path = os.path.join(MODEL_DIR, f'model_{today_str}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{today_str}.pkl')
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Đã lưu model: {model_path}")
    return model, scaler


def predict_next_5d(model, scaler, df_features, tickers):
    preds = {}
    for tic in tickers:
        sub = df_features[df_features['ticker'] == tic].sort_values('date')
        sub = sub.dropna(subset=FEATURE_COLS).tail(SEQUENCE_LENGTH)
        if len(sub) < SEQUENCE_LENGTH:
            preds[tic] = {'pred_return_5d': np.nan, 'last_date': None}
            continue
        seq = sub[FEATURE_COLS].values
        seq_scaled = scaler.transform(seq)
        pred = model.predict(seq_scaled[np.newaxis, ...], verbose=0)[0, 0]
        preds[tic] = {
            'pred_return_5d': pred,
            'last_date': sub['date'].iloc[-1],
            'last_close': sub['close'].iloc[-1] if 'close' in sub.columns else None,
        }
    return preds


if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y%m%d')
    model_path = os.path.join(MODEL_DIR, f'model_{today_str}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{today_str}.pkl')

    print("Đang tải dữ liệu...")
    df = load_data('merged_long_format.csv')
    df_features = add_vnindex_context(df)
    df_features = add_features(df_features)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"📦 Đã có model train hôm nay ({today_str}), dùng lại (không train mới).")
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print(f"🔧 Chưa có model hôm nay, đang train mới (dữ liệu đến ngày gần nhất)...")
        model, scaler = train_today_model(df_features, BASELINE_TICKERS, today_str)

    # Độ lệch chuẩn return 5-ngày lịch sử từng mã — để đối chiếu quy mô dự đoán
    hist_std = df_features.dropna(subset=['target_raw']).groupby('ticker')['target_raw'].std()

    preds = predict_next_5d(model, scaler, df_features, BASELINE_TICKERS)

    rows = []
    for tic, info in preds.items():
        rows.append({
            'Mã': tic,
            'Dữ liệu gần nhất': info['last_date'].date() if info['last_date'] is not None else None,
            'Dự đoán return 5 ngày': f"{info['pred_return_5d']:+.2%}" if not np.isnan(info['pred_return_5d']) else 'N/A',
            'Độ lệch chuẩn 5d lịch sử': f"{hist_std.get(tic, np.nan):.2%}",
        })
    result = pd.DataFrame(rows).sort_values('Dự đoán return 5 ngày', ascending=False)

    print("\n" + "=" * 70)
    print(f"DỰ ĐOÁN RETURN 5 NGÀY TỚI — {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 70)
    print(result.to_string(index=False))

    print("\n⚠️  LƯU Ý QUAN TRỌNG:")
    print("- DSR của model này (0.7174) CHƯA vượt ngưỡng 0.95 tự đặt ra khi kiểm định.")
    print("- 'Độ lệch chuẩn 5d lịch sử' cho biết mức biến động TỰ NHIÊN của mã đó —")
    print("  nếu |dự đoán| nhỏ hơn nhiều so với độ lệch chuẩn này, dự đoán gần như")
    print("  nằm trong nhiễu bình thường, không phải tín hiệu mạnh.")
    print("- Đây là MỘT điểm dự đoán, không phải lời khuyên đầu tư.")

    result.to_csv(f'prediction_{today_str}.csv', index=False)
    print(f"\n✅ Đã lưu: prediction_{today_str}.csv")
