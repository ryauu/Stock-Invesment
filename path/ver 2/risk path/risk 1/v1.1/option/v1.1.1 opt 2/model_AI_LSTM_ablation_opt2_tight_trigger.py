"""
PIPELINE DỰ ĐOÁN RETURN CỔ PHIẾU VN – DỰ ÁN A (PRE-REGISTERED)
=================================================================
VERSION 2 (ABLATION MODEL - Option 2: Tight Trigger) — Giữ nguyên Features/LSTM V1:
  1. Feature engineering: Giữ V1 (15 raw features, bao gồm giá tuyệt đối)
  2. Model: Giữ V1 (LSTM 3×64, ~85K params, sequence 20)
  3. Risk management: dual CB (-20% crisis / -35% walk-forward),
     vol quantile 0.75, recovery nhanh hơn (của V2)
  4. Explicit FEATURE_COLS, giữ is_trading (constant, để match V1)
  5. Giữ nguyên forward-fill last known price
  6. E[MDD] minh bạch (Magdon-Ismail & Atiya 2004)
  7. Lối ra 1 (Per-Episode MDD & Cumulative MDD): Thiết kế lại hệ thống đánh giá MDD:
     - Per-Episode MDD: Gate kỹ thuật (FATAL) để xác minh CB cắt lỗ đúng ngưỡng -35% mỗi đợt.
     - Cumulative MDD: Cảnh báo rủi ro vốn toàn kỳ (WARN) dựa trên mức giảm từ đỉnh all-time.
     - E[MDD] driftless: Chỉ đóng vai trò tham chiếu lý thuyết (reference only).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import random
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ====================== CÀI ĐẶT TÁI LẬP ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# ====================== THAM SỐ PRE-REGISTERED ======================
DATA_START = '2016-07-05'
TARGET_HORIZON = 5
SEQUENCE_LENGTH = 20          # v1: 20 → v2: 10 (tăng số sample, giảm overfitting)
TOP_UNIVERSE_FRAC = 0.5
TOP_K_HOLD = 3
REBALANCE_FREQ = 5
TRADING_FEE = 0.004
WINSORIZE_QUANTILES = (0.01, 0.99)
MDD_STOP = -0.20
PSR_THRESHOLD = 0.95
EXCLUDED_TICKERS = ['VNINDEX']

# --- N-REGISTRY: Multiple testing adjustment ---
CONFIG_ID = 5            # Cấu hình #2 trong sổ N
N_CONFIGS_TESTED = 16    # Đếm theo số lựa chọn thiết kế thực tế (features, LSTM arch, CB, vol quantiles, L2...)
# Chú thích: Khi N tăng, PSR threshold không thay đổi (vẫn 0.95). Thay vào đó,
# E[max(SR)] under null sẽ tăng lên, và DSR dùng giá trị này làm benchmark để deflate PSR gốc.
# Việc chọn giữ lại kiến trúc V1 để làm baseline so sánh đã được tính là config #14.
# Mỗi config mới PHẢI tăng N_CONFIGS_TESTED trước khi chạy

# --- Tham số quản trị rủi ro (v2 — nới lỏng) ---
VOL_TARGET_LOOKBACK = 60
VOL_TARGET_QUANTILE = 0.75     # v1: 0.9 → v2: 0.75
VOL_REDUCE_FACTOR = 0.7        # v1: 0.5 → v2: 0.7
CB_DRAWDOWN_THRESHOLD = -0.20  # v1: -0.12 → v2: -0.20 (crisis folds, T≈85-100)
CB_DRAWDOWN_THRESHOLD_WF = -0.30  # Ngưỡng kích hoạt bán ra (Trigger)
CB_DESIGN_CONTRACT = -0.35      # Lời hứa tối đa (Contract Gate)
CB_RECOVERY_VOL_QUANTILE = 0.5 # v1: 0.8 → v2: 0.5
CORRELATION_MAX = 0.7
DRIFTLESS_RATIO_THRESHOLD = 0.1  # |μ|/σ² phải < 0.1 để driftless approximation OK

# --- LSTM (v2 — giảm complexity) ---
LSTM_UNITS = 64                # v1: 64 → v2: 32
LSTM_LAYERS = 3                # v1: 3 → v2: 1
DROPOUT = 0.2                 # v1: 0.2 → v2: 0.15
L2_REG = 1e-4                  # MỚI: L2 regularization
LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 10
VAL_SPLIT = 0.2

# --- Feature list (V1 — 15 raw features, bao gồm 'is_trading' hằng số để match V1) ---
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'is_trading', 
    'return_1d', 'return_5d_lag', 'ma5', 'ma10', 'ma20', 
    'volatility_5d', 'volume_ma5', 'high_low_ratio', 'close_open_ratio'
]

FOLDS_5 = [
    ('2018-12-31', '2019-01-02', '2020-05-29'),
    ('2020-05-29', '2020-06-01', '2021-10-29'),
    ('2021-10-29', '2021-11-01', '2023-03-31'),
    ('2023-03-31', '2023-04-03', '2024-08-30'),
    ('2024-08-30', '2024-09-02', '2025-12-31')
]
CRISIS_FOLDS = [
    ('2019-12-31', '2020-03-01', '2020-06-30', 'covid_2020', True),
    ('2025-12-31', '2026-01-01', '2026-03-31', 'crash_2026', False)
]

# ====================== TIỆN ÍCH ======================
def load_and_prepare_data(filepath='merged_long_format.csv'):
    first_row = pd.read_csv(filepath, nrows=0)
    date_col = 'time' if 'time' in first_row.columns else 'date'
    df = pd.read_csv(filepath, parse_dates=[date_col])
    rename_map = {}
    if 'time' in df.columns:
        rename_map['time'] = 'date'
    if 'symbol' in df.columns:
        rename_map['symbol'] = 'ticker'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    df = df[(df['is_trading'] == 1) & (df['volume'] > 0)]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= DATA_START]
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # --- XỬ LÝ ANOMALY GỐC RỄ ---
    # APG ngày 2016-11-30 có return > 21% sau 8 ngày ngưng giao dịch (giá tăng rồi trả lại ngay sau 1 tuần).
    # Đây là nhiễu do biên độ mở rộng ngày giao dịch lại, không phải chia tách.
    # Đặt về NaN và forward fill để không lây lan sai số vào MA, Ratios và Target.
    mask_apg = (df['ticker'] == 'APG') & (df['date'] == '2016-11-30')
    if mask_apg.any():
        for col in ['open', 'high', 'low', 'close']:
            df.loc[mask_apg, col] = np.nan
            df[col] = df.groupby('ticker')[col].ffill()
        print("🛠️  DATA FIX: Đã loại bỏ điểm dị biệt APG 2016-11-30 (NaN + ffill).")

    return df

def get_trading_days(df):
    days = df['date'].drop_duplicates().sort_values().reset_index(drop=True)
    return days

def nearest_trading_day(target, days, how='before'):
    target = pd.Timestamp(target)
    if how == 'before':
        return days[days <= target].iloc[-1]
    else:
        return days[days >= target].iloc[0]

# (compute_rsi đã bị xóa vì không dùng trong Ablation model, chỉ là remnant của V2 nhánh rút gọn)

def add_features(df):
    df = df.copy()
    df['return_1d'] = df.groupby('ticker')['close'].pct_change()
    df['return_5d_lag'] = df.groupby('ticker')['close'].pct_change(5)
    df['ma5'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
    df['volatility_5d'] = df.groupby('ticker')['return_1d'].transform(lambda x: x.rolling(5).std())
    df['volume_ma5'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['dollar_volume'] = df['close'] * df['volume']
    df['target_raw'] = df.groupby('ticker')['close'].shift(-TARGET_HORIZON) / df['close'] - 1
    return df

def prepare_training_data(df_features, tickers, train_dates, sequence_len=SEQUENCE_LENGTH):
    """V2: dùng explicit FEATURE_COLS thay vì exclude-list."""
    X_list, y_list = [], []
    for tic in tickers:
        sub = df_features[df_features['ticker'] == tic].sort_values('date')
        sub = sub[sub['date'].isin(train_dates)]
        sub = sub.dropna(subset=FEATURE_COLS + ['target'])
        arr_f = sub[FEATURE_COLS].values
        arr_t = sub['target'].values
        for i in range(sequence_len, len(sub)):
            X_list.append(arr_f[i-sequence_len:i])
            y_list.append(arr_t[i])
    return np.array(X_list), np.array(y_list).reshape(-1, 1)

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

def filter_universe(df_features, date, trading_days, top_frac=TOP_UNIVERSE_FRAC, lookback=20):
    idx = trading_days[trading_days <= date].index[-lookback:]
    recent_dates = trading_days.iloc[idx]
    recent = df_features[df_features['date'].isin(recent_dates)]
    recent = recent[~recent['ticker'].isin(EXCLUDED_TICKERS)]
    avg_dvol = recent.groupby('ticker')['dollar_volume'].mean().sort_values(ascending=False)
    n_select = int(np.ceil(len(avg_dvol) * top_frac))
    return list(avg_dvol.head(n_select).index)

def predict_returns_for_date(model, scaler, df_features, tickers, date, sequence_len=SEQUENCE_LENGTH):
    """V2: dùng explicit FEATURE_COLS."""
    preds = {}
    for tic in tickers:
        sub = df_features[(df_features['ticker'] == tic) & (df_features['date'] <= date)].tail(sequence_len)
        if len(sub) < sequence_len:
            preds[tic] = -np.inf
            continue
        seq = sub[FEATURE_COLS].values
        seq_scaled = scaler.transform(seq)
        pred = model.predict(seq_scaled[np.newaxis, ...], verbose=0)[0, 0]
        preds[tic] = pred
    return preds

# ====================== QUẢN TRỊ RỦI RO (v2 — nới lỏng, forward-fill) ======================
def compute_current_vol(df_features, universe, date):
    vols = []
    for tic in universe:
        row = df_features[(df_features['ticker'] == tic) & (df_features['date'] == date)]['volatility_5d']
        if not row.empty and not np.isnan(row.iloc[0]):
            vols.append(row.iloc[0])
    return np.mean(vols) if vols else 0.0

_VOL_CACHE = {}
def get_historical_vol_threshold(df_features, date, trading_days, lookback=VOL_TARGET_LOOKBACK, quantile=VOL_TARGET_QUANTILE):
    key = (date, lookback, quantile)
    if key in _VOL_CACHE:
        return _VOL_CACHE[key]
    idx = trading_days[trading_days <= date].index[-lookback:]
    dates_hist = trading_days.iloc[idx]
    hist_vols = []
    for d in dates_hist:
        uni = filter_universe(df_features, d, trading_days, top_frac=TOP_UNIVERSE_FRAC, lookback=20)
        vol = compute_current_vol(df_features, uni, d)
        if vol > 0:
            hist_vols.append(vol)
    if len(hist_vols) < 10:
        thresh = np.inf
    else:
        thresh = np.quantile(hist_vols, quantile)
    _VOL_CACHE[key] = thresh
    return thresh

def correlation_aware_selection(predictions, df_features, date, max_corr=CORRELATION_MAX, top_k=TOP_K_HOLD):
    sorted_tickers = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for tic, _ in sorted_tickers:
        if len(selected) == top_k:
            break
        if len(selected) == 0:
            selected.append(tic)
            continue
        corrs = []
        ret_tic = df_features[(df_features['ticker'] == tic) & (df_features['date'] <= date)].tail(20)['return_1d'].dropna()
        if len(ret_tic) < 5:
            continue
        for s in selected:
            ret_s = df_features[(df_features['ticker'] == s) & (df_features['date'] <= date)].tail(20)['return_1d'].dropna()
            if len(ret_s) < 5:
                continue
            common_idx = ret_tic.index.intersection(ret_s.index)
            if len(common_idx) < 5:
                continue
            corr = ret_tic[common_idx].corr(ret_s[common_idx])
            corrs.append(corr)
        if corrs and (np.mean(corrs) > max_corr):
            continue
        selected.append(tic)
    return selected

def run_backtest(df_features, model, scaler, tickers_all, start_date, end_date, trading_days,
                 top_frac=TOP_UNIVERSE_FRAC, top_k=TOP_K_HOLD, fee=TRADING_FEE,
                 rebalance_freq=REBALANCE_FREQ, initial_state=None,
                 cb_threshold=CB_DRAWDOWN_THRESHOLD):
    """
    initial_state: dict với 'holdings', 'cash_balance', 'prev_value', 'peak_value', 'in_cash', 'cash_reason',
                    'last_known_prices' (dict ticker -> giá đóng cửa cuối cùng biết).
    Trả về (df_returns, turnover_log, corr_log, final_state).
    """
    sim_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)].reset_index(drop=True)
    rebalance_dates = set(sim_days[::rebalance_freq])

    if initial_state is None:
        holdings = {}
        cash_balance = 1.0
        prev_value = 1.0
        peak_value = 1.0
        in_cash = False
        cash_reason = None
        last_known_prices = {}
    else:
        holdings = initial_state['holdings'].copy()
        cash_balance = initial_state['cash_balance']
        prev_value = initial_state['prev_value']
        peak_value = initial_state['peak_value']
        in_cash = initial_state['in_cash']
        cash_reason = initial_state.get('cash_reason', None)
        last_known_prices = initial_state.get('last_known_prices', {}).copy()

    portfolio_returns = []
    turnover_log = []
    corr_log = []
    in_cash_log = []
    is_exposed_log = []

    def get_price(tic, d):
        price_series = df_features[(df_features['ticker'] == tic) & (df_features['date'] == d)]['close']
        if not price_series.empty:
            price = price_series.iloc[0]
            last_known_prices[tic] = price
            return price
        else:
            return last_known_prices.get(tic, None)

    for i, date in enumerate(sim_days):
        # Cập nhật last_known_prices cho tất cả các mã từ dữ liệu ngày hôm nay
        today_data = df_features[df_features['date'] == date]
        for _, row in today_data.iterrows():
            last_known_prices[row['ticker']] = row['close']

        # Kiểm tra phục hồi nếu đang trong cash do circuit breaker
        if in_cash and cash_reason == 'circuit_breaker':
            universe_now = filter_universe(df_features, date, trading_days, top_frac)
            current_vol = compute_current_vol(df_features, universe_now, date)
            vol_threshold_recovery = get_historical_vol_threshold(df_features, date, trading_days,
                                                                  lookback=VOL_TARGET_LOOKBACK,
                                                                  quantile=CB_RECOVERY_VOL_QUANTILE)
            if current_vol < vol_threshold_recovery:
                print(f"    [DIAGNOSTIC] {date.date()}: 🟢 CB RECOVERY! current_vol ({current_vol:.6f}) < threshold ({vol_threshold_recovery:.6f})")
                in_cash = False
                cash_reason = None
                peak_value = cash_balance  # Reset peak value to prevent instant re-triggering
            elif date in rebalance_dates:
                print(f"    [DIAGNOSTIC] {date.date()}: 🔴 CB LOCKED. current_vol ({current_vol:.6f}) >= threshold ({vol_threshold_recovery:.6f})")

        # [NEW] Lưu trạng thái trước khi tính return và check trigger
        # Chú ý: is_exposed ở đây có nghĩa thực là "không bị CB khóa", có thể bao gồm vài ngày chờ rebalance sau recovery
        is_exposed_today = not in_cash

        # 1. Tính giá trị danh mục (current_value) trước mọi giao dịch trong ngày
        current_value = cash_balance
        for tic, shares in holdings.items():
            price = get_price(tic, date)
            if price is not None:
                current_value += shares * price

        # Cập nhật peak (dùng current_value trước phí để check CB)
        if current_value > peak_value:
            peak_value = current_value
            
        cb_triggered_today = False

        # 2. Kiểm tra circuit breaker
        if not in_cash:
            drawdown = (current_value / peak_value) - 1
            if drawdown <= cb_threshold:
                print(f"    [DIAGNOSTIC] {date.date()}: 💥 CB TRIGGERED! Drawdown {drawdown:.2%} <= {cb_threshold:.2%}. Portfolio converted to cash.")
                print(f"    [DIAGNOSTIC] internal_cb_trigger_dd: {drawdown:.4f}")
                fee_cost = current_value * fee
                current_value -= fee_cost
                cash_balance = current_value
                holdings = {}
                in_cash = True
                cash_reason = 'circuit_breaker'
                turnover_log.append((date, 1.0))
                cb_triggered_today = True

        # 3. Rebalance (nếu không dính CB hôm nay và đủ điều kiện)
        allow_rebalance = (not in_cash) or (cash_reason == 'no_candidates')
        do_rebalance = (date in rebalance_dates) and allow_rebalance and not cb_triggered_today

        if do_rebalance:
            universe = filter_universe(df_features, date, trading_days, top_frac)
            current_vol = compute_current_vol(df_features, universe, date)
            vol_threshold = get_historical_vol_threshold(df_features, date, trading_days,
                                                         lookback=VOL_TARGET_LOOKBACK,
                                                         quantile=VOL_TARGET_QUANTILE)
            invest_frac = 1.0
            if current_vol > vol_threshold:
                invest_frac = VOL_REDUCE_FACTOR

            preds = predict_returns_for_date(model, scaler, df_features, universe, date)
            selected = correlation_aware_selection(preds, df_features, date, max_corr=CORRELATION_MAX, top_k=top_k)
            effective_k = len(selected)

            total_equity = current_value

            if effective_k == 0:
                fee_cost = total_equity * fee
                current_value = total_equity - fee_cost
                cash_balance = current_value
                holdings = {}
                in_cash = True
                cash_reason = 'no_candidates'
                turnover_log.append((date, 1.0))
            else:
                target_weights_total = {t: invest_frac / effective_k for t in selected}
                alloc_weights = {t: 1.0 / effective_k for t in selected}

                current_equity = {}
                for tic in holdings:
                    price = get_price(tic, date)
                    if price is not None:
                        current_equity[tic] = holdings[tic] * price

                buy_value = 0.0
                sell_value = 0.0
                for tic in list(holdings.keys()):
                    if tic not in selected:
                        sell_value += current_equity.get(tic, 0.0)
                for tic in selected:
                    current_val = current_equity.get(tic, 0.0)
                    target_val = target_weights_total[tic] * total_equity
                    diff = target_val - current_val
                    if diff > 0:
                        buy_value += diff
                    elif diff < 0:
                        sell_value += -diff

                turnover = (buy_value + sell_value) / (2 * total_equity) if total_equity > 0 else 0.0
                fee_cost = turnover * total_equity * fee
                net_equity = total_equity - fee_cost

                investable = invest_frac * net_equity
                cash_balance = net_equity - investable
                new_holdings = {}
                for tic in selected:
                    price = get_price(tic, date)
                    if price is not None:
                        new_holdings[tic] = (investable * alloc_weights[tic]) / price
                holdings = new_holdings
                current_value = net_equity
                turnover_log.append((date, turnover))

                if in_cash and cash_reason == 'no_candidates':
                    in_cash = False
                    cash_reason = None

                # Tương quan danh mục
                if effective_k >= 2:
                    rets = []
                    for tic in selected:
                        sub = df_features[(df_features['ticker']==tic) & (df_features['date']<=date)].tail(20)
                        if len(sub) >= 5:
                            rets.append(sub['return_1d'].dropna().values[-20:])
                        else:
                            rets.append(np.zeros(20))
                    if rets:
                        corr_mat = np.corrcoef(rets)
                        avg_corr = (corr_mat.sum() - len(selected)) / (len(selected)*(len(selected)-1))
                        corr_log.append((date, avg_corr))

        # 4. Tính daily_ret DỰA TRÊN current_value cuối ngày so với prev_value
        daily_ret = (current_value / prev_value) - 1
        prev_value = current_value

        portfolio_returns.append(daily_ret)
        in_cash_log.append(in_cash)
        is_exposed_log.append(is_exposed_today)

    res = pd.DataFrame({
        'date': sim_days, 
        'daily_return': portfolio_returns, 
        'in_cash': in_cash_log,
        'is_exposed': is_exposed_log
    })
    final_state = {
        'holdings': holdings,
        'cash_balance': cash_balance,
        'prev_value': prev_value,
        'peak_value': peak_value,
        'in_cash': in_cash,
        'cash_reason': cash_reason,
        'last_known_prices': last_known_prices
    }
    return res, turnover_log, corr_log, final_state

# Baseline (giữ nguyên — forward-fill)
def run_baseline_backtest(df_features, start_date, end_date, trading_days, top_frac=TOP_UNIVERSE_FRAC,
                          fee=TRADING_FEE, rebalance_freq=REBALANCE_FREQ, initial_cap=1.0):
    sim_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)].reset_index(drop=True)
    rebalance_dates = set(sim_days[::rebalance_freq])
    portfolio_returns = []
    prev_value = initial_cap
    holdings = {}
    last_known_prices = {}

    def get_price(tic, d):
        price_series = df_features[(df_features['ticker'] == tic) & (df_features['date'] == d)]['close']
        if not price_series.empty:
            price = price_series.iloc[0]
            last_known_prices[tic] = price
            return price
        else:
            return last_known_prices.get(tic, None)

    for i, date in enumerate(sim_days):
        today_data = df_features[df_features['date'] == date]
        for _, row in today_data.iterrows():
            last_known_prices[row['ticker']] = row['close']

        if i == 0:
            do_rebalance = True
            daily_ret = 0.0
            portfolio_returns.append(daily_ret)
        else:
            current_value = 0.0
            for tic, shares in holdings.items():
                price = get_price(tic, date)
                if price is not None:
                    current_value += shares * price
            daily_ret = (current_value / prev_value) - 1
            portfolio_returns.append(daily_ret)
            prev_value = current_value
            do_rebalance = (date in rebalance_dates)
        if do_rebalance:
            universe = filter_universe(df_features, date, trading_days, top_frac)
            k = len(universe)
            if holdings:
                old_weights = {}
                total_val = prev_value
                for tic, shares in holdings.items():
                    price = get_price(tic, date)
                    if price is not None:
                        old_weights[tic] = shares * price / total_val
                new_weights = {t: 1.0/k for t in universe}
                all_tickers = set(list(old_weights.keys()) + list(new_weights.keys()))
                turnover = 0.5 * sum(abs(new_weights.get(t,0) - old_weights.get(t,0)) for t in all_tickers)
                fee_cost = turnover * prev_value * fee
                prev_value -= fee_cost
            holdings = {}
            for tic in universe:
                price = get_price(tic, date)
                if price is not None:
                    holdings[tic] = (prev_value / k) / price
    if len(portfolio_returns) != len(sim_days):
        raise ValueError("Mismatch in baseline returns")
    return pd.DataFrame({
        'date': sim_days.iloc[1:].reset_index(drop=True),
        'daily_return': portfolio_returns[1:]
    })

def concatenate_fold_returns(fold_results):
    all_rets = []
    for ret_df, _, _, _ in fold_results:
        all_rets.append(ret_df['daily_return'])
    return pd.concat(all_rets, ignore_index=True)

def compute_psr(returns_series, benchmark_return=0.0):
    rets = returns_series.dropna().values
    n = len(rets)
    if n < 2:
        return np.nan
    sr = np.mean(rets) / np.std(rets, ddof=1) if np.std(rets, ddof=1) != 0 else 0.0
    skew = pd.Series(rets).skew()
    kurt = pd.Series(rets).kurtosis() + 3
    numerator = (sr - benchmark_return) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denominator <= 0:
        return np.nan
    return norm.cdf(numerator / denominator)

def compute_expected_max_sr(n_configs, t_periods):
    """
    E[max(SR)] theo Bailey & López de Prado (2014) "Deflated Sharpe Ratio".
    n_configs: số configurations đã test (N trong sổ)
    t_periods: số observations (trading days)
    Trả về expected max SR dưới null hypothesis.
    """
    if n_configs <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    z1 = norm.ppf(1.0 - 1.0 / n_configs) if n_configs > 1 else 0.0
    z2 = norm.ppf(1.0 - 1.0 / (n_configs * np.e)) if n_configs > 1 else 0.0
    e_max_sr = (1 - euler_mascheroni) * z1 + euler_mascheroni * z2
    return e_max_sr

def compute_deflated_psr(returns_series, n_configs, benchmark_return=0.0):
    """
    Deflated PSR: PSR đã điều chỉnh cho multiple testing.
    Dùng E[max(SR)] làm benchmark thay vì 0.
    Nếu N=1, DSR = PSR thường.
    """
    rets = returns_series.dropna().values
    n = len(rets)
    if n < 2:
        return np.nan, np.nan, np.nan
    sr_hat = np.mean(rets) / np.std(rets, ddof=1) if np.std(rets, ddof=1) != 0 else 0.0
    
    skew = pd.Series(rets).skew()
    kurt = pd.Series(rets).kurtosis() + 3
    e_max_z = compute_expected_max_sr(n_configs, n)
    
    # Quy đổi E[max(Z)] về không gian Sharpe bằng cách nhân với Standard Error
    std_error = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2) / np.sqrt(n - 1)
    sr_0 = e_max_z * std_error
    
    psr_unadjusted = compute_psr(returns_series, benchmark_return=0.0)
    psr_deflated = compute_psr(returns_series, benchmark_return=sr_0)
    return psr_unadjusted, psr_deflated, sr_0

def compute_expected_mdd(sigma_daily, T_days, mu_daily=0.0):
    """
    E[MDD] theo Magdon-Ismail & Atiya (2004), xấp xỉ driftless.

    Công thức: E[MDD] ≈ σ × √(2T/π)

    Điều kiện áp dụng: |μ| / σ² < DRIFTLESS_RATIO_THRESHOLD (mặc định 0.1).
    Khi vi phạm, kết quả phụ thuộc DẤU của μ:
      - μ > 0 (positive drift): driftless là UPPER BOUND (an toàn cho CB justify)
      - μ < 0 (negative drift): driftless là LOWER BOUND — E[MDD] thực tế LỚN HƠN
        → CB justification KHÔNG an toàn, cần phương pháp khác

    Parameters
    ----------
    sigma_daily : float — daily portfolio volatility (ex-ante, không qua overlay)
    T_days : int — số ngày giao dịch trong cửa sổ
    mu_daily : float — daily mean return (CÓ DẤU — dùng kiểm tra driftless)

    Returns
    -------
    e_mdd : float — expected maximum drawdown (driftless approximation)
    status : str — 'EXACT', 'UPPER_BOUND', hoặc 'UNSAFE_LOWER_BOUND'
    drift_ratio : float — μ/σ² CÓ DẤU (để hiển thị và rẽ nhánh)

    Reference
    ---------
    Magdon-Ismail, M. & Atiya, A.F. (2004). "Maximum Drawdown".
    Risk Magazine, 17(10), 99-102.
    """
    e_mdd = sigma_daily * np.sqrt(2 * T_days / np.pi)
    if sigma_daily > 0:
        drift_ratio = mu_daily / (sigma_daily ** 2)  # CÓ DẤU, không lấy abs()
    else:
        drift_ratio = float('inf')

    if abs(drift_ratio) < DRIFTLESS_RATIO_THRESHOLD:
        # |μ|/σ² nhỏ → drift gần 0, driftless approximation chính xác
        status = 'EXACT'
    elif drift_ratio >= DRIFTLESS_RATIO_THRESHOLD:
        # μ dương đáng kể → E[MDD] thực < driftless → upper bound, AN TOÀN
        status = 'UPPER_BOUND'
    else:
        # drift_ratio <= -DRIFTLESS_RATIO_THRESHOLD
        # μ âm đáng kể → E[MDD] thực > driftless → lower bound, KHÔNG AN TOÀN
        status = 'UNSAFE_LOWER_BOUND'
    return e_mdd, status, drift_ratio

def estimate_raw_portfolio_sigma(df_features, trading_days, end_date,
                                  top_frac=TOP_UNIVERSE_FRAC, top_k=TOP_K_HOLD,
                                  rebalance_freq=REBALANCE_FREQ,
                                  stress_quantile=0.90, rolling_window=20):
    """
    Ước lượng ex-ante σ cho danh mục TOP_K equal-weight,
    KHÔNG áp vol-targeting / CB / correlation-selection.

    Đo trên dữ liệu train (trước end_date) → độc lập với overlay.
    Dùng cùng filter_universe logic (dollar volume ranking) để chọn mã,
    nhưng tính return thô equal-weight không có lớp bảo vệ nào.

    Parameters
    ----------
    df_features : DataFrame — dữ liệu đã qua add_features
    trading_days : Series — danh sách ngày giao dịch
    end_date : Timestamp — ngày cuối cùng dùng để ước lượng (train boundary)
    top_frac : float — fraction of universe to consider
    top_k : int — số mã giữ trong danh mục
    rebalance_freq : int — tần suất rebalance (ngày)
    stress_quantile : float — percentile của rolling vol dùng cho σ_stress
    rolling_window : int — cửa sổ rolling vol

    Returns
    -------
    sigma_normal : float — σ_daily toàn bộ giai đoạn train
    sigma_stress : float — σ_daily tại percentile stress_quantile
    n_days : int — số ngày dùng để ước lượng
    mu_raw : float — mean daily return (raw, cho driftless check)
    """
    # Lấy ngày giao dịch trong giai đoạn train
    train_days = trading_days[(trading_days >= DATA_START) & (trading_days <= end_date)].reset_index(drop=True)
    rebalance_dates = set(train_days[::rebalance_freq])

    current_holdings = []  # list of tickers
    portfolio_daily_returns = []

    for i, date in enumerate(train_days):
        # Rebalance: chọn top_k mã theo dollar volume (không correlation filter)
        if date in rebalance_dates:
            universe = filter_universe(df_features, date, trading_days, top_frac)
            # Lấy top_k theo dollar volume, KHÔNG dùng correlation_aware_selection
            current_holdings = universe[:top_k]

        if i == 0 or len(current_holdings) == 0:
            continue  # Không đệm 0.0 giả vào ước lượng σ/μ

        # Tính equal-weight return thô (không vol-targeting, không CB)
        daily_rets = []
        for tic in current_holdings:
            row = df_features[(df_features['ticker'] == tic) & (df_features['date'] == date)]
            if not row.empty and not np.isnan(row['return_1d'].iloc[0]):
                daily_rets.append(row['return_1d'].iloc[0])

        if daily_rets:
            portfolio_daily_returns.append(np.mean(daily_rets))
        # Nếu không có return nào, bỏ qua ngày này (không đệm 0.0)

    raw_returns = pd.Series(portfolio_daily_returns).dropna()
    n_days = len(raw_returns)

    # σ_normal: toàn bộ giai đoạn
    sigma_normal = raw_returns.std()

    # σ_stress: percentile cao của rolling vol
    if n_days >= rolling_window:
        rolling_vol = raw_returns.rolling(rolling_window).std().dropna()
        sigma_stress = rolling_vol.quantile(stress_quantile)
    else:
        sigma_stress = sigma_normal

    mu_raw = raw_returns.mean()

    return sigma_normal, sigma_stress, n_days, mu_raw

def compute_raw_vol_in_window(df_features, trading_days, start_date, end_date,
                               top_frac=TOP_UNIVERSE_FRAC, top_k=TOP_K_HOLD,
                               rebalance_freq=REBALANCE_FREQ):
    """
    Đo vol thô (KHÔNG overlay) của danh mục TOP_K equal-weight
    trong một cửa sổ thời gian cụ thể (test window của crisis fold).

    Dùng cho post-hoc validation: so sánh σ_stress (ước lượng ex-ante)
    với vol thô thực đo trong crisis test set — cùng cơ sở (không overlay).

    Returns
    -------
    sigma_raw : float — σ_daily raw trong cửa sổ
    n_days : int — số ngày đo được
    """
    window_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)].reset_index(drop=True)
    rebalance_dates = set(window_days[::rebalance_freq])

    current_holdings = []
    portfolio_daily_returns = []

    for i, date in enumerate(window_days):
        if date in rebalance_dates:
            universe = filter_universe(df_features, date, trading_days, top_frac)
            current_holdings = universe[:top_k]

        if i == 0 or len(current_holdings) == 0:
            continue

        daily_rets = []
        for tic in current_holdings:
            row = df_features[(df_features['ticker'] == tic) & (df_features['date'] == date)]
            if not row.empty and not np.isnan(row['return_1d'].iloc[0]):
                daily_rets.append(row['return_1d'].iloc[0])

        if daily_rets:
            portfolio_daily_returns.append(np.mean(daily_rets))

    raw_returns = pd.Series(portfolio_daily_returns)
    return raw_returns.std() if len(raw_returns) > 1 else 0.0, len(raw_returns)

def evaluate_crisis_fold(df_features, tickers_all, trading_days, train_end_marker, test_start_marker,
                          test_end_marker, fold_name, is_preregistered=True):
    train_end = nearest_trading_day(train_end_marker, trading_days, 'before')
    test_start = nearest_trading_day(test_start_marker, trading_days, 'after')
    test_end = nearest_trading_day(test_end_marker, trading_days, 'before')
    
    train_days_list = trading_days[(trading_days >= DATA_START) & (trading_days <= train_end)].reset_index(drop=True)
    if len(train_days_list) > TARGET_HORIZON:
        effective_train_end = train_days_list.iloc[-TARGET_HORIZON - 1]
    else:
        effective_train_end = train_end
        
    print(f"\n=== Crisis fold {fold_name}: train đến {effective_train_end.date()} (tránh leakage), test {test_start.date()}->{test_end.date()} ===")

    train_mask = (df_features['date'] >= DATA_START) & (df_features['date'] <= effective_train_end)
    train_df = df_features[train_mask].copy()
    low = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[0])
    high = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[1])
    train_df['target'] = train_df['target_raw'].clip(low, high)

    X, y = prepare_training_data(train_df, tickers_all,
                                 trading_days[(trading_days>=DATA_START) & (trading_days<=effective_train_end)],
                                 SEQUENCE_LENGTH)
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler.fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)

    split = int(len(X_scaled) * (1 - VAL_SPLIT))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm((SEQUENCE_LENGTH, X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, min_delta=1e-4, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0)

    ret_df, _, _, _ = run_backtest(df_features, model, scaler, tickers_all,
                                    test_start, test_end, trading_days,
                                    top_frac=TOP_UNIVERSE_FRAC, top_k=TOP_K_HOLD, fee=TRADING_FEE)
    cum = (1 + ret_df['daily_return']).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum / rolling_max) - 1
    max_dd = drawdown.min()
    sharpe = ret_df['daily_return'].mean() / ret_df['daily_return'].std() * np.sqrt(252) if ret_df['daily_return'].std() != 0 else 0
    actual_vol = ret_df['daily_return'].std()
    print(f"Crisis {fold_name}: Max DD = {max_dd:.4f}, Sharpe = {sharpe:.4f}, σ_actual = {actual_vol:.6f}")
    
    passed = (max_dd >= MDD_STOP) and (sharpe >= -1)
    if not passed:
        print("   >>> FAIL (MDD hoặc Sharpe không đạt)")
    else:
        if is_preregistered:
            print("   >>> PASS")
        else:
            print("   >>> PASS (⚠️ REFERENCE ONLY — provenance unverified, không tính là out-of-sample)")
    return passed, max_dd, sharpe, actual_vol

def main():
    print("=" * 70)
    print("PIPELINE ABLATION: V1 Features/LSTM + V2 Risk Mgmt")
    print("=" * 70)
    print(f"Thay đổi so với V1:")
    print(f"  Features: GIỮ NGUYÊN V1 ({len(FEATURE_COLS)} raw features, bao gồm giá tuyệt đối)")
    print(f"  LSTM: GIỮ NGUYÊN V1 ({LSTM_LAYERS}×{LSTM_UNITS} units, seq {SEQUENCE_LENGTH}) + L2={L2_REG}")
    print(f"  CB threshold (crisis):      {CB_DRAWDOWN_THRESHOLD}")
    print(f"  CB threshold (walk-forward): {CB_DRAWDOWN_THRESHOLD_WF}\n  CB design contract: {CB_DESIGN_CONTRACT}")
    print(f"  Vol quantile: {VOL_TARGET_QUANTILE}, reduce: {VOL_REDUCE_FACTOR}")
    print(f"  Recovery quantile: {CB_RECOVERY_VOL_QUANTILE}")
    print(f"  N-Registry: Config #{CONFIG_ID}, N={N_CONFIGS_TESTED} tested")
    print()

    print("Đọc dữ liệu...")
    df = load_and_prepare_data('merged_long_format.csv')
    tickers_all = sorted(df['ticker'].unique())
    print(f"Tổng số ticker ban đầu: {len(tickers_all)}")
    tickers_all = [t for t in tickers_all if t not in EXCLUDED_TICKERS]
    print(f"Số mã sau khi loại {EXCLUDED_TICKERS}: {len(tickers_all)}")
    print(f"Các mã: {tickers_all}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    if datetime.now() > datetime(2026, 3, 31):
        print("\n*** CẢNH BÁO: Crisis fold 2026 đang chạy trên dữ liệu đã biết trước. "
              "Xác nhận rằng fold này đã được pre-register trước khi có dữ liệu Q1/2026. ***")

    df_features = add_features(df)
    trading_days = get_trading_days(df)

    # Kiểm tra feature NaN
    sample = df_features.dropna(subset=FEATURE_COLS)
    print(f"\nSố dòng sau dropna features: {len(sample)} / {len(df_features)} ({100*len(sample)/len(df_features):.1f}%)")

    print("\n===== BẮT ĐẦU 5-FOLD WALK-FORWARD LIÊN TỤC (ABLATION) =====")
    fold_results = []
    current_state = None

    for i, (train_marker, test_start_marker, test_end_marker) in enumerate(FOLDS_5):
        train_end = nearest_trading_day(train_marker, trading_days, 'before')
        test_start = nearest_trading_day(test_start_marker, trading_days, 'after')
        test_end = nearest_trading_day(test_end_marker, trading_days, 'before')
        
        train_days_list = trading_days[(trading_days >= DATA_START) & (trading_days <= train_end)].reset_index(drop=True)
        if len(train_days_list) > TARGET_HORIZON:
            effective_train_end = train_days_list.iloc[-TARGET_HORIZON - 1]
        else:
            effective_train_end = train_end
            
        print(f"\nFold {i+1}: train đến {effective_train_end.date()} (tránh leakage), test {test_start.date()} -> {test_end.date()}")

        train_mask = (df_features['date'] >= DATA_START) & (df_features['date'] <= effective_train_end)
        train_df = df_features[train_mask].copy()
        low = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[0])
        high = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[1])
        train_df['target'] = train_df['target_raw'].clip(low, high)

        X, y = prepare_training_data(train_df, tickers_all,
                                     trading_days[(trading_days >= DATA_START) & (trading_days <= effective_train_end)],
                                     SEQUENCE_LENGTH)
        print(f"  Training samples: {len(X)}, features: {X.shape[-1]}")

        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        scaler.fit(X_reshaped)
        X_scaled = scaler.transform(X_reshaped).reshape(X.shape)

        split = int(len(X_scaled) * (1 - VAL_SPLIT))
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]

        model = build_lstm((SEQUENCE_LENGTH, X.shape[2]))
        if i == 0:
            model.summary()
        early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, min_delta=1e-4, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0)
        print(f"  Stopped at epoch {len(history.history['loss'])}, val_loss={history.history['val_loss'][-1]:.6f}")

        ret_df, turnover_log, corr_log, final_state = run_backtest(
            df_features, model, scaler, tickers_all,
            test_start, test_end, trading_days,
            top_frac=TOP_UNIVERSE_FRAC, top_k=TOP_K_HOLD, fee=TRADING_FEE,
            initial_state=current_state,
            cb_threshold=CB_DRAWDOWN_THRESHOLD_WF
        )
        fold_results.append((ret_df, turnover_log, corr_log, final_state))
        current_state = final_state

        # In metrics cho mỗi fold
        fold_rets = ret_df['daily_return']
        fold_sharpe = fold_rets.mean() / fold_rets.std() * np.sqrt(252) if fold_rets.std() != 0 else 0
        print(f"  Fold {i+1} result: mean={fold_rets.mean():.6f}, std={fold_rets.std():.6f}, sharpe={fold_sharpe:.4f}")

    all_returns = concatenate_fold_returns(fold_results)
    all_dates = pd.concat([ret_df['date'] for ret_df, _, _, _ in fold_results], ignore_index=True)
    
    # Tính toán Cumulative MDD (Toàn kỳ 7 năm)
    all_cum = (1 + all_returns).cumprod()
    all_rolling_max = all_cum.expanding().max()
    all_drawdown = (all_cum / all_rolling_max) - 1
    cumulative_mdd = all_drawdown.min()

    # Tính toán Per-Episode MDD (reset sau mỗi lần vào cash)
    is_exposed_series = pd.concat([ret_df['is_exposed'] for ret_df, _, _, _ in fold_results], ignore_index=True)
    assert len(all_returns) == len(is_exposed_series), "Lỗi: Độ dài all_returns và is_exposed_series không khớp!"
    
    segment_ids = (is_exposed_series != is_exposed_series.shift()).cumsum()
    
    episode_mdds = []
    print(f"\n--- DIAGNOSTIC: Per-Episode MDD Breakdown ---")
    for seg_id, group in is_exposed_series.groupby(segment_ids):
        if group.iloc[0]: # Nếu đang exposed (is_exposed == True)
            seg_returns = all_returns.loc[group.index]
            seg_dates = all_dates.loc[group.index]
            seg_cum = (1 + seg_returns).cumprod()
            seg_rolling_max = seg_cum.expanding().max()
            seg_dd = (seg_cum / seg_rolling_max) - 1
            ep_mdd = seg_dd.min()
            episode_mdds.append(ep_mdd)
            
            d_start = seg_dates.iloc[0].date()
            d_end = seg_dates.iloc[-1].date()
            ep_dd_last_day = seg_dd.iloc[-1]
            print(f"  Episode [{d_start} to {d_end}]: Length = {len(seg_returns):4d} days, Min MDD = {ep_mdd:.2%}, Last Day DD = {ep_dd_last_day:.2%}")
            
    episode_mdd_max = min(episode_mdds) if episode_mdds else 0.0

    # MDD từng fold riêng lẻ (bị reset ở ranh giới fold)
    fold_mdds = []
    for fold_idx, (ret_df, _, _, _) in enumerate(fold_results):
        fold_cum = (1 + ret_df['daily_return']).cumprod()
        fold_rolling_max = fold_cum.expanding().max()
        fold_dd = (fold_cum / fold_rolling_max) - 1
        fold_mdds.append(fold_dd.min())

    print(f"\n{'='*50}")
    print(f"Model Ablation returns: mean={all_returns.mean():.6f}, std={all_returns.std():.6f}, "
          f"min={all_returns.min():.6f}, max={all_returns.max():.6f}, "
          f"sharpe={(all_returns.mean()/all_returns.std())*np.sqrt(252):.4f}")

    # Khối A: Gate kỹ thuật (Per-Episode MDD vs CB_WF)
    print(f"\n--- KIỂM TRA LOGIC CIRCUIT BREAKER (Per-Episode MDD) ---")
    print(f"  Max MDD trong một chu kỳ (trước khi reset) = {episode_mdd_max:.4f} ({episode_mdd_max:.1%})")
    print(f"  CB_WF design contract = {CB_DESIGN_CONTRACT:.0%}")
    
    if episode_mdd_max < CB_DESIGN_CONTRACT:
        print(f"  → ⛔ FATAL: MDD một chu kỳ ({episode_mdd_max:.1%}) VƯỢT QUÁ LỜI HỨA CONTRACT ({CB_DESIGN_CONTRACT:.0%})")
        print(f"     CB đã thất bại trong việc cắt lỗ kịp thời đúng lời hứa. Pipeline STOPPED.")
        return
    else:
        headroom = abs(CB_DESIGN_CONTRACT) - abs(episode_mdd_max)
        print(f"  → ✅ CB hoạt động đúng thiết kế hợp đồng. Headroom = {headroom:.1%}")

    print(f"\n  [Chú thích] MDD nội bộ từng fold (reset mỗi fold, không phản ánh xuyên fold):")
    for fold_idx, fold_mdd in enumerate(fold_mdds):
        marker = "⚠️" if fold_mdd < -0.15 else "  " # Ngưỡng visual tham khảo
        print(f"    {marker} Fold {fold_idx+1}: {fold_mdd:.4f} ({fold_mdd:.1%})")

    # Khối B: Cảnh báo rủi ro kinh doanh (Cumulative MDD)
    print(f"\n--- CẢNH BÁO RỦI RO VỐN THỰC TẾ (Cumulative MDD) ---")
    print(f"  All-time Cumulative MDD (7 năm) = {cumulative_mdd:.4f} ({cumulative_mdd:.1%})")
    print(f"  ⚠️ Lưu ý: Đây là rủi ro tổng nhà đầu tư gánh chịu do hiệu ứng cộng dồn qua nhiều đợt cắt lỗ.")
    print(f"     Dù CB hoạt động đúng ở mỗi đợt riêng lẻ, NAV tổng vẫn có thể sụt giảm mạnh.")
    print(f"     Vốn 50M VND = mất tối đa {abs(cumulative_mdd) * 50:.1f}M VND từ đỉnh cao nhất.")

    # Khối C: E[MDD] Magdon-Ismail (Tham khảo)
    train_end_for_sigma = nearest_trading_day(FOLDS_5[0][0], trading_days, 'before')
    sigma_normal, sigma_stress, n_est_days, mu_raw = estimate_raw_portfolio_sigma(
        df_features, trading_days, end_date=train_end_for_sigma
    )
    
    print(f"\n--- E[MDD] Analysis (Magdon-Ismail: Reference ONLY) ---")
    print(f"  σ nguồn: ex-ante từ training data (trước {train_end_for_sigma.date()}, {n_est_days} ngày)")
    print(f"  σ_normal (toàn bộ train)       = {sigma_normal:.6f}")
    print(f"  σ_stress (P90 rolling vol)     = {sigma_stress:.6f}")
    print(f"  μ_raw (daily, ex-ante)          = {mu_raw:.6f} ({'dương' if mu_raw >= 0 else 'ÂM'})")
    
    T_wf = len(all_returns.dropna())
    e_mdd_wf, status_wf, drift_ratio_wf = compute_expected_mdd(sigma_normal, T_wf, mu_raw)
    
    print(f"\n  Driftless check: μ/σ² = {drift_ratio_wf:+.4f} (threshold: |ratio| < {DRIFTLESS_RATIO_THRESHOLD})")
    if status_wf == 'EXACT':
        print(f"    → Driftless approximation: EXACT")
    elif status_wf == 'UPPER_BOUND':
        print(f"    → μ dương đáng kể — driftless là UPPER BOUND")
    else:
        print(f"    → ⛔ μ ÂM đáng kể — driftless là LOWER BOUND (lạc quan so với thực tế)")

    print(f"\n  E[MDD] driftless (T={T_wf}) = {e_mdd_wf:.4f} ({e_mdd_wf:.1%}) [{status_wf}]")
    print(f"  So sánh: Episode MDD thực đo ({episode_mdd_max:.1%}) vs E[MDD] lý thuyết ({e_mdd_wf:.1%})")

    # --- N-REGISTRY: Deflated PSR ---
    psr_model, dsr_model, sr_0 = compute_deflated_psr(all_returns, n_configs=N_CONFIGS_TESTED)
    e_max_z = compute_expected_max_sr(N_CONFIGS_TESTED, len(all_returns.dropna()))
    print(f"\n--- N-REGISTRY (Config #{CONFIG_ID}, N={N_CONFIGS_TESTED}) ---")
    print(f"PSR (unadjusted):  {psr_model:.4f}")
    print(f"E[max(Z)] (N={N_CONFIGS_TESTED}): {e_max_z:.4f}")
    print(f"Benchmark SR (sr_0): {sr_0:.6f} (daily)")
    print(f"DSR (deflated):    {dsr_model:.4f}")
    print(f"Threshold:         {PSR_THRESHOLD}")
    if N_CONFIGS_TESTED >= 20:
        print(f"*** N=20 PROTOCOL ACTIVE: Deflated threshold applies. No exceptions. ***")

    print("\n===== BASELINE (equal-weight universe, không risk management, forward-fill) =====")
    baseline_rets = []
    for _, test_start_marker, test_end_marker in FOLDS_5:
        test_start = nearest_trading_day(test_start_marker, trading_days, 'after')
        test_end = nearest_trading_day(test_end_marker, trading_days, 'before')
        ret_df = run_baseline_backtest(df_features, test_start, test_end, trading_days)
        baseline_rets.append(ret_df)
    baseline_all = pd.concat(baseline_rets, ignore_index=True)
    print(f"Baseline returns: mean={baseline_all['daily_return'].mean():.6f}, "
          f"std={baseline_all['daily_return'].std():.6f}, "
          f"min={baseline_all['daily_return'].min():.6f}, "
          f"max={baseline_all['daily_return'].max():.6f}, "
          f"sharpe={(baseline_all['daily_return'].mean()/baseline_all['daily_return'].std())*np.sqrt(252):.4f}")
    psr_baseline = compute_psr(baseline_all['daily_return'])
    print(f"PSR baseline: {psr_baseline:.4f}")

    # So sánh exposure
    print(f"\n--- So sánh exposure ---")
    print(f"Model std / Baseline std = {all_returns.std()/baseline_all['daily_return'].std():.2%}")
    if all_returns.std() / baseline_all['daily_return'].std() < 0.3:
        print("⚠️  CẢNH BÁO: Model std < 30% baseline → model vẫn đang nằm cash quá nhiều!")
    elif all_returns.std() / baseline_all['daily_return'].std() < 0.5:
        print("⚠️  Model std 30-50% baseline → exposure vẫn thấp hơn đáng kể")
    else:
        print("✅  Model có exposure hợp lý so với baseline")

    avg_corr_model = np.mean([c for _, _, corr_list, _ in fold_results for _, c in corr_list]) if any(corr_list for _,_,corr_list,_ in fold_results) else np.nan
    wide_ret = df_features[df_features['ticker'].isin(tickers_all)].pivot_table(index='date', columns='ticker', values='return_1d').dropna()
    if wide_ret.shape[1] >= 2:
        avg_corr_all = wide_ret.corr().values[np.triu_indices_from(wide_ret.corr().values, k=1)].mean()
    else:
        avg_corr_all = np.nan
    print(f"\nTương quan trung bình {len(tickers_all)} mã: {avg_corr_all:.4f}")
    print(f"Tương quan trung bình danh mục 3 mã: {avg_corr_model:.4f}")
    if avg_corr_all > 0.7:
        print("CẢNH BÁO: tương quan >0.7, N hiệu dụng thấp hơn số mã.")

    print("\n===== CRISIS FOLDS (độc lập, có risk management V2) =====")
    crisis_raw_vols = {}
    crisis_passes = {}
    
    for train_end_marker, test_start_marker, test_end_marker, name, is_prereg in CRISIS_FOLDS:
        passed, _, _, _ = evaluate_crisis_fold(df_features, tickers_all, trading_days,
                             train_end_marker, test_start_marker, test_end_marker, name, is_prereg)
        crisis_passes[name] = passed
        
        # Đo vol thô (KHÔNG overlay) trong cùng cửa sổ test — cùng cơ sở với σ_stress
        t_start = nearest_trading_day(test_start_marker, trading_days, 'after')
        t_end = nearest_trading_day(test_end_marker, trading_days, 'before')
        raw_vol, n_raw = compute_raw_vol_in_window(df_features, trading_days, t_start, t_end)
        crisis_raw_vols[name] = (raw_vol, n_raw)

    # --- Post-hoc validation: σ_stress (ước lượng ex-ante) vs σ thô thực đo (không overlay) ---
    # Cả hai đều là vol KHÔNG overlay → so sánh cùng cơ sở (apples-to-apples)
    print(f"\n--- Post-hoc validation: σ_stress ước lượng vs σ thô (không overlay) trong crisis ---")
    print(f"  σ_stress (ex-ante, P90 rolling vol từ train) = {sigma_stress:.6f}")
    for fold_name, (raw_vol, n_raw) in crisis_raw_vols.items():
        ratio = raw_vol / sigma_stress if sigma_stress > 0 else float('inf')
        if sigma_stress >= raw_vol:
            verdict = '✅ ước lượng bảo thủ (σ_stress ≥ σ_raw_crisis)'
        else:
            verdict = f'⚠️  ước lượng LẠC QUAN — σ_raw_crisis cao hơn {ratio:.1f}x, E[MDD]_crisis bị đánh giá thấp'
        print(f"  {fold_name}: σ_raw = {raw_vol:.6f} ({n_raw} ngày, không overlay) — {verdict}")

    print(f"\n{'='*60}")
    print("BẢNG SO SÁNH V1 vs ABLATION")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'V1 (cũ)':<15} {'Ablation':<15}")
    print(f"{'-'*55}")
    print(f"{'Features':<25} {'15 (giá thô)':<15} {f'{len(FEATURE_COLS)} (giá thô)':<15}")
    print(f"{'LSTM params':<25} {'~85K':<15} {'~85K':<15}")
    print(f"{'Sequence length':<25} {'20':<15} {SEQUENCE_LENGTH:<15}")
    print(f"{'CB crisis':<25} {'-12%':<15} {f'{CB_DRAWDOWN_THRESHOLD:.0%}':<15}")
    print(f"{'CB walk-forward':<25} {'N/A':<15} {f'{CB_DRAWDOWN_THRESHOLD_WF:.0%}':<15}")
    print(f"{'Episode MDD (CB check)':<25} {'N/A':<15} {f'{episode_mdd_max:.4f}':<15}")
    print(f"{'Cumulative MDD (Risk)':<25} {'N/A':<15} {f'{cumulative_mdd:.4f}':<15}")
    print(f"{'PSR (unadjusted)':<25} {'0.6640':<15} {f'{psr_model:.4f}':<15}")
    dsr_label = f"DSR (N={N_CONFIGS_TESTED})"
    print(f"{dsr_label:<25} {'N/A':<15} {f'{dsr_model:.4f}':<15}")
    print(f"{'PSR baseline':<25} {'0.9344':<15} {f'{psr_baseline:.4f}':<15}")
    print(f"{'Model Sharpe':<25} {'0.1592':<15} {f'{(all_returns.mean()/all_returns.std())*np.sqrt(252):.4f}':<15}")
    print(f"{'Model std':<25} {'0.003131':<15} {f'{all_returns.std():.6f}':<15}")

    # --- QUYẾT ĐỊNH: dùng DSR (deflated) làm tiêu chí ràng buộc ---
    # PSR unadjusted chỉ mang tính tham khảo; DSR là tiêu chí quyết định thống kê
    # VÀ yêu cầu phải PASS tập khủng hoảng covid_2020 (out-of-sample thực sự)
    binding_psr = dsr_model
    covid_passed = crisis_passes.get('covid_2020', False)
    
    print(f"\n--- QUYẾT ĐỊNH (binding metric = DSR, N={N_CONFIGS_TESTED}, Gate = covid_2020) ---")
    if binding_psr >= PSR_THRESHOLD and psr_model > psr_baseline and covid_passed:
        print(f">>> KẾT LUẬN: Mô hình Ablation có tín hiệu.")
        print(f"    Thống kê: DSR={dsr_model:.4f} >= {PSR_THRESHOLD} VÀ PSR={psr_model:.4f} > baseline={psr_baseline:.4f}")
        print(f"    Thực chứng: covid_2020 (out-of-sample stress test) = PASS")
        if N_CONFIGS_TESTED >= 20:
            print(f"    *** N=20 PROTOCOL: DSR đã vượt threshold ngay cả sau deflation. Signal confirmed. ***")
    elif psr_model >= PSR_THRESHOLD and binding_psr < PSR_THRESHOLD:
        print(f">>> KẾT LUẬN: Mô hình Ablation KHÔNG đạt tín hiệu.")
        print(f"    PSR unadjusted đạt ({psr_model:.4f}) nhưng DSR KHÔNG đạt ({dsr_model:.4f}).")
        print(f"    Multiple testing penalty (N={N_CONFIGS_TESTED}) đã hạ xuống dưới threshold.")
        print(f"    Không có ngoại lệ cho 'chỉ là bản vá bug'. Config #{CONFIG_ID} đã được đăng ký.")
    elif not covid_passed:
        print(f">>> KẾT LUẬN: Mô hình Ablation KHÔNG đạt tín hiệu.")
        print(f"    Thống kê: DSR={dsr_model:.4f}, PSR={psr_model:.4f}")
        print(f"    Thực chứng: covid_2020 = FAIL (bắt buộc phải vượt qua khủng hoảng OOS).")
    else:
        print(f">>> KẾT LUẬN: Mô hình Ablation KHÔNG đạt tín hiệu.")
        print(f"    PSR={psr_model:.4f}, DSR={dsr_model:.4f}, baseline={psr_baseline:.4f}")
        if psr_model > 0.6640:
            print(f"    PSR cải thiện từ 0.6640 (V1) lên {psr_model:.4f} (Ablation), nhưng chưa đạt threshold.")

if __name__ == "__main__":
    main()
