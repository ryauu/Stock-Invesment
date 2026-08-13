"""
PREDICT_LIVE.PY — Trích từ best_ranknet.py (N=32, RankNet) để chạy dự đoán SỐNG.
KHÔNG dùng để backtest/tune lại — chỉ dùng để lấy khuyến nghị mã cho ngày giao dịch tiếp theo.

Cách chạy: đặt file này CÙNG THƯ MỤC với best_ranknet.py và merged_long_format.csv
(merged_long_format.csv phải đã cập nhật đến ngày gần nhất bạn có).

    python predict_live.py

LƯU Ý QUAN TRỌNG:
- Model sẽ được TRAIN LẠI bằng TOÀN BỘ dữ liệu có đến ngày mới nhất (không dùng checkpoint
  cũ, vì checkpoint cũ chỉ train đến 2026-03-17 để giữ holdout test hợp lệ — ràng buộc đó
  không còn áp dụng khi dùng sống).
- Đây KHÔNG phải một N mới, không ghi vào sổ đăng ký N-Registry — chỉ là suy luận (inference)
  từ cấu hình N=32 đã chốt, không phải một thử nghiệm để so sánh/chọn lựa.
- Vì ngân sách 1.5 triệu không đủ mua TOP_K_HOLD=3 mã theo đúng thiết kế gốc (mỗi lô chẵn
  100 cổ phiếu có thể cần 1.5-4 triệu/mã), script này chọn ra mã đứng đầu bảng xếp hạng
  model MÀ bạn đủ tiền mua ít nhất 1 lô — không đại diện đúng cho hiệu suất đã backtest
  của N=32 (vốn dựa trên đa dạng hóa 3 mã).
"""

import os
import json
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ---- Trích các hàm/hằng số cần thiết từ best_ranknet.py (không chạy main() vì có if __name__ guard) ----
from best_ranknet import (
    load_and_prepare_data, add_vnindex_context, add_features, get_trading_days,
    prepare_training_data_by_date, train_model_custom, build_lstm,
    predict_returns_for_date, correlation_aware_selection,
    DATA_START, SEQUENCE_LENGTH, FEATURE_COLS, TARGET_HORIZON,
    WINSORIZE_QUANTILES, VAL_SPLIT, EXCLUDED_TICKERS, TOP_UNIVERSE_FRAC,
)

BUDGET = 1_500_000          # Ngân sách thật của bạn (VND)
LOT_SIZE = 100               # Lô chẵn tối thiểu tại VN

# ---- Checkpoint tái sử dụng — tránh train lại từ đầu mỗi lần chạy ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "path", "live", "predict_live.weights.h5")
STATE_PATH = os.path.join(BASE_DIR, "path", "live", "predict_live_state.json")
CACHE_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "daily_model_cache", "ai_predictions.json"))
FORCE_RETRAIN = False   # đặt True nếu muốn ép train lại từ đầu (vd sau khi đổi cấu hình)


def main():
    print("Đọc và chuẩn bị dữ liệu (đến ngày mới nhất có sẵn)...")
    csv_path = os.path.join(BASE_DIR, "merged_long_format.csv")
    df = load_and_prepare_data(csv_path)
    tickers_all = sorted(df['ticker'].unique())
    tickers_all = [t for t in tickers_all if t not in EXCLUDED_TICKERS]

    df_features = add_vnindex_context(df)
    df_features = add_features(df_features)
    trading_days = get_trading_days(df)

    latest_date = trading_days.max()
    days_stale = (pd.Timestamp.now().normalize() - latest_date).days
    weekday_today = pd.Timestamp.now().dayofweek  # 5=T7, 6=CN
    print(f"Ngày dữ liệu mới nhất: {latest_date.date()} (cách hôm nay {days_stale} ngày)")
    if weekday_today >= 5:
        print("  ℹ️  Hôm nay là T7/CN — thị trường VN không giao dịch, dữ liệu không có phiên mới là bình thường.")
    elif days_stale > 3:
        print(f"  ⚠️  CẢNH BÁO: dữ liệu có vẻ CŨ hơn bình thường ({days_stale} ngày) — "
              f"kiểm tra lại đã cập nhật merged_long_format.csv chưa trước khi tin kết quả.")

    # ---- Kiểm tra: đã train với đúng ngày dữ liệu này chưa? ----
    last_trained_date = None
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            last_trained_date = json.load(f).get("last_trained_date")

    skip_training = (
        not FORCE_RETRAIN
        and last_trained_date == str(latest_date.date())
        and os.path.exists(CHECKPOINT_PATH)
    )

    # ---- Train bằng TOÀN BỘ dữ liệu đến ngày mới nhất (không giữ holdout) ----
    train_mask = (df_features['date'] >= DATA_START) & (df_features['date'] <= latest_date)
    train_df = df_features[train_mask].copy()
    low = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[0])
    high = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[1])
    train_df['target'] = train_df['target_raw'].clip(low, high)

    print("Đang chuẩn bị dữ liệu training theo ngày...")
    X_grouped, y_grouped, sample_dates = prepare_training_data_by_date(
        train_df, tickers_all,
        trading_days[(trading_days >= DATA_START) & (trading_days <= latest_date)],
        SEQUENCE_LENGTH
    )
    unique_dates = np.sort(np.unique(sample_dates))
    split_idx = int(len(unique_dates) * (1 - VAL_SPLIT))
    split_date = unique_dates[split_idx]

    feature_dim = X_grouped[0].shape[-1]
    model = build_lstm((SEQUENCE_LENGTH, feature_dim))
    # Scaler PHẢI được fit trên đúng tập train (dates < split_date) để đồng nhất với lúc train_model_custom
    train_idx = [i for i, d in enumerate(sample_dates) if d < split_date]
    from sklearn.preprocessing import StandardScaler
    X_train_concat = np.concatenate([X_grouped[i] for i in train_idx], axis=0)
    scaler = StandardScaler().fit(X_train_concat.reshape(-1, feature_dim))

    if skip_training:
        print(f"[SKIP TRAIN] Dữ liệu chưa đổi kể từ lần train gần nhất ({last_trained_date}). "
              f"Load thẳng checkpoint có sẵn.")
        model.load_weights(CHECKPOINT_PATH)
    else:
        if os.path.exists(CHECKPOINT_PATH) and not FORCE_RETRAIN:
            print(f"Có dữ liệu mới kể từ {last_trained_date} -> {latest_date.date()}. "
                  f"WARM-START từ checkpoint cũ (nhanh hơn train từ đầu nhiều).")
            model.load_weights(CHECKPOINT_PATH)
        else:
            print("Không tìm thấy checkpoint cũ (hoặc FORCE_RETRAIN=True) -> train từ đầu.")
            
        # Truyền checkpoint_path=None để hàm KHÔNG tự động skip quá trình train
        model, _ = train_model_custom(model, X_grouped, y_grouped, sample_dates, split_date,
                                            checkpoint_path=None)
        
        # Tự lưu lại checkpoint sau khi đã train (cập nhật) xong
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        model.save_weights(CHECKPOINT_PATH)
        print(f"    [CHECKPOINT] Đã cập nhật model tốt nhất tại: {CHECKPOINT_PATH}")
        
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump({"last_trained_date": str(latest_date.date())}, f)

    # ---- Dự đoán cho ngày gần nhất ----
    print(f"\nDự đoán xếp hạng cho ngày {latest_date.date()}...")
    preds = predict_returns_for_date(model, scaler, df_features, tickers_all, latest_date, SEQUENCE_LENGTH)
    ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)

    print("\n--- BẢNG XẾP HẠNG MODEL ---")
    print("    (Model dùng RankNet — pairwise ranking loss, chỉ học THỨ TỰ tốt/xấu")
    print("     giữa các mã, KHÔNG dự đoán % tăng/giảm cụ thể. Điểm số tuyệt đối")
    print("     không có ý nghĩa % — chỉ thứ hạng dưới đây mới đáng tin.)")

    calib = None
    calib_path = os.path.join(BASE_DIR, "rank_calibration.csv")
    if os.path.exists(calib_path):
        calib = pd.read_csv(calib_path).set_index("rank")
        print("    (Kèm bảng hiệu chỉnh lịch sử: mã ở thứ hạng này trong quá khứ đã")
        print("     thực sự tăng/giảm trung bình bao nhiêu % sau 5 ngày — THAM KHẢO,")
        print("     không phải dự đoán cho lần này, chạy historical_calibration.py để tạo/cập nhật)")

    for rank, (tic, score) in enumerate(ranked, start=1):
        if calib is not None and rank in calib.index:
            row = calib.loc[rank]
            print(f"  #{rank:2d}  {tic}   [lịch sử: {row['mean']*100:+.2f}% TB, "
                  f"win rate {row['win_rate']*100:.0f}%, n={int(row['n'])}]")
        else:
            print(f"  #{rank:2d}  {tic}")

    # ---- Chọn mã cao nhất mà 1.5 triệu đủ mua ít nhất 1 lô ----
    print(f"\n--- LỌC THEO NGÂN SÁCH {BUDGET:,} VND (lô tối thiểu {LOT_SIZE} cổ phiếu) ---")
    chosen = None
    for tic, score in ranked:
        price_row = df_features[(df_features['ticker'] == tic) & (df_features['date'] == latest_date)]
        if price_row.empty:
            continue
        price = price_row['close'].iloc[0]
        lot_cost = price * LOT_SIZE
        affordable = lot_cost <= BUDGET
        print(f"  {tic}: giá={price:,.0f} VND, 1 lô={lot_cost:,.0f} VND -> "
              f"{'ĐỦ TIỀN' if affordable else 'không đủ'}")
        if affordable and chosen is None:
            chosen = (tic, price, lot_cost)

    print("\n--- KHUYẾN NGHỊ ---")
    if chosen:
        tic, price, lot_cost = chosen
        n_lots = int(BUDGET // lot_cost)
        print(f">>> Mua {n_lots} lô mã {tic} (giá {price:,.0f} VND/cp), "
              f"tổng ~{n_lots * lot_cost:,.0f} VND, còn dư {BUDGET - n_lots*lot_cost:,.0f} VND")
    else:
        print(">>> KHÔNG có mã nào đủ ngân sách 1 lô — giữ cash, chờ vốn nhiều hơn hoặc chờ giá giảm.")

    print("\n⚠️  Nhắc lại: đây là danh mục 1 MÃ (không đa dạng hóa như N=32 gốc, TOP_K_HOLD=3).")
    print("    Rủi ro tập trung cao hơn nhiều so với số liệu backtest/holdout đã có.")

    # ---- LƯU CACHE AI CHO Main.py ĐỌC ----
    cache_data = {
        "date": str(latest_date.date()),
        "rankings": {},
        "top_pick": None
    }
    
    for rank, (t, score) in enumerate(ranked, start=1):
        info = {"rank": rank, "score": float(score)}
        if calib is not None and rank in calib.index:
            row = calib.loc[rank]
            info["mean_5d"] = float(row['mean'])
            info["win_rate"] = float(row['win_rate'])
        cache_data["rankings"][t] = info
        
    if chosen:
        t, price, lot_cost = chosen
        n_lots = int(BUDGET // lot_cost)
        cache_data["top_pick"] = {
            "ticker": t,
            "price": float(price),
            "n_lots": n_lots,
            "total_cost": float(n_lots * lot_cost),
            "budget_remaining": float(BUDGET - n_lots*lot_cost)
        }
        
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)
    print(f"\n[CACHE] Đã lưu dự đoán AI vào {CACHE_PATH} để Main.py sử dụng.")


if __name__ == "__main__":
    main()