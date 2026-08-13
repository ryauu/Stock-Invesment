"""
HISTORICAL_CALIBRATION.PY — Hướng 2: bảng hiệu chỉnh lịch sử cho N=32 (RankNet).

RankNet chỉ học THỨ TỰ, không học % tăng/giảm. Script này KHÔNG tạo dự đoán mới —
nó dùng lại các checkpoint đã train sẵn từ 5 fold + covid_2020 (path/ranknet/checkpoints/)
để chạy lại dự đoán trên đúng các ngày lịch sử đó, rồi đối chiếu với target_raw THỰC TẾ
(đã biết, vì là quá khứ) để trả lời: "trong lịch sử, khi 1 mã được xếp hạng #k, nó thực
sự đã tăng/giảm trung bình bao nhiêu % trong 5 ngày sau đó?"

Đây là THỐNG KÊ MÔ TẢ (post-hoc), không phải một N mới, không dùng để tune lại model.
Chạy 1 lần, kết quả lưu vào rank_calibration.csv để predict_live.py đọc và hiển thị.

Yêu cầu: các checkpoint path/ranknet/checkpoints/ranknet_fold_{1..5}.weights.h5 và
ranknet_crisis_covid_2020.weights.h5 phải đã tồn tại (từ các lần chạy best_ranknet.py trước đó).

    python historical_calibration.py
"""

import os
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from best_ranknet import (
    load_and_prepare_data, add_vnindex_context, add_features, get_trading_days,
    prepare_training_data_by_date, train_model_custom, build_lstm,
    predict_returns_for_date, nearest_trading_day,
    DATA_START, SEQUENCE_LENGTH, FEATURE_COLS, TARGET_HORIZON,
    WINSORIZE_QUANTILES, VAL_SPLIT, EXCLUDED_TICKERS,
    FOLDS_5, CRISIS_FOLDS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "path", "checkpoints")
OUTPUT_CSV = os.path.join(BASE_DIR, "rank_calibration.csv")


def get_model_and_scaler_for_fold(df_features, tickers_all, trading_days, train_end_marker, checkpoint_path):
    """Tái tạo scaler + load weights checkpoint đã có sẵn (không train lại vì checkpoint tồn tại)."""
    train_end = nearest_trading_day(train_end_marker, trading_days, 'before')
    train_days_list = trading_days[(trading_days >= DATA_START) & (trading_days <= train_end)].reset_index(drop=True)
    if len(train_days_list) > TARGET_HORIZON:
        effective_train_end = train_days_list.iloc[-TARGET_HORIZON - 1]
    else:
        effective_train_end = train_end

    train_mask = (df_features['date'] >= DATA_START) & (df_features['date'] <= effective_train_end)
    train_df = df_features[train_mask].copy()
    low = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[0])
    high = train_df['target_raw'].quantile(WINSORIZE_QUANTILES[1])
    train_df['target'] = train_df['target_raw'].clip(low, high)

    X_grouped, y_grouped, sample_dates = prepare_training_data_by_date(
        train_df, tickers_all,
        trading_days[(trading_days >= DATA_START) & (trading_days <= effective_train_end)],
        SEQUENCE_LENGTH
    )
    unique_dates = np.sort(np.unique(sample_dates))
    split_idx = int(len(unique_dates) * (1 - VAL_SPLIT))
    split_date = unique_dates[split_idx]

    feature_dim = X_grouped[0].shape[-1]
    model = build_lstm((SEQUENCE_LENGTH, feature_dim))

    if not os.path.exists(checkpoint_path):
        print(f"  ⚠️  Không thấy checkpoint {checkpoint_path} — bỏ qua fold này.")
        return None, None

    model, scaler = train_model_custom(model, X_grouped, y_grouped, sample_dates, split_date,
                                        checkpoint_path=checkpoint_path)
    return model, scaler


def collect_rank_vs_actual(df_features, tickers_all, trading_days, model, scaler, test_start_marker, test_end_marker):
    """Với mỗi ngày trong khoảng test, xếp hạng model rồi đối chiếu target_raw thực tế."""
    test_start = nearest_trading_day(test_start_marker, trading_days, 'after')
    test_end = nearest_trading_day(test_end_marker, trading_days, 'before')
    test_dates = trading_days[(trading_days >= test_start) & (trading_days <= test_end)]

    records = []
    for date in test_dates:
        preds = predict_returns_for_date(model, scaler, df_features, tickers_all, date, SEQUENCE_LENGTH)
        ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        for rank, (tic, _) in enumerate(ranked, start=1):
            row = df_features[(df_features['ticker'] == tic) & (df_features['date'] == date)]
            if row.empty or pd.isna(row['target_raw'].iloc[0]):
                continue  # chưa có đủ 5 ngày tương lai để biết kết quả thật (vd cuối dữ liệu)
            actual_return = row['target_raw'].iloc[0]
            records.append({'rank': rank, 'ticker': tic, 'date': date, 'actual_return_5d': actual_return})
    return records


def main():
    print("Đọc dữ liệu...")
    csv_path = os.path.join(BASE_DIR, "merged_long_format.csv")
    df = load_and_prepare_data(csv_path)
    tickers_all = sorted(df['ticker'].unique())
    tickers_all = [t for t in tickers_all if t not in EXCLUDED_TICKERS]
    df_features = add_vnindex_context(df)
    df_features = add_features(df_features)
    trading_days = get_trading_days(df)

    all_records = []

    print("\n===== Chạy lại dự đoán lịch sử trên 5 fold walk-forward =====")
    for i, (train_marker, test_start_marker, test_end_marker) in enumerate(FOLDS_5):
        checkpoint_path = f"{CHECKPOINT_DIR}/ranknet_fold_{i+1}.weights.h5"
        print(f"Fold {i+1}: load {checkpoint_path}")
        model, scaler = get_model_and_scaler_for_fold(df_features, tickers_all, trading_days, train_marker, checkpoint_path)
        if model is None:
            continue
        recs = collect_rank_vs_actual(df_features, tickers_all, trading_days, model, scaler,
                                       test_start_marker, test_end_marker)
        all_records.extend(recs)
        print(f"  -> thu thập {len(recs)} điểm dữ liệu (mã x ngày)")

    print("\n===== Chạy lại dự đoán lịch sử trên covid_2020 (crisis fold, OOS thật) =====")
    for train_marker, test_start_marker, test_end_marker, name, is_prereg in CRISIS_FOLDS:
        if name != 'covid_2020':
            continue  # crash_2026 không có provenance hợp lệ, không dùng để hiệu chỉnh
        checkpoint_path = f"{CHECKPOINT_DIR}/ranknet_crisis_{name}.weights.h5"
        print(f"{name}: load {checkpoint_path}")
        model, scaler = get_model_and_scaler_for_fold(df_features, tickers_all, trading_days, train_marker, checkpoint_path)
        if model is None:
            continue
        recs = collect_rank_vs_actual(df_features, tickers_all, trading_days, model, scaler,
                                       test_start_marker, test_end_marker)
        all_records.extend(recs)
        print(f"  -> thu thập {len(recs)} điểm dữ liệu (mã x ngày)")

    if not all_records:
        print("\n⚠️  Không thu thập được dữ liệu nào — kiểm tra lại đường dẫn checkpoint.")
        return

    rec_df = pd.DataFrame(all_records)

    print(f"\n===== BẢNG HIỆU CHỈNH: Thứ hạng model vs Return thực tế 5 ngày sau =====")
    print(f"(Tổng {len(rec_df)} điểm dữ liệu từ 5 fold walk-forward + covid_2020, KHÔNG dùng crash_2026)")
    summary = rec_df.groupby('rank')['actual_return_5d'].agg(
        mean='mean', std='std', median='median', win_rate=lambda x: (x > 0).mean(), n='count'
    ).reset_index()

    print(f"\n{'Rank':<6}{'Mean 5d':<12}{'Std':<12}{'Median':<12}{'Win rate':<12}{'N':<8}")
    for _, r in summary.iterrows():
        print(f"#{int(r['rank']):<5}{r['mean']*100:>+8.2f}%   {r['std']*100:>7.2f}%   "
              f"{r['median']*100:>+7.2f}%   {r['win_rate']*100:>7.1f}%    {int(r['n'])}")

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\nĐã lưu bảng hiệu chỉnh vào {OUTPUT_CSV} — predict_live.py sẽ tự đọc file này nếu có.")


if __name__ == "__main__":
    main()
