"""
check_anomalies_baseline11.py

Quét dị biệt toàn diện cho 11 mã baseline (FTS, HCM, ORS, SSI, VIX, BSI, CTS,
AGR, VDS, APG, TVS) — kết hợp 2 kiểu kiểm tra đã dùng trước đây cho 9 mã đa
ngành:

  1. Dị biệt SAU khi ngưng giao dịch dài (kiểu APG 2016-11-30: return >15%
     ngay sau chuỗi is_trading=0 kéo dài) — return_threshold=15%,
     gap_threshold=5 ngày.
  2. Dị biệt KHÔNG cần gap — cú nhảy giá >10% giữa 2 phiên giao dịch thật
     liên tiếp bất kỳ (nghi vấn chia tách/cổ tức bằng cổ phiếu/phát hành
     thêm chưa điều chỉnh, kiểu GEX đã phát hiện trước đây).

Chỉ liệt kê để bạn xác nhận thủ công — KHÔNG tự động vá bất cứ gì.

Chạy: python check_anomalies_baseline11.py
Yêu cầu: merged_long_format.csv cùng thư mục.
"""
import pandas as pd

TARGET_TICKERS = ['FTS', 'HCM', 'ORS', 'SSI', 'VIX', 'BSI', 'CTS', 'AGR', 'VDS', 'APG', 'TVS']
GAP_THRESHOLD = 5          # số ngày ngưng giao dịch liên tiếp
GAP_RETURN_THRESHOLD = 0.15    # 15% cho kiểu dị biệt (1)
NOGAP_RETURN_THRESHOLD = 0.10  # 10% cho kiểu dị biệt (2)


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
    return df[df['ticker'].isin(TARGET_TICKERS)].copy()


def check_gap_anomalies(df):
    """Kiểu (1): return bất thường ngay sau chuỗi ngưng giao dịch dài."""
    results = []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date').reset_index(drop=True)
        last_trading_close = None
        last_trading_date = None
        consecutive_non_trading = 0

        for _, row in group.iterrows():
            if row['is_trading'] == 0 or row['volume'] == 0 or pd.isna(row['close']):
                consecutive_non_trading += 1
                continue
            if last_trading_close is not None and consecutive_non_trading >= GAP_THRESHOLD:
                ret = (row['close'] / last_trading_close) - 1
                if abs(ret) > GAP_RETURN_THRESHOLD:
                    results.append({
                        'Loại': 'Sau ngưng GD dài',
                        'Mã': ticker,
                        'Ngày trước': last_trading_date.strftime('%Y-%m-%d'),
                        'Ngày sau': row['date'].strftime('%Y-%m-%d'),
                        'Số ngày ngưng GD': consecutive_non_trading,
                        'Return': f"{ret:.2%}",
                    })
            last_trading_close = row['close']
            last_trading_date = row['date']
            consecutive_non_trading = 0
    return pd.DataFrame(results)


def check_nogap_anomalies(df):
    """Kiểu (2): cú nhảy giá lớn giữa 2 phiên giao dịch thật liên tiếp bất kỳ."""
    df_trading = df[(df['is_trading'] == 1) & (df['volume'] > 0) & df['close'].notna()].copy()
    results = []
    for ticker, group in df_trading.groupby('ticker'):
        group = group.sort_values('date').reset_index(drop=True)
        group['prev_close'] = group['close'].shift(1)
        group['prev_date'] = group['date'].shift(1)
        group['ret_1d'] = (group['close'] / group['prev_close']) - 1

        for _, row in group.iterrows():
            if pd.isna(row['ret_1d']):
                continue
            if abs(row['ret_1d']) > NOGAP_RETURN_THRESHOLD:
                results.append({
                    'Loại': 'Không cần gap',
                    'Mã': ticker,
                    'Ngày trước': row['prev_date'].strftime('%Y-%m-%d'),
                    'Ngày sau': row['date'].strftime('%Y-%m-%d'),
                    'Close trước': row['prev_close'],
                    'Close sau': row['close'],
                    'Return': f"{row['ret_1d']:.2%}",
                })
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Đang tải dữ liệu...")
    df = load_data()

    print(f"\n{'=' * 80}")
    print(f"KIỂU (1): DỊ BIỆT SAU NGƯNG GIAO DỊCH ≥{GAP_THRESHOLD} NGÀY (return > {GAP_RETURN_THRESHOLD:.0%})")
    print(f"{'=' * 80}")
    gap_anomalies = check_gap_anomalies(df)
    if gap_anomalies.empty:
        print("✅ Không phát hiện dị biệt kiểu (1) trong 11 mã baseline.")
    else:
        pd.set_option('display.width', 140)
        print(gap_anomalies.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"KIỂU (2): NHẢY GIÁ >{NOGAP_RETURN_THRESHOLD:.0%} GIỮA 2 PHIÊN LIÊN TIẾP (không cần gap)")
    print(f"{'=' * 80}")
    nogap_anomalies = check_nogap_anomalies(df)
    if nogap_anomalies.empty:
        print("✅ Không phát hiện dị biệt kiểu (2) trong 11 mã baseline.")
    else:
        pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
        print(nogap_anomalies.to_string(index=False))

    print("\nLưu ý: KHÔNG tự động vá gì cả. Nếu có dị biệt, đối chiếu lịch sự kiện")
    print("chia tách/cổ tức/phát hành thêm thật (Cafef/Vietstock/HOSE) trước khi quyết")
    print("định có cần điều chỉnh giá hay loại bỏ, giống cách đã xử lý với GEX trước đây.")

    all_anomalies = pd.concat([gap_anomalies, nogap_anomalies], ignore_index=True)
    if not all_anomalies.empty:
        all_anomalies.to_csv("anomalies_baseline11.csv", index=False)
        print("\n✅ Đã lưu: anomalies_baseline11.csv")
