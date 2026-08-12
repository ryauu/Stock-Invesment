"""
check_raw_sharpe_candidates.py

Xếp hạng 6 mã ứng viên (BSI, CTS, AGR, VDS, APG, TVS) theo Sharpe ratio THÔ
(chưa qua LSTM, chưa qua risk overlay) — CHỈ trên dữ liệu TRAIN (trước fold 1,
trước 2018-12-21) để tránh mọi rò rỉ thông tin từ giai đoạn test/crisis.

Đây chỉ là công cụ THAM KHẢO để xếp hạng — không tự động loại mã nào.
Quyết định giữ/loại (nếu có) phải chốt TRƯỚC khi nhìn bảng kết quả, không phải sau.

Chạy: python check_raw_sharpe_candidates.py
Yêu cầu: db_utils.py cùng thư mục, .env đã cấu hình kết nối DB.
"""
import numpy as np
import pandas as pd
from db_utils import console, load_raw

CANDIDATE_TICKERS = ['bsi', 'cts', 'agr', 'vds', 'apg', 'tvs','fts', 'hcm', 'ors', 'ssi', 'vix','hdg','ree','cmg','fpt','msn','pnj']
TRAIN_END = '2018-12-21'  # đúng mốc train-end của Fold 1, tránh leakage sang test


def compute_raw_sharpe(symbols: list, train_end: str) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        raw = load_raw(symbol)
        if raw.empty:
            console.print(f"⚠️  Bảng '{symbol}' rỗng, bỏ qua.")
            continue

        raw = raw.copy()
        raw['time'] = pd.to_datetime(raw['time'])
        raw = raw.sort_values('time')

        # CHỈ lấy dữ liệu train (trước train_end) — không đụng test/crisis
        train_df = raw[raw['time'] <= pd.Timestamp(train_end)].copy()
        if len(train_df) < 30:
            console.print(f"⚠️  '{symbol}': quá ít dữ liệu train ({len(train_df)} dòng), bỏ qua.")
            continue

        train_df['return_1d'] = train_df['close'].pct_change()
        rets = train_df['return_1d'].dropna()

        mean_ret = rets.mean()
        std_ret = rets.std()
        sharpe_raw = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

        rows.append({
            'symbol': symbol.upper(),
            'n_train_days': len(rets),
            'mean_return_1d': mean_ret,
            'std_return_1d': std_ret,
            'sharpe_raw_train': sharpe_raw,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values('sharpe_raw_train', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
    console.print(f"[bold magenta]===== SHARPE THÔ (TRAIN-ONLY, trước {TRAIN_END}) — 6 MÃ ỨNG VIÊN =====[/]")
    result = compute_raw_sharpe(CANDIDATE_TICKERS, TRAIN_END)

    if result.empty:
        console.print("⚠️  Không có dữ liệu để tính.")
    else:
        console.print(result.to_string(index=False))
        console.print("\n[bold yellow]Lưu ý: đây chỉ là bảng xếp hạng tham khảo — không tự loại mã nào.[/]")
        console.print("[bold yellow]Nếu muốn lọc, hãy đặt ngưỡng TRƯỚC khi nhìn bảng này lần sau.[/]")

    result.to_csv("raw_sharpe_candidates.csv", index=False)
    console.print("\n✅ Đã lưu: raw_sharpe_candidates.csv")
