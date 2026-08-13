"""
AUTO_RUN.PY — Chạy tự động predict_live.py (N=32 + bảng hiệu chỉnh lịch sử), ghi log có ngày tháng.
Dùng để đặt lịch chạy tự động (Windows Task Scheduler) — không cần đụng tay mỗi lần.

QUAN TRỌNG: script này KHÔNG tự cập nhật merged_long_format.csv — bạn đã có nguồn/script
riêng lo việc đó. Cần đặt lịch để AUTO_RUN.PY chạy SAU KHI dữ liệu đã được cập nhật xong
(vd cách nhau 15-30 phút), nếu không sẽ dự đoán trên dữ liệu cũ mà không biết.

Cách chạy thủ công để test trước: python auto_run.py
Cách đặt lịch tự động: xem hướng dẫn Windows Task Scheduler bên dưới file này.
"""

import os
import sys
import io
import contextlib
import traceback
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = os.path.join(LOG_DIR, f"predict_{timestamp}.log")

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print(f"===== AUTO RUN: {datetime.now()} =====\n")
            import predict_live
            predict_live.main()
    except Exception as e:
        buf.write(f"\n\n!!! LỖI KHI CHẠY: {e}\n")
        buf.write(traceback.format_exc())
    finally:
        output = buf.getvalue()
        # In ra console (nếu chạy tay) VÀ ghi vào file log (nếu chạy tự động, không ai xem console)
        print(output)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n[Đã ghi log: {log_path}]")


if __name__ == "__main__":
    main()

# ======================================================================
# HƯỚNG DẪN ĐẶT LỊCH TỰ ĐỘNG (Windows Task Scheduler) — làm 1 LẦN DUY NHẤT
# ======================================================================
# 1. Mở "Task Scheduler" (gõ tìm trong Start Menu)
# 2. Bên phải chọn "Create Basic Task..."
# 3. Đặt tên (vd "Predict Live Stock"), Next
# 4. Trigger: chọn "Daily" (hoặc "Weekly" nếu chỉ muốn chạy 1 lần/tuần theo REBALANCE_FREQ=5)
# 5. Đặt GIỜ CHẠY sau thời điểm script cập nhật CSV của bạn đã chạy xong (vd cách 15-30 phút,
#    để chắc chắn dữ liệu mới đã có sẵn trước khi model dự đoán)
# 6. Action: "Start a program"
#      Program/script:  đường dẫn tới python.exe của bạn
#                        (tìm bằng lệnh "where python" trong Command Prompt)
#      Add arguments:    auto_run.py
#      Start in:         đường dẫn thư mục chứa auto_run.py, best_ranknet.py, predict_live.py,
#                         merged_long_format.csv (BẮT BUỘC điền đúng, nếu không script sẽ báo
#                         lỗi "không tìm thấy file")
# 7. Finish. Có thể click chuột phải task vừa tạo -> "Run" để test ngay không cần chờ đến giờ.
# 8. Kết quả mỗi lần chạy sẽ nằm trong thư mục logs/predict_YYYY-MM-DD_HH-MM.log
#    -> mở file này bất cứ lúc nào để xem khuyến nghị, không cần mở terminal.
# ======================================================================