from datetime import datetime
import time
def in_trading_time():
    now = datetime.now()
    current = now.hour * 60 + now.minute

    morning_start = 9*60
    morning_end = 11*60 + 30

    afternoon_start = 13*60
    afternoon_end = 15*60
    
    return (morning_start<=current <= morning_end) or (afternoon_start <= current <=afternoon_end)
def until_trading_time(hour,minute):
    now = datetime.now()
    target = now.replace(hour=hour,minute=minute,second=0,microsecond=0)
    from datetime import timedelta
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()
def get_next_action():
    if in_trading_time():
        return True, 0

    now = datetime.now()

    # trước 9h → chờ mở cửa
    if now.hour < 9:
        t = (9, 0)

    # nghỉ trưa → chờ 13h
    elif now.hour < 13:
        t = (13, 0)

    # sau giờ → chờ ngày mai
    else:
        t = (9, 0)

    return False, until_trading_time(*t)
        