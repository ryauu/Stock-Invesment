import requests, os
def fetch_data(ticker:str):
    #Select multi API
    url = os.getenv(ticker.upper())
    if not url:
        raise RuntimeError(f"Don't have link API for stock {ticker}!")
    #Get API
    try:
        check = requests.get(url, timeout=10)
        check.raise_for_status()
        res = check.json()
    except requests.RequestException as error:
        raise RuntimeError(f"Yêu cầu API lỗi {error}")
    except ValueError as error:
        raise RuntimeError(f"Lỗi không đọc được JSON {error}")

    if not isinstance(res, dict) or "data" not in res:
        raise RuntimeError("API không hợp lệ")

    item = res.get("data")
    if not item:
        raise RuntimeError("Không có data")
    return item