import requests, os
#Nguồn
URL_VIX= os.getenv("VIX")
def VIX():
    try:
        check = requests.get(URL_VIX, timeout=10)
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
