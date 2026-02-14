import requests
#Nguồn
URL= "https://assets.msn.com/service/Finance/QuoteSummary?apikey=0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM&activityId=698745e9-cc23-43d0-a1d7-1474cb056147&ocid=finance-utils-peregrine&cm=vi-vn&it=web&scn=ANON&ids=bxcyrw&intents=Quotes,QuoteDetails&wrapodata=false"
def fetch_quote():
    try:
        check= requests.get(URL, timeout=10)# kiểm tra lỗi nếu >10 giây sẽ ngắt
        check.raise_for_status()
        data= check.json()# Chuyển dữ liệu thành JSON
    except requests.RequestException as error:
        raise RuntimeError(f"Yêu cầu API lỗi{error}")
    except ValueError as error:
        raise RuntimeError(f"Lỗi ko đọc được file{error}")
    if len(data) == 0 or not isinstance(data,list):
        raise RuntimeError("API trống hoặc dữ liệu không hợp lệ")
    item = data[0]
    if "quote" not in item:
        raise RuntimeError("Lỗi dữ liệu")
    return item["quote"]