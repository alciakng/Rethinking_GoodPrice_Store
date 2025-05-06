import pandas as pd

# 분기 코드 → 날짜형으로 변환 함수
def convert_quarter_to_date(code):
    year = int(str(code)[:4])
    quarter = int(str(code)[-1])
    month = (quarter - 1) * 3 + 1
    return pd.to_datetime(f"{year}-{month:02d}-01")