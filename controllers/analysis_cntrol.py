import pandas as pd
from prophet import Prophet
from tqdm import tqdm
import numpy as np

from common_util import convert_quarter_to_date

# ==================================================
# 트렌드 분석(trend_analysis)
# - 매출 트렌드 분석 : 분기별 매출의 증가, 감소추세를 정량화
# - 점포 수 트렌드 분석 : 분기별 점포수의 증가, 감소추세를 정량화  
# ==================================================
def trend_analysis(df : pd.dataFrame, target_col : str):

    # Prophet을 통한 추세 분석 함수
    results = []

    # tqdm으로 진행률 확인
    for adstrd_cd, group in tqdm(df.groupby('행정동_코드')):
        group = group.sort_values('기준_년분기_코드')
        group['ds'] = group['기준_년분기_코드'].apply(convert_quarter_to_date)
        group['y'] = group[target_col]

        if len(group) < 4:  # 최소 분기 수 제한
            continue

        try:
            # Prophet 추세파악 
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False
            )

            model.fit(group[['ds', 'y']])
            forecast = model.predict(group[['ds']])

            # 추세(trend), time(분기) 추출
            trend = forecast['trend'].values
            time = group['ds'].map(pd.Timestamp.toordinal).values

            # 기울기 계산(최소제곱법으로 직선의 기울기 계산)
            slope = ((trend - trend.mean()) * (time - time.mean())).sum() / ((time - time.mean())**2).sum()

            results.append({
                '행정동_코드': adstrd_cd,
                '추세기울기': slope,
                'trend_min': trend.min(),
                'trend_max': trend.max(),
                'trend_diff': trend.max() - trend.min(),
            })

        except Exception as e:
            print(f"{adstrd_cd} 처리 중 오류 발생: {e}")

    return results


# ==================================================
# 클래스 분류
# - 클래스 1 : 매출액 증가추세, 점포수 증가추세 
# - 클래스 2 : 매출액 감소추세, 점포수 증가추세 
# - 클래스 3 : 매출액 감소추세, 점포수 감소추세 
# - 클래스 4 : 매출액 증가추세, 점포수 감소추세  
# ==================================================
def classify(row):
    if row['sales_slope'] > 0 and abs(row['store_slope']) > 0:
        return 'Class1'  # Class 1
    elif row['sales_slope'] < 0 and abs(row['store_slope']) > 0:
        return 'Class2'  # Class 2
    elif row['sales_slope'] < 0 and row['store_slope'] < 0:
        return 'Class3'  # Class 3
    else:
        return 'Class4'  # Class 4
    
