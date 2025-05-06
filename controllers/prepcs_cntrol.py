import os
import pandas as pd
import plotly.express as px
import geopandas as gpd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# 폴더경로 
base_dir = os.path.dirname(os.path.dirname(__file__)) # 상위폴더
datset_dir = os.path.join(base_dir, 'data')

# ==========================
# 지도 경계데이터 정제함수(shp)
# ==========================
def prepcs_shp():
    # 행정구역 경계
    geometry = gpd.read_file(os.path.join(datset_dir, "sig.shp"),encoding="utf8")

    return geometry

# ==========================
# 매출액 데이터 정제 함수
# ==========================
def prepcs_sales(cond : str):
    sales = pd.read_csv(os.path.join(datset_dir, "sales.csv"), encoding='cp949')
    
    # 서비스업종코드 음식업 한정(한식음식점, 일식음식점, 양식음식점, 제과점, 치킨전문점) 
    sales_filter = sales[sales['서비스_업종_코드'].isin(['CS100001','CS100003','CS100004','CS100005','CS100007'])]

    # 행정동-기준년분기별 매출액 그룹화 
    # cond가 리스트인 경우: 컬럼 합 → 그룹 평균
    if isinstance(cond, list):
        sales_filter['기준_매출_합계'] = sales_filter[cond].sum(axis=1)
        sales_grouped = sales_filter.groupby(['행정동_코드', '기준_년분기_코드'])['기준_매출_합계'].mean().reset_index()
    else:
        # 단일 컬럼이면 그대로 평균
        sales_grouped = sales_filter.groupby(['행정동_코드', '기준_년분기_코드'])[[cond]].mean().reset_index()

    return sales_grouped

# ==========================
# 점포수 데이터 정제 함수
# ==========================
def prepcs_markets():
    market = pd.read_csv(os.path.join(datset_dir, "market.csv"), encoding='cp949')
    
    # 서비스업종코드 음식업 한정(한식음식점, 일식음식점, 양식음식점, 제과점, 치킨전문점) 
    market_filter = market[market['서비스_업종_코드'].isin(['CS100001','CS100003','CS100004','CS100005','CS100007'])]
    # 행정동-기준년분기별 매출액 그룹화 
    market_grouped = market_filter.groupby(['행정동_코드', '기준_년분기_코드'])['점포_수'].mean().reset_index()

    return market_grouped


# ===============================================================================================================
# 매출액과 점포수의 interaction weight 를 계산
# - 클래스 1,3 : 매출액, 점포수 동시 증가, 감소  
#   => interaction_weight = 매출액 영향도 * 점포수 영향도  
# - 클래스 2,4 : 매출액, 점포수 방향성 반대((증가, 감소),(감소, 증가)) 
#   => interaction_weight = 매출액 영향도 * np.exp(-abs(점포수 영향도)), 매출액 증가(감소) 분을 점포수 감소(증가) 분의 영향도로 감쇄
# ===============================================================================================================
def calc_interaction_weight(row):
    if row['class'] in [1, 3]:
        return row['sales_norm'] * row['markets_norm']
    elif row['class'] in [2, 4]:
        return row['sales_norm'] * np.exp(-abs(row['markets_norm']))
    else:
        return 0

# ==========================
# 행정구역-수치데이터 병합
# ==========================
def prepcs_derived_feature(df_base : pd.DataFrame):
    
    # interact_growth 계산
    scaler = MinMaxScaler()
    df_base['sales_norm'] = scaler.fit_transform(df_base[['sales_slope']])
    df_base['markets_norm'] = scaler.fit_transform(df_base[['markets_slope']])
    df_base['interact_growth'] = df_base.apply(calc_interaction_weight, axis=1)

    return df_base


