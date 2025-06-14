import json
import geojson
from matplotlib import cm
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests                      
import Levenshtein
import pydeck as pdk

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from mapboxgl.viz import ChoroplethViz
from mapboxgl.utils import df_to_geojson
from mapboxgl.utils import create_color_stops
from mapboxgl.utils import create_numeric_stops
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import zscore
from scipy.stats import skew

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from stargazer.stargazer import Stargazer
from util.common_util import load_clustered_geodataframe


# =======================================
# 0. 공통함수_정의
# - zscore_scale 
# - check_variable_skewness (왜도파악)
# - check_outliers_std (이상치파악)
# =======================================
 
# 로그변환 함수
def apply_log_transform(df, columns):
    """
    로그 또는 log1p 변환을 수행하고, 원본 컬럼명 뒤에 '_log'를 붙여 새로운 컬럼 생성
    :param df: 원본 DataFrame
    :param columns: 로그 변환할 컬럼 리스트
    :return: 변환된 DataFrame (in-place 아님)
    """
    df_transformed = df.copy()
    
    for col in columns:
        if (df_transformed[col] <= 0).any():
            print(col + " → log1p 적용")
            df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
        else:
            print(col + " → log 적용")
            df_transformed[f'{col}_log'] = np.log(df_transformed[col])
    
    return df_transformed


# zscore_scale
def apply_zscore_scaling(df, columns):
    """
    지정된 컬럼에 대해 Z-score 스케일링을 수행
    :param df: 원본 DataFrame
    :param columns: 스케일링할 컬럼 리스트
    :return: 스케일링된 DataFrame (in-place 아님)
    """
    df_scaled = df.copy()
    
    for col in columns:
        print(col + " → zscore scaling")
        df_scaled[col] = zscore(df_scaled[col])
    
    return df_scaled

mpl.rc('font', family='AppleGothic') # 한글깨짐 문제
def check_variable_skewness(df, threshold=1.0):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew_info = {}

    for col in numeric_cols:
        col_skew = skew(df[col].dropna())
        skew_info[col] = col_skew

    skew_df = pd.DataFrame.from_dict(skew_info, orient='index', columns=['Skewness'])
    skew_df = skew_df.sort_values('Skewness', ascending=False)

    # 기준선 표시
    def skew_label(value):
        if abs(value) < 0.5:
            return '∼ 대칭'
        elif abs(value) < threshold:
            return '약간 왜도'
        else:
            return '강한 왜도'

    skew_df['해석'] = skew_df['Skewness'].apply(skew_label)

    print("📊 변수별 왜도 요약:")
    print(skew_df)

    # 히스토그램 시각화
    fig, axs = plt.subplots(nrows=int(np.ceil(len(numeric_cols)/3)), ncols=3, figsize=(16, int(len(numeric_cols)*1.5)))
    axs = axs.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axs[i])
        axs[i].set_title(f"{col} (Skew: {skew_info[col]:.2f})")

    plt.tight_layout()
    plt.show()

    return skew_df

def check_outliers_std(df, threshold=3.0):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}

    for col in numeric_cols:
        data = df[col].dropna()
        mean = data.mean()
        std = data.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_ratio = len(outliers) / len(data)

        outlier_info[col] = {
            'Outliers': len(outliers),
            'Outlier_Ratio': outlier_ratio,
            'Mean': mean,
            'Std': std,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        }

    # 요약표 생성
    outlier_df = pd.DataFrame(outlier_info).T
    outlier_df = outlier_df.sort_values('Outlier_Ratio', ascending=False)

    print("📏 3σ 기준 이상치 탐지 결과:")
    print(outlier_df[['Outliers', 'Outlier_Ratio']])

    # 시각화
    fig, axs = plt.subplots(nrows=int(np.ceil(len(numeric_cols)/3)), ncols=3, figsize=(16, int(len(numeric_cols)*1.5)))
    axs = axs.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axs[i], color='lightcoral')
        axs[i].axvline(outlier_info[col]['Lower_Bound'], color='blue', linestyle='--', label='Lower Bound')
        axs[i].axvline(outlier_info[col]['Upper_Bound'], color='blue', linestyle='--', label='Upper Bound')
        axs[i].set_title(f"{col} (Outliers: {outlier_info[col]['Outliers']})")
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    return outlier_df

# 이상치제거 함수 
def drop_outlier_rows_std(df, cols, threshold=3.0):
    df_cleaned = df.copy()
    valid_cols = []

    # 유효한 수치형 컬럼만 필터링
    for col in cols:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            valid_cols.append(col)
        else:
            print(f"⚠️ 컬럼 '{col}'은 존재하지 않거나 수치형이 아닙니다. 제외됩니다.")

    if not valid_cols:
        print("❌ 이상치 검출할 수 있는 수치형 컬럼이 없습니다.")
        return df_cleaned

    # 이상치 마스크 생성
    outlier_mask = pd.DataFrame(False, index=df.index, columns=valid_cols)

    for col in valid_cols:
        data = df[col]
        mean = data.mean()
        std = data.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        outlier_mask[col] = (data < lower_bound) | (data > upper_bound)

    # 이상치 포함된 행 식별
    rows_with_outliers = outlier_mask.any(axis=1)
    num_outliers = rows_with_outliers.sum()

    print(f"🧹 이상치 포함 행 제거: {num_outliers}개 행 삭제됨")

    # 인덱스 유지한 채 이상치 행 제거 (reset_index 제거)
    df_no_outliers = df_cleaned[~rows_with_outliers]

    return df_no_outliers

# rmsle 계산함수 
def compute_rmsle_from_result(result, df):
    """
    PanelOLS 회귀 결과에서 RMSLE를 계산하는 함수
    
    Parameters:
    -----------
    result : linearmodels.panel.results.PanelEffectsResults
        PanelOLS의 회귀 결과 객체 (ex: result = model.fit())
    
    df : pd.DataFrame
        원본 데이터프레임 (fitted_values의 인덱스와 맞아야 함)
    
    Returns:
    --------
    rmsle : float
        로그 역변환 후 예측값과 실제값 간 RMSLE
    """
    # 종속변수명 추출
    y_var = result.model.dependent.vars[0]

    # 예측값 (log scale)
    y_pred_log = result.fitted_values

    # 실제값 (log scale)
    y_true_log = df.loc[y_pred_log.index, y_var]

    # 로그 역변환
    y_pred_actual = np.expm1(y_pred_log)
    y_true_actual = np.expm1(y_true_log)

    # 음수 방지 (RMSLE는 음수 입력 불가)
    y_pred_fixed = np.clip(y_pred_actual, 0, None)
    y_true_fixed = np.clip(y_true_actual, 0, None)

    # RMSLE 계산
    rmsle = np.sqrt(mean_squared_log_error(y_true_fixed, y_pred_fixed))
    return rmsle


def save_full_model_output(results, rmsle=None, filename="full_model_output.csv"):
    # --- 요약 통계 ---
    n_obs = results.nobs   
    n_periods = getattr(result, 'time_info', {}).get('total', None) 
    k = len(results.params)
    r2 = results.rsquared
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - k - 1)

    summary_dict = {
        "No. of Observations": n_obs,
        "No. of Time Periods": n_periods,
        "R-squared (Overall)": round(r2, 4),
        "Adjusted R-squared": round(adj_r2, 4),
        "R-squared (Within)": round(results.rsquared_within, 4),
        "R-squared (Between)": round(results.rsquared_between, 4),
        "Log-likelihood": round(results.loglik, 2),
        "F-statistic": round(results.f_statistic.stat, 2),
        "F-statistic (p-value)": round(results.f_statistic.pval, 4),
        "RMSLE": round(rmsle, 4) if rmsle is not None else "N/A"
    }

    df_summary = pd.DataFrame(summary_dict.items(), columns=["Metric", "Value"])

    # --- 계수 테이블 ---
    df_coef = pd.DataFrame({
        'Variable': results.params.index,
        'Coef.': results.params.values,
        'Std.Err.': results.std_errors.values,
        'T-Stat': results.tstats.values,
        'P-Value': results.pvalues.values,
        'CI Lower': results.conf_int().iloc[:, 0].values,
        'CI Upper': results.conf_int().iloc[:, 1].values,
    }).round(4)

    # --- 구분선 ---
    separator = pd.DataFrame([["---", "---"]], columns=df_summary.columns)

    # --- 합치기 ---
    df_combined = pd.concat([df_summary, separator, df_coef], ignore_index=True)

    # 저장
    df_combined.to_csv(filename, index=False, encoding='utf-8-sig')

# =======================================
# 1. 데이터 불러오기 및 병합
#   - Part1. Sale, Store, 그 외 데이터 병합 
#   - Part2. 착한가격업소 데이터 병합 
#   - Part3. 임대료 데이터 병합
#   - Part4. 상권-행정동 shp 매핑테이블 생성 
# =======================================
# ---------------------------------------
# Part1. Sale, Store, 그 외 데이터 병합 
# ---------------------------------------

# Sales_데이터 
Sales_2021 = pd.read_csv('./data/매출금액_2021_행정동.csv', encoding='utf-8')
Sales_2022 = pd.read_csv('./data/매출금액_2022_행정동.csv', encoding='utf-8')
Sales_2023 = pd.read_csv('./data/매출금액_2023_행정동.csv', encoding='utf-8')
Sales_2024 = pd.read_csv('./data/매출금액_2024_행정동.csv', encoding='cp949')

# 점포_데이터 
Stores_2021 = pd.read_csv('./data/점포_2021_행정동.csv', encoding='utf-8')
Stores_2022 = pd.read_csv('./data/점포_2022_행정동.csv', encoding='utf-8')
Stores_2023 = pd.read_csv('./data/점포_2023_행정동.csv', encoding='utf-8')
Stores_2024 = pd.read_csv('./data/점포_2024_행정동.csv', encoding='cp949')

# 기타_통제변수_데이터 
Indicators = pd.read_csv('./data/상권변화지표_행정동.csv', encoding='cp949')
Incomes = pd.read_csv('./data/소득금액_행정동.csv', encoding='cp949')
Apartments = pd.read_csv('./data/아파트단지수_행정동.csv', encoding='cp949')
Floatings = pd.read_csv('./data/유동인구수_행정동.csv', encoding='cp949')
Workers = pd.read_csv('./data/직장인구_행정동.csv', encoding='cp949')
Facilities = pd.read_csv('./data/집객시설수_행정동.csv', encoding='cp949')
Residents = pd.read_csv('./data/상주인구수_행정동.csv', encoding='cp949')

# 운영_영업_개월 차이 
Indicators['운영_영업_개월_차이'] = Indicators['운영_영업_개월_평균'] - Indicators['서울_운영_영업_개월_평균']
Indicators['폐업_영업_개월_차이'] = Indicators['폐업_영업_개월_평균'] - Indicators['서울_폐업_영업_개월_평균']


# 데이터 2023~'24년으로 한정해서 분석
# 매출 데이터 병합
Sales = pd.concat([Sales_2021, Sales_2022, Sales_2023, Sales_2024], ignore_index=True)

# 점포 데이터 병합
Stores = pd.concat([Stores_2021, Stores_2022, Stores_2023, Stores_2024], ignore_index=True)


# 기준_년분기_코드 필터 함수 정의
def filter_by_year(df):
    return df[df['기준_년분기_코드'].astype(str).str[:4].astype(int).between(2021, 2024)]

# 필터링 적용
Sales = filter_by_year(Sales)
Stores = filter_by_year(Stores)
Indicators = filter_by_year(Indicators)
Incomes = filter_by_year(Incomes)
Apartments = filter_by_year(Apartments)
Floatings = filter_by_year(Floatings)
Workers = filter_by_year(Workers)
Facilities = filter_by_year(Facilities)
Residents = filter_by_year(Residents)

# 필요한컬럼만 필터 
Sales= Sales[Sales['서비스_업종_코드'].isin(['CS100001','CS100002', 'CS100003', 'CS100004', 'CS100005', 'CS100008'])] 
Stores= Stores[Stores['서비스_업종_코드'].isin(['CS100001', 'CS100002', 'CS100003', 'CS100004', 'CS100005', 'CS100008'])] 

Sales_grouped = Sales.groupby(['기준_년분기_코드', '행정동_코드','행정동_코드_명'])['당월_매출_금액'].sum().reset_index()
Stores_grouped = Stores.groupby(['기준_년분기_코드', '행정동_코드','행정동_코드_명'])[['유사_업종_점포_수','개업_점포_수','폐업_점포_수']].sum().reset_index()

Stores_grouped.rename(columns={'유사_업종_점포_수' : '점포_수'},inplace=True)
Stores_grouped['개업_률'] = round(Stores_grouped['개업_점포_수'] / Stores_grouped['점포_수'],2)
Stores_grouped['폐업_률'] = round(Stores_grouped['폐업_점포_수'] / Stores_grouped['점포_수'],2)

Indicators = Indicators[['기준_년분기_코드','행정동_코드','상권_변화_지표','상권_변화_지표_명','운영_영업_개월_평균','폐업_영업_개월_평균','운영_영업_개월_차이','폐업_영업_개월_차이']]
Incomes = Incomes[['기준_년분기_코드','행정동_코드','월_평균_소득_금액','음식_지출_총금액','의료비_지출_총금액','교육_지출_총금액']]
Apartments = Apartments[['기준_년분기_코드','행정동_코드','아파트_단지_수','아파트_평균_시가']]
Floatings = Floatings[['기준_년분기_코드','행정동_코드','남성_유동인구_수','여성_유동인구_수','연령대_10_유동인구_수','연령대_20_유동인구_수','연령대_30_유동인구_수','연령대_40_유동인구_수','연령대_50_유동인구_수','연령대_60_이상_유동인구_수']]
Workers = Workers[['기준_년분기_코드','행정동_코드','남성_직장_인구_수','여성_직장_인구_수','연령대_10_직장_인구_수','연령대_20_직장_인구_수','연령대_30_직장_인구_수','연령대_40_직장_인구_수','연령대_50_직장_인구_수','연령대_60_이상_직장_인구_수']]
Facilities = Facilities[['기준_년분기_코드','행정동_코드','집객시설_수']]
Residents = Residents[['기준_년분기_코드','행정동_코드','총_상주인구_수']]

# 통신정보 정제 
# 연령대 그룹핑 함수
def categorize_age(age):
    if age in [20, 25, 30]:
        return 'age_20_30'
    elif age in [35, 40, 45, 50]:
        return 'age_35_50'
    elif age in [55, 60, 65, 70, 75]:
        return 'age_55_75'
    else:
        return 'other'

# 처리할 분기 목록
quarters = ['20231','20232','20233','20234','20241','20242','20243','20244']
final_list = []

for quarter in quarters:
    filename = f'통신정보_{quarter}.csv'

    # CSV 파일 읽기
    df = pd.read_csv('./data/' + filename)

    # 필요한 열 필터링
    df_filtered = df[['행정동코드', '행정동', '연령대', '총인구수', '1인가구수']].copy()

    # 숫자형으로 변환
    df_filtered['총인구수'] = df_filtered['총인구수'].str.replace(',', '', regex=False).astype(float)
    df_filtered['1인가구수'] = df_filtered['1인가구수'].str.replace(',', '', regex=False).astype(float)

    # 연령대 그룹 분류
    df_filtered['age_group'] = df_filtered['연령대'].apply(categorize_age)

    # 총 인구수 및 1인 가구수 집계
    total_pop = df_filtered.groupby(['행정동코드', '행정동'])['총인구수'].sum().reset_index(name='총인구수_합')
    single_households = df_filtered.groupby(['행정동코드', '행정동'])['1인가구수'].sum().reset_index(name='1인가구수_합')

    # 연령대별 인구 집계
    age_group_pop = df_filtered[df_filtered['age_group'] != 'other']
    age_group_sum = age_group_pop.groupby(['행정동코드', '행정동', 'age_group'])['총인구수'].sum().unstack(fill_value=0).reset_index()

    # 병합
    df_merged = total_pop.merge(single_households, on=['행정동코드', '행정동']).merge(age_group_sum, on=['행정동코드', '행정동'])

    # 숫자형 변환 후 비율 계산
    df_merged['총인구수_합'] = pd.to_numeric(df_merged['총인구수_합'], errors='coerce')
    df_merged['1인가구수_합'] = pd.to_numeric(df_merged['1인가구수_합'], errors='coerce')

    df_merged['1인_가구비'] = df_merged['1인가구수_합'] / df_merged['총인구수_합']
    df_merged['20_30_인구비'] = df_merged.get('age_20_30', 0) / df_merged['총인구수_합']
    df_merged['31_50_인구비'] = df_merged.get('age_35_50', 0) / df_merged['총인구수_합']
    df_merged['51_75_인구비'] = df_merged.get('age_55_75', 0) / df_merged['총인구수_합']

    # 기준 코드 추가
    df_merged['기준_년분기_코드'] = quarter

    # 결과 저장
    result_df = df_merged[['기준_년분기_코드', '행정동코드', '행정동', '총인구수_합','1인가구수_합', '1인_가구비', '20_30_인구비', '31_50_인구비', '51_75_인구비']]
    final_list.append(result_df)

# 모든 분기 병합
Population = pd.concat(final_list, ignore_index=True)
# 형식 통일 
Population['기준_년분기_코드'] = Population['기준_년분기_코드'].astype(int)
# 병합 키 설정
merge_keys = ['기준_년분기_코드','행정동_코드']

# Sales 기준으로 컬럼 단위 병합
df_상권데이터 = Sales_grouped.merge(Stores_grouped, on=merge_keys, how='left') \
                          .merge(Indicators, on=merge_keys, how='left') \
                          .merge(Incomes, on=merge_keys, how='left') \
                          .merge(Apartments, on=merge_keys, how='left') \
                          .merge(Floatings, on=merge_keys, how='left') \
                          .merge(Workers, on=merge_keys, how='left') \
                          .merge(Facilities, on=merge_keys, how='left') \
                          .merge(Residents, on=merge_keys, how='left') 

# Population 코드명 기준 조인
df_상권데이터.rename(columns={'행정동_코드_명_x':'행정동'},inplace=True)
df_상권데이터.loc[df_상권데이터['행정동_코드'] == 11620685, '행정동'] = '신사동(관악)'
df_상권데이터.loc[df_상권데이터['행정동_코드'] == 11680510, '행정동'] = '신사동(강남)'
df_상권데이터['행정동'] = df_상권데이터['행정동'].str.replace('?', '·', regex=False)

Population.loc[(Population['행정동코드'] == 1121068) & (Population['행정동'] == '신사동'), '행정동'] = '신사동(관악)'
Population.loc[(Population['행정동코드'] == 1123051) & (Population['행정동'] == '신사동'), '행정동'] = '신사동(강남)'

df_상권데이터 = df_상권데이터.merge(Population, on=['기준_년분기_코드','행정동'], how='left')

# 결측치 대체할 컬럼 리스트
job_cols = [
    '남성_직장_인구_수', '여성_직장_인구_수',
    '연령대_10_직장_인구_수', '연령대_20_직장_인구_수', '연령대_30_직장_인구_수',
    '연령대_40_직장_인구_수', '연령대_50_직장_인구_수', '연령대_60_이상_직장_인구_수',
    '총인구수_합','1인가구수_합','1인_가구비','20_30_인구비','31_50_인구비','51_75_인구비'
]

# 분기 기준으로 그룹별 평균 구하기
grouped_means = df_상권데이터.groupby('기준_년분기_코드')[job_cols].transform('mean')

# 결측치를 분기별 평균값으로 대체
df_상권데이터[job_cols] = df_상권데이터[job_cols].fillna(grouped_means)
df_상권데이터.info()

# export
df_상권데이터.to_csv('상권데이터.csv',encoding='utf-8-sig', index=False)

# ---------------------------------------
# Part2. 착한가격업소 데이터 병합 
# ---------------------------------------

# --- 1. 좌표 가져오는 함수
def get_coords_from_address(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json' # 카카오 주소 호출 API
    headers = {'Authorization': f'KakaoAK {'386797ea7e88e3189c4ae3389f5e13c6'}'}
    params = {"query": address}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        data = res.json()
        
        if data.get('documents'):
            doc = data['documents'][0]
            return float(doc['x']), float(doc['y'])  # (경도, 위도)
    
    except requests.exceptions.HTTPError as e:
        print(f"[HTTPError] 주소 요청 실패 - {address} | {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"[ConnectionError] 인터넷 연결 오류 - {address} | {e}")
    except requests.exceptions.Timeout as e:
        print(f"[Timeout] 요청 시간 초과 - {address} | {e}")
    except requests.exceptions.RequestException as e:
        print(f"[RequestException] 기타 요청 오류 - {address} | {e}")
    except Exception as e:
        print(f"[UnknownError] 알 수 없는 오류 - {address} | {e}")
    
    return None, None

# --- 2. 좌표 → 행정동 코드/명
def get_region_code_from_coords(x, y):
    url = "https://api.vworld.kr/req/address" # 국토부 디지털 트윈국토 주소 API
    params = {
        "service": "address",
        "request": "getAddress",
        "point": f"{x},{y}",  # 경도, 위도 순서
        "crs": "EPSG:4326",
        "format": "json",
        "type": "both",
        "key": '248F6D1B-0D46-3D34-85E2-0463D838D5CB'
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()  # HTTP 상태코드가 4xx/5xx면 예외 발생
        data = response.json()

        if data['response']['status'] == 'OK':
            # "type"이 "road"인 결과만 필터링
            result = data['response']['result'][1]

            # 행정동 추출: 행정동 없으면 법정동 fallback
            dong_name = result.get('level4A') or result.get('level4L')
            dong_code = result.get('level4AC') or result.get('level4LC')

            return dong_name, dong_code
        else:
            print(f"[API Response Error] 상태: {data['response']['status']} | 좌표: ({x}, {y})")
            return None, None

    except requests.exceptions.HTTPError as e:
        print(f"[HTTPError] 응답 코드 오류 | 좌표: ({x}, {y}) | {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"[ConnectionError] 연결 실패 | 좌표: ({x}, {y}) | {e}")
    except requests.exceptions.Timeout as e:
        print(f"[Timeout] 응답 지연 | 좌표: ({x}, {y}) | {e}")
    except requests.exceptions.RequestException as e:
        print(f"[RequestException] 요청 실패 | 좌표: ({x}, {y}) | {e}")
    except (KeyError, IndexError) as e:
        print(f"[ParsingError] 결과 구조 파싱 실패 | 좌표: ({x}, {y}) | {e}")
    except Exception as e:
        print(f"[UnknownError] 알 수 없는 오류 | 좌표: ({x}, {y}) | {e}")

    return None, None

# --- 3. 주소 리스트를 받아 tqdm 적용하며 행정동 정보 반환
def get_dong_info_parallel(addresses, max_workers=10):
    results = []

    def worker(address):
        x, y = get_coords_from_address(address)
        if x is not None and y is not None:
            dong_name, dong_code = get_region_code_from_coords(x, y)
        else:
            dong_name, dong_code = None, None
        return {'주소': address, '행정동_명': dong_name, '행정동_코드': dong_code}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, addr): addr for addr in addresses}
        for future in tqdm(as_completed(futures), total=len(futures), desc="병렬 행정동 매핑 중"):
            result = future.result()
            results.append(result)

    return pd.DataFrame(results)

# 착한가격업소_2023~2024
GoodPrices_Data = {
    "20233": "./data/착한가격업소_20233.csv",
    "20241": "./data/착한가격업소_20241.csv",
    "20242": "./data/착한가격업소_20242.csv",
    "20243": "./data/착한가격업소_20243.csv",
    "20244": "./data/착한가격업소_20244.csv"
}

df_list = []
for quarter, path in GoodPrices_Data.items():
    df = pd.read_csv(path, encoding='cp949')  # 필요시 encoding='cp949'
    df['기준_년분기_코드'] = quarter           # 분기 컬럼 추가
    df_list.append(df)


# 1. 하나의 데이터프레임으로 병합
GoodPrices = pd.concat(df_list, ignore_index=True)
GoodPrices_서울특별시 = GoodPrices[GoodPrices['시도'] =='서울특별시']

# 2. 주소 행정동 매핑 
df_주소_행정동매핑 = get_dong_info_parallel(GoodPrices_서울특별시['주소'])
df_주소_행정동매핑 = df_주소_행정동매핑.drop_duplicates(subset='주소', keep='first')

# 3. 병합 
GoodPrices_서울특별시 = GoodPrices_서울특별시.merge(df_주소_행정동매핑,on=['주소'],how='left')
GoodPrices_서울특별시.to_csv('착한가격업소.csv',encoding='utf-8-sig', index=False)

# 4. 행정동 매핑 안된 곳 추가정제 (수기)
GoodPrices_서울특별시[GoodPrices_서울특별시['행정동_명'].isna()]
# -----------------------------------------------
# Part3. 임대료 데이터 병합 
# -----------------------------------------------
중대형상가_20211 = pd.read_csv('./data/중대형상가_20211.csv', encoding='utf-8')
중대형상가_20212 = pd.read_csv('./data/중대형상가_20212.csv', encoding='utf-8')
중대형상가_20213 = pd.read_csv('./data/중대형상가_20213.csv', encoding='utf-8')
중대형상가_20214 = pd.read_csv('./data/중대형상가_20214.csv', encoding='utf-8')

중대형상가_20221 = pd.read_csv('./data/중대형상가_20221.csv', encoding='utf-8')
중대형상가_20222 = pd.read_csv('./data/중대형상가_20222.csv', encoding='utf-8')
중대형상가_20223 = pd.read_csv('./data/중대형상가_20223.csv', encoding='utf-8')
중대형상가_20224 = pd.read_csv('./data/중대형상가_20224.csv', encoding='utf-8')

중대형상가_20231 = pd.read_csv('./data/중대형상가_20231.csv', encoding='utf-8')
중대형상가_20232 = pd.read_csv('./data/중대형상가_20232.csv', encoding='utf-8')
중대형상가_20233 = pd.read_csv('./data/중대형상가_20233.csv', encoding='utf-8')
중대형상가_20234 = pd.read_csv('./data/중대형상가_20234.csv', encoding='utf-8')

중대형상가_20241 = pd.read_csv('./data/중대형상가_20241.csv', encoding='cp949')
중대형상가_20242 = pd.read_csv('./data/중대형상가_20242.csv', encoding='cp949')
중대형상가_20243 = pd.read_csv('./data/중대형상가_20243.csv', encoding='cp949')
중대형상가_20244 = pd.read_csv('./data/중대형상가_20244.csv', encoding='cp949')
중대형상가_20251 = pd.read_csv('./data/중대형상가_20251.csv', encoding='cp949')


# 서울특별시 & 비상권 제외 필터링 함수
def 필터링(df):
    return df[
        df['소재지'].str.startswith("서울특별시") &
        (df['상권명'] != '0.비상권')
    ]

# 필터링 적용
중대형상가_20211 = 필터링(중대형상가_20211)
중대형상가_20212 = 필터링(중대형상가_20212)
중대형상가_20213 = 필터링(중대형상가_20213)
중대형상가_20214 = 필터링(중대형상가_20214)

중대형상가_20221 = 필터링(중대형상가_20221)
중대형상가_20222 = 필터링(중대형상가_20222)
중대형상가_20223 = 필터링(중대형상가_20223)
중대형상가_20224 = 필터링(중대형상가_20224)

중대형상가_20231 = 필터링(중대형상가_20231)
중대형상가_20232 = 필터링(중대형상가_20232)
중대형상가_20233 = 필터링(중대형상가_20233)
중대형상가_20234 = 필터링(중대형상가_20234)

중대형상가_20241 = 필터링(중대형상가_20241)
중대형상가_20242 = 필터링(중대형상가_20242)
중대형상가_20243 = 필터링(중대형상가_20243)
중대형상가_20244 = 필터링(중대형상가_20244)
중대형상가_20251 = 필터링(중대형상가_20251)

# 분기별 월세 컬럼 목록
분기컬럼 = ['제1월시장임대료_㎡당월세임대료', '제2월시장임대료_㎡당월세임대료', '제3월시장임대료_㎡당월세임대료']

# 분기별 평균 계산 함수
def 분기평균(df, 컬럼리스트, 분기이름):
    grouped = df.groupby('상권명')[컬럼리스트].median()
    grouped['평균임대료'] = grouped.mean(axis=1)
    grouped['기준_년분기_코드'] = 분기이름
    return grouped[['기준_년분기_코드','평균임대료']].reset_index()

# 컬럼명 통일
dfs = [중대형상가_20211, 중대형상가_20212, 중대형상가_20213, 중대형상가_20214, 중대형상가_20221, 중대형상가_20222, 중대형상가_20223, 중대형상가_20224]  # 필요한 데이터프레임 리스트

for df in dfs:
    df.rename(columns={'제1월시장임대료_m당월세임대료' : '제1월시장임대료_㎡당월세임대료', 
                       '제2월시장임대료_m당월세임대료' : '제2월시장임대료_㎡당월세임대료',
                       '제3월시장임대료_m당월세임대료' : '제3월시장임대료_㎡당월세임대료'}, inplace=True)

# 각각의 분기별 평균 계산
지역별_2021_1분기_평균 = 분기평균(중대형상가_20211, 분기컬럼, '20211')
지역별_2021_2분기_평균 = 분기평균(중대형상가_20212, 분기컬럼, '20212')
지역별_2021_3분기_평균 = 분기평균(중대형상가_20213, 분기컬럼, '20213')
지역별_2021_4분기_평균 = 분기평균(중대형상가_20214, 분기컬럼, '20214')

지역별_2022_1분기_평균 = 분기평균(중대형상가_20221, 분기컬럼, '20221')
지역별_2022_2분기_평균 = 분기평균(중대형상가_20222, 분기컬럼, '20222')
지역별_2022_3분기_평균 = 분기평균(중대형상가_20223, 분기컬럼, '20223')
지역별_2022_4분기_평균 = 분기평균(중대형상가_20224, 분기컬럼, '20224')

지역별_2023_1분기_평균 = 분기평균(중대형상가_20231, 분기컬럼, '20231')
지역별_2023_2분기_평균 = 분기평균(중대형상가_20232, 분기컬럼, '20232')
지역별_2023_3분기_평균 = 분기평균(중대형상가_20233, 분기컬럼, '20233')
지역별_2023_4분기_평균 = 분기평균(중대형상가_20234, 분기컬럼, '20234')

지역별_2024_1분기_평균 = 분기평균(중대형상가_20241, 분기컬럼, '20241')
지역별_2024_2분기_평균 = 분기평균(중대형상가_20242, 분기컬럼, '20242')
지역별_2024_3분기_평균 = 분기평균(중대형상가_20243, 분기컬럼, '20243')
지역별_2024_4분기_평균 = 분기평균(중대형상가_20244, 분기컬럼, '20244')
지역별_2025_1분기_평균 = 분기평균(중대형상가_20251, 분기컬럼, '20251')

# 병합
지역별_임대료 = pd.concat([
    지역별_2021_1분기_평균,
    지역별_2021_2분기_평균,
    지역별_2021_3분기_평균,
    지역별_2021_4분기_평균,
    지역별_2022_1분기_평균,
    지역별_2022_2분기_평균,
    지역별_2022_3분기_평균,
    지역별_2022_4분기_평균,
    지역별_2023_1분기_평균,
    지역별_2023_2분기_평균,
    지역별_2023_3분기_평균,
    지역별_2023_4분기_평균,
    지역별_2024_1분기_평균,
    지역별_2024_2분기_평균,
    지역별_2024_3분기_평균,
    지역별_2024_4분기_평균,
    지역별_2025_1분기_평균
], axis=0, ignore_index=True)

# ---------------------------------------------------------------------------------------------------------
# 상권-행정동 공간정보조인을 통해 임대료 상권의 행정동코드를 매핑하였음
# ---------------------------------------------------------------------------------------------------------

# 1. 상권 구획도 로드 및 좌표계 설정
gdf_상권 = gpd.read_file("./data/최종상권368.shp").to_crs(epsg=4326)
gdf_상권 = gdf_상권[gdf_상권['시도코드'] == '11']

gdf_행정동 = gpd.read_file("./data/sig.shp", encoding='utf-8')
gdf_행정동 = gdf_행정동.set_crs(epsg=5181).to_crs(epsg=4326)

# 2. 상권 중심점 GeoDataFrame 생성
gdf_centroids = gpd.GeoDataFrame(
    gdf_상권.drop(columns='geometry'),  # 기존 geometry 제거
    geometry=gdf_상권.geometry.centroid,
    crs=gdf_상권.crs
)

# 3. 공간조인: 중심점이 포함되는 행정동 찾기
gdf_매핑 = gpd.sjoin(
    gdf_centroids,
    gdf_행정동[['ADSTRD_CD', 'ADSTRD_NM', 'geometry']],
    how='left',
    predicate='within'  # 중심점이 행정동 경계 내에 있는지
)

# 4. 컬럼 이름 정리 (선택)
gdf_매핑 = gdf_매핑.rename(columns={'ADSTRD_CD': '행정동_코드', 'ADSTRD_NM': '행정동_명'})
gdf_매핑 = gdf_매핑[['상권명', '행정동_명', '행정동_코드']]

# 6. 지역별 임대료에 병합 
지역별_임대료 = 지역별_임대료.merge(gdf_매핑, on=['상권명'], how='left')
지역별_임대료.to_csv('지역별_임대료.csv',encoding='utf-8-sig', index=False)

# ===============================================================================
# 2. 분석을 위한 파생변수 생성 
#
# [[Part1. 착한가격업소]]
#   1. 기준_년분기_코드, 행정동코드 별로 업소수
#   2. 분기변화에 따른 업체수 유지율(전분기업소대비 남아있는 업체수)
#   3. 분기변화에 따른 업소수 변화량(전분기업소수대비 증가/감소한 업소수)
# [[Part2. 데이터프레임을 최종 병합한다.]]
#   - 임대료, 착한가격업소_요약, 매출액, 점포수 등 프레임 병합
# [[Part3. 매출액, 점포수, 임대료 변화량]] - 물가대리변수
#   - 임대료, 매출액, 점포수 정규화 
#   - Resional Price Proxy(RPP) = 임대료_norm*0.6 + 매출액/점포수_norm*0.1
# ===============================================================================

# ---------------------------------------
# Part1. 착한가격업소 파생변수 생성 
# ---------------------------------------
GoodPrices = pd.read_csv('./data/착한가격업소.csv', encoding='utf-8')

GoodPrices[['기준_년분기_코드','행정동_명','행정동_코드']] = GoodPrices[['기준_년분기_코드','행정동_명','행정동_코드']].astype('str')
GoodPrices['행정동_코드'] = GoodPrices['행정동_코드'].str[:8]

# 1. 기준_년분기_코드, 행정동코드 별 업소수 계산
shop_counts = GoodPrices.groupby(['기준_년분기_코드', '행정동_코드','행정동_명'])['업소명'].nunique().reset_index()
shop_counts.rename(columns={'업소명': '업소수'}, inplace=True)

# 2. 분기 변화에 따른 유지율 및 업소수 증감량 계산
quarters = sorted(GoodPrices['기준_년분기_코드'].unique())
records = []

# 업소명 띄어쓰기 붙이기 
def clean_name(name):
    if pd.isna(name):
        return ""
    return name.replace(" ", "").strip().lower()

# 업소명이 일부 변경된 경우도 있으므로, 유사도 기반으로 매칭한다. 
# ex) 평범식당 -> 제일평범식당 
def fuzzy_match(name1, name2):
    return (name1 in name2 or 
            name2 in name1 or
            Levenshtein.ratio(name1, name2) >= 0.5
           )

# 각 분기별로 전분기대비 업소수 증감, 업소의 유지율(retension)을 구한다.
for i in range(1, len(quarters)):
    prev_q = quarters[i - 1]
    curr_q = quarters[i]

    df_prev = GoodPrices[GoodPrices['기준_년분기_코드'] == prev_q]
    df_curr = GoodPrices[GoodPrices['기준_년분기_코드'] == curr_q]

    all_dongs = set(df_prev['행정동_코드']) | set(df_curr['행정동_코드'])

    for dong in all_dongs:
        prev_names = df_prev[df_prev['행정동_코드'] == dong]['업소명'].dropna().apply(clean_name).tolist()
        curr_names = df_curr[df_curr['행정동_코드'] == dong]['업소명'].dropna().apply(clean_name).tolist()

        prev_count = len(prev_names)
        curr_count = len(curr_names)

        retained = set()
        for pname in prev_names:
            for cname in curr_names:
                if fuzzy_match(pname, cname):
                    retained.add(pname)
                    break  # 하나라도 매칭되면 그 이전 이름은 유지된 것으로 처리

        retention_rate = len(retained) / prev_count if prev_count > 0 else None
        change_count = curr_count - prev_count if prev_count > 0 else None

        records.append({
            '기준_년분기_코드': curr_q,
            '행정동_코드': dong,
            '전분기대비_유지율': retention_rate,
            '전분기대비_증감업소수': change_count
        })

change_df = pd.DataFrame(records)

# 최종 병합
GoodPrices_summary = pd.merge(shop_counts, change_df, on=['기준_년분기_코드', '행정동_코드'], how='left')
GoodPrices_summary.to_csv('착한가격업소_요약.csv',encoding='utf-8-sig', index=False)

# ------------------------------------------------------------------
# Part2. 임대료, 착한가격업소_요약, 매출액, 점포수 등의 데이터프레임을 최종 병합한다.
# -------------------------------------------------------------------
df_상권데이터= pd.read_csv('./data/상권데이터.csv', encoding='utf-8')
df_지역별_임대료 = pd.read_csv('./data/지역별_임대료.csv', encoding='utf-8')
df_착한가격업소_요약 = pd.read_csv('./data/착한가격업소_요약.csv', encoding='utf-8')

# 컬럼형식 변경 
df_상권데이터[['기준_년분기_코드','행정동_코드']] = df_상권데이터[['기준_년분기_코드','행정동_코드']].astype('str')
df_지역별_임대료[['기준_년분기_코드','행정동_코드']] = df_지역별_임대료[['기준_년분기_코드','행정동_코드']].astype('str')
df_착한가격업소_요약[['기준_년분기_코드','행정동_코드']] = df_착한가격업소_요약[['기준_년분기_코드','행정동_코드']].astype('str')

# 2023~24년도 데이터만 필터링
years = ('2023', '2024')

df_base_2023_2024 = df_상권데이터[df_상권데이터['기준_년분기_코드'].str.startswith(years)]
df_지역별_임대료_2023_2024 = df_지역별_임대료[df_지역별_임대료['기준_년분기_코드'].str.startswith(years)]
df_착한가격업소_요약_2023_2024 = df_착한가격업소_요약[df_착한가격업소_요약['기준_년분기_코드'].str.startswith(years)]

# 행정동코드 통일 
df_지역별_임대료_2023_2024['행정동_코드'] = df_지역별_임대료_2023_2024['행정동_코드'].str[:8]
df_착한가격업소_요약_2023_2024['행정동_코드'] = df_착한가격업소_요약_2023_2024['행정동_코드'].str[:8]

# data merge 
merge_keys = ['기준_년분기_코드','행정동_코드']

# 분석을 위해 임대료가 있는 지역을 base 테이블로하여 join 한다.
df_GoodPrice = df_지역별_임대료_2023_2024.merge(df_base_2023_2024, on=merge_keys, how='left') \
                                      .merge(df_착한가격업소_요약_2023_2024, on=merge_keys, how='left')

# version2
df_GoodPrice = df_base_2023_2024.merge(df_착한가격업소_요약_2023_2024, on=merge_keys, how='left')

# 임시 결측치 제거 
df_GoodPrice = df_GoodPrice[df_GoodPrice['기준_년분기_코드'] != '20245']
df_GoodPrice = df_GoodPrice[df_GoodPrice['행정동_코드']!='nan']
df_GoodPrice = df_GoodPrice[df_GoodPrice['행정동'].notna()]

# cond_new(착한가격업소 신규등장), cond_empty(착한가격업소 없는지역)
cond_new = (df_GoodPrice['업소수'].notna()) & (df_GoodPrice['전분기대비_유지율'].isna())
cond_empty = (df_GoodPrice['업소수'].isna()) & (df_GoodPrice['전분기대비_유지율'].isna())

# 1. 신규 진입 구역: 유지율 = 1, 증감 = 업소수
df_GoodPrice.loc[cond_new, '전분기대비_유지율'] = 1
df_GoodPrice.loc[cond_new, '전분기대비_증감업소수'] = df_GoodPrice.loc[cond_new, '업소수']

# 2. 완전 공백 구역: 유지율 = 0, 증감 = 0
df_GoodPrice.loc[cond_empty, ['업소수', '전분기대비_유지율', '전분기대비_증감업소수']] = 0

# 총_유동인구 
df_GoodPrice['총_유동인구_수'] = df_GoodPrice['남성_유동인구_수'] + df_GoodPrice['여성_유동인구_수'] 

# 점포수_대비_매출액 생성
df_GoodPrice['점포수_대비_매출액'] = df_GoodPrice.apply(
    lambda row: 0 if row['점포_수'] == 0 or pd.isna(row['점포_수'])
    else int(np.floor(row['당월_매출_금액'] / row['점포_수'])),
    axis=1
)

# 10~30대 유동인구 합계 
df_GoodPrice['유동인구_10_30대'] = (
    df_GoodPrice['연령대_10_유동인구_수'] +
    df_GoodPrice['연령대_20_유동인구_수'] +
    df_GoodPrice['연령대_30_유동인구_수']
)

# 40~60대 이상 유동인구수 합계
df_GoodPrice['유동인구_40_이상'] = (
    df_GoodPrice['연령대_40_유동인구_수'] +
    df_GoodPrice['연령대_50_유동인구_수'] +
    df_GoodPrice['연령대_60_이상_유동인구_수']
)

# 총_직장인구
df_GoodPrice['총_직장인구_수'] = df_GoodPrice['남성_직장_인구_수'] + df_GoodPrice['여성_직장_인구_수'] 
# 지역별 점포수 대비 업소수 
df_GoodPrice['착한가격_업소수_비중'] = (
    df_GoodPrice['업소수'] / df_GoodPrice['점포_수']
).round(3)

# 최종데이터셋 export
df_GoodPrice.to_csv('./model/상권_착한가격업소_병합.csv',encoding='utf-8-sig', index=False)


#df_GoodPrice['log_임대료'] = np.log(df_GoodPrice['평균임대료'])
#df_GoodPrice['log_점포수_대비_매출액'] = np.log(df_GoodPrice['점포수_대비_매출액'])

#df_GoodPrice = zscore_scale(df_GoodPrice,'log_임대료')
#df_GoodPrice = zscore_scale(df_GoodPrice,'log_점포수_대비_매출액')

#df_GoodPrice['물가_proxy'] = df_GoodPrice['log_임대료']*0.6 + df_GoodPrice['log_점포수_대비_매출액']*0.4


# ===============================================================================
#  1. 회귀_분석
#  [Model1]
#   H1. 지역 내 외식지출비가 높은지역일수록 착한가격업소 수 비중이 감소한다. - 검증 
#   H2. 지역 내 폐업률이 높은지역일수록 착한가격업소 수 비중이 증가한다 - 검증 
#   H3. 지역 내 상권축소 지역일수록 착한가격업소 수 비중이 증가한다. - 검증
#   H4. 지역 내 20_30대 인구비가 높은지역일수록 착한가격업소 수 비중이 감소한다. - 기각 
#   [Model2]
#   H6. 지역 내 상권축소 지역에 따라 20_30대 인구비가 착한가격업소 수에 미치는 영향이 다를 것이다.
#   [Model3]
#   추가. H2,H3,H4 의 lag_1 독립 시차변수를 통해 역인과성에 대한 강건성 검증 
# ===============================================================================

# 타입변환 
df_GoodPrice['기준_년분기_코드'] = df_GoodPrice['기준_년분기_코드'].astype(int)

# 임시코드(20234~20244 분기 한정)
df_GoodPrice = df_GoodPrice[df_GoodPrice['기준_년분기_코드'].isin([20234, 20241, 20242, 20243, 20244])]

# / 대체 - 회귀분석에서 인식오류  
# df_GoodPrice['상권명']= df_GoodPrice['상권명'].str.replace('/', '', regex=False)

# 1. 데이터 인덱스 설정
df_panel = df_GoodPrice.set_index(['행정동_코드', '기준_년분기_코드']).sort_index()

# 2. 더미 변수 생성
time_dummies = pd.get_dummies(df_panel.reset_index()['기준_년분기_코드'], prefix='분기', drop_first=True)

# 3. 기존 변수와 더미 병합
df_model = pd.concat([
    df_panel.reset_index(drop=True),
    time_dummies
], axis=1)

# 4. 인덱스 재설정
df_model = df_model.set_index(df_panel.index).sort_index()

# 5. 카테고리형 변수 원핫 인코딩
df_model = pd.get_dummies(df_model, columns=['상권_변화_지표'], drop_first=False)
df_model = df_model.drop(columns=['상권_변화_지표_HH'])

# 6. 다중공선성 확인 corr Matrix 

# 다중공선성 확인 대상 변수 리스트
cols = ['점포수_대비_매출액', '월_평균_소득_금액', '음식_지출_총금액','의료비_지출_총금액','교육_지출_총금액',
        '아파트_단지_수', '아파트_평균_시가', '총_유동인구_수', '유동인구_10_30대', '유동인구_40_이상', '총_직장인구_수', '총_상주인구_수','집객시설_수','운영_영업_개월_차이','폐업_영업_개월_차이','개업_률','폐업_률',
        '1인_가구비','20_30_인구비','31_50_인구비','51_75_인구비']

# corr_matrix
corr_matrix = df_model[cols].dropna().corr().round(2)

# 히트맵 그리기
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    aspect='auto',
    title='📊 변수 간 상관계수 히트맵 (Plotly)'
)

fig.update_layout(
    width=800,
    height=700,
    margin=dict(l=50, r=50, t=50, b=50),
    coloraxis_colorbar=dict(title="상관계수")
)

fig.show()

# 7. VIF 확인 
vif_cols = ['점포수_대비_매출액', '아파트_평균_시가', '음식_지출_총금액','의료비_지출_총금액','교육_지출_총금액',
            '아파트_단지_수', '총_유동인구_수','유동인구_10_30대','유동인구_40_이상', '총_직장인구_수', '총_상주인구_수','집객시설_수','운영_영업_개월_차이','폐업_영업_개월_차이','개업_률','폐업_률',
            '1인_가구비','20_30_인구비','31_50_인구비','51_75_인구비']

X = add_constant(df_model[vif_cols].dropna())
bool_cols = X.select_dtypes(include=['bool']).columns
X[bool_cols] = X[bool_cols].astype(float)

# VIF 계산
vif_data = pd.DataFrame()
vif_data['변수'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# corr 큰 월_평균소득, 집객시설_수, 운영_영업_개월_차이 제거
print(vif_data)

# 8. 왜도 확인 & 로그변환 
skew_test_columns = [
    '업소수',
    '착한가격_업소수_비중',
    '점포수_대비_매출액',
    '월_평균_소득_금액',
    '음식_지출_총금액',
    '의료비_지출_총금액',
    '교육_지출_총금액',
    '아파트_단지_수',
    '아파트_평균_시가',
    '총_유동인구_수',
    '유동인구_10_30대',
    '유동인구_40_이상',
    '총_직장인구_수',
    '총_상주인구_수',
    '집객시설_수',
    '운영_영업_개월_차이',
    '폐업_영업_개월_차이',
    '개업_률',
    '폐업_률',
    '1인_가구비',
    '20_30_인구비',
    '31_50_인구비',
    '51_75_인구비'
]

check_variable_skewness(df_model[skew_test_columns])

skew_columns = skew_test_columns.copy()
skew_columns.remove('총_상주인구_수')       # 상주인구수 대칭 
skew_columns.remove('개업_률')            # 개업율
skew_columns.remove('폐업_률')            # 폐업율
skew_columns.remove('폐업_영업_개월_차이')
skew_columns.remove('운영_영업_개월_차이')

# 최종확정된 변수를 기준으로 로그변환 
df_model = apply_log_transform(df_model, skew_columns)

# 9. 독립변수 최종확정 
# - 아파트_평균_시가_log, 집객시설 수, 운영_영업_개월_차이는 다중공선성이 높아 제거
ind_columns = [
    '점포수_대비_매출액_log',
    '음식_지출_총금액_log',
    '의료비_지출_총금액_log',
    '교육_지출_총금액_log',
    '아파트_평균_시가_log',
    '아파트_단지_수_log',
    '유동인구_10_30대_log',
    '유동인구_40_이상_log',
    '총_직장인구_수_log',
    '총_상주인구_수',
    '집객시설_수_log',
    '상권_변화_지표_LL',
    '상권_변화_지표_HL',
    '상권_변화_지표_LH',
    '개업_률',
    '폐업_률',
    '1인_가구비_log',
    '20_30_인구비_log',
    '31_50_인구비_log'
]
dep_columns= ['업소수_log','착한가격_업소수_비중','착한가격_업소수_비중_log']
dummy_columns= ['분기_20241','분기_20242','분기_20243','분기_20244']

# 10. 이상치 파악 후 제거 
check_outliers_std(df_model[ind_columns],3.0)
df_model_drop_outlier = drop_outlier_rows_std(df_model, ind_columns)

scale_columns = ind_columns.copy()
scale_columns.remove('상권_변화_지표_LL')       
scale_columns.remove('상권_변화_지표_HL')            
scale_columns.remove('상권_변화_지표_LH')            

# 11. 독립변수 단위 스케일링
df_model_after_scaling = apply_zscore_scaling(df_model_drop_outlier,scale_columns)

selected_columns = ind_columns + dep_columns + dummy_columns
df_final = df_model_after_scaling[selected_columns].copy()
df_final.columns = df_final.columns.str.strip()

# -----------------------------------
# 12. Model1 회귀식 구성 (가설1,2,3,4,5)
# -----------------------------------
base_formula = '착한가격_업소수_비중_log ~ 1 + 점포수_대비_매출액_log + 상권_변화_지표_HL + 상권_변화_지표_LH + 상권_변화_지표_LL + 폐업_률 + 음식_지출_총금액_log + 아파트_평균_시가_log + 아파트_단지_수_log + 유동인구_10_30대_log + 유동인구_40_이상_log + 총_직장인구_수_log + 집객시설_수_log + 총_상주인구_수 + 1인_가구비_log + 20_30_인구비_log + 31_50_인구비_log'
dummy_formula = ' + '.join(time_dummies.columns.tolist())
full_formula = base_formula + ' + ' + dummy_formula

# PanelOLS 적합
model = PanelOLS.from_formula(full_formula, data=df_final)
result = model.fit()
print(result.summary)

# rmsle
rmsle = compute_rmsle_from_result(result, df_final)

# 모델1 결과 저장 
save_full_model_output(result,rmsle,"./model/model1_results.csv")

# --------------------------------------------------------------------------
# 13. Model2 회귀식 구성 (가설6) - (조절변수 - 상호작용 항) 착한가격업소수 비중의 추가 증/감 검증
# --------------------------------------------------------------------------
base_formula2 = '착한가격_업소수_비중_log ~ 1 + 점포수_대비_매출액_log + 상권_변화_지표_HL + 상권_변화_지표_LH + 상권_변화_지표_LL + 폐업_률 + 음식_지출_총금액_log + 아파트_평균_시가_log + 아파트_단지_수_log + 총_유동인구_수_log + 총_직장인구_수_log + 집객시설_수_log + 총_상주인구_수 + 1인_가구비_log + 20_30_인구비_log + 상권_변화_지표_HL:20_30_인구비_log + 상권_변화_지표_LH:20_30_인구비_log + 상권_변화_지표_LL:20_30_인구비_log + 31_50_인구비_log'
dummy_formula2 = ' + '.join(time_dummies.columns.tolist())
full_formula2 = base_formula2 + ' + ' + dummy_formula2

# PanelOLS 적합
model2 = PanelOLS.from_formula(full_formula2, data=df_final)
result2 = model2.fit()
print(result2.summary)

# rmsle
rmsle2 = compute_rmsle_from_result(result2, df_final)

# 모델2 결과 저장 
save_full_model_output(result2,rmsle2,"./model/model2_results.csv")

# -----------------------------
# 14. Model3 시차변수 추가 후 재검증 
# -----------------------------
df_final = df_final.sort_values(by=['행정동_코드', '기준_년분기_코드'])

# 시차 생성 대상 변수 리스트
lag_vars = [
    '폐업_률',
    '음식_지출_총금액_log',
    '상권_변화_지표_HL',
    '상권_변화_지표_LH',
    '상권_변화_지표_LL',
    '20_30_인구비_log'
]

# 각 변수에 대해 -1분기 시차 생성
for var in lag_vars:
    df_final[f'{var}_lag1'] = df_final.groupby('행정동_코드')[var].shift(1)

# 첫 분기 제거(lag1 값이 null 인 분기)
first_quarter_idx = df_final.groupby('행정동_코드').head(1).index
df_lagged = df_final.drop(index=first_quarter_idx)

# 회귀식 구성 (가설1, 2) - 시차변수
lag_time_dummies = pd.get_dummies(df_lagged.reset_index()['기준_년분기_코드'], prefix='분기', drop_first=True)

base_formula3 = '착한가격_업소수_비중_log ~ 1 + 점포수_대비_매출액_log + 상권_변화_지표_HL_lag1 + 상권_변화_지표_LH_lag1 + 상권_변화_지표_LL_lag1 + 폐업_률_lag1 + 음식_지출_총금액_log_lag1 + 아파트_평균_시가_log + 아파트_단지_수_log + 유동인구_10_30대_log + 유동인구_40_이상_log + 총_직장인구_수_log + 집객시설_수_log + 총_상주인구_수 + 1인_가구비_log + 20_30_인구비_log_lag1 + 31_50_인구비_log'
dummy_formula3 = ' + '.join(lag_time_dummies.columns.tolist())
full_formula3 = base_formula3 + ' + ' + dummy_formula3

# PanelOLS 적합
model3 = PanelOLS.from_formula(full_formula3, data=df_lagged)
result3 = model3.fit()
print(result3.summary)

# rmsle
rmsle3 = compute_rmsle_from_result(result3, df_final)

# 모델3 결과 저장 
save_full_model_output(result3,rmsle3,"./model/model3_results.csv")

# ===========================================================
# 4. 유의한 변수를 통해 clustering 
#  - 공통전처리(스케일링, SPCA) 
#  [Part1. 계층적 Clustering]
#  [Part2. K-means 클러스터링]
#  [Part3. 병합 및 클러스터링 명칭부여]
#  [Part4. 비교시각화]
# ===========================================================
df_for_cluster = pd.read_csv('./model/상권_착한가격업소_병합.csv', encoding='utf-8')

# 착한가격_업소수_비중
df_for_cluster['착한가격_업소수_비중'] = (
    df_for_cluster['업소수'] / df_for_cluster['점포_수']
).round(3)

# 더미변수 생성
df_for_cluster = pd.get_dummies(df_for_cluster, columns=['상권_변화_지표'], drop_first=False)

# 왜도파악 
skew_columns = ['착한가격_업소수_비중','음식_지출_총금액','폐업_률','20_30_인구비']
scale_columns = ['음식_지출_총금액_log','폐업_률_log','20_30_인구비_log']
check_variable_skewness(df_for_cluster[skew_columns])

# 최종확정된 변수를 기준으로 로그변환 
df_for_cluster = apply_log_transform(df_for_cluster, skew_columns)

# 독립변수 단위 스케일링
df_for_cluster = apply_zscore_scaling(df_for_cluster, scale_columns)

# ▶ 유의변수 기반 클러스터링용 데이터 추출
features = ['음식_지출_총금액_log', '폐업_률_log', '20_30_인구비_log','상권_변화_지표_HL','상권_변화_지표_LH','상권_변화_지표_LL']
y_var = '착한가격_업소수_비중'
df_spca = df_for_cluster[[y_var]+features].dropna().copy()

# ▶ 스케일링 (연속형만)
#scale_vars = ['폐업_률', '음식_지출_총금액', '20_30_인구비']
#dummy_vars = ['상권_변화_지표_HL']
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(df_spca[scale_vars])
#X_dummy = df_spca[['상권_변화_지표_HL']].astype(float).values

X_final = df_spca[features]
y = df_spca[y_var]

# ▶ PCA 2차원 축소
pls = PLSRegression(n_components=2)
X_pls = pls.fit_transform(X_final, y)[0]  # 주성분 점수만 추출

# ▶ 공통 결과 저장용 DF
df_pls_clustered = pd.DataFrame(X_pls, columns=['SPC1', 'SPC2'], index=df_spca.index)
df_pls_analysis= pd.DataFrame(pls.x_weights_, columns=['SPC1', 'SPC2'], index=features[:pls.x_weights_.shape[0]])

df_pls_analysis.index.name = "feature"  # 인덱스 이름 설정
df_pls_analysis.to_csv("./model/pls_results.csv", index=True, encoding="utf-8-sig")

# ---------------------------------------
# [Part1. 계층적 Clustering]
# ---------------------------------------
Z = linkage(X_pls, method='ward')
hier_labels = fcluster(Z, t=4, criterion='maxclust')

df_pls_clustered['hier_cluster'] = hier_labels

# ---------------------------------------
# [Part2. K-means 클러스터링]
# ---------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(X_pls)

df_pls_clustered['kmeans_cluster'] = kmeans_labels

# ---------------------------------------
# [Part3. 병합 및 클러스터링 명칭부여]
# ---------------------------------------
# 인덱스를 기준으로 병합 (둘 다 동일한 순서일 경우)
df_cluster = df_for_cluster.join(df_pls_clustered)

# SPC1, SPC2 축 이름 재정의 (예: 소비축, 젊은층축 등)
df_cluster = df_cluster.rename(columns={
    'SPC1': '소비활성도_축',
    'SPC2': '2030_소비절제_축'
})

# 클러스터 번호에 따른 명칭 매핑 딕셔너리
hier_cluster_labels = {
    1: '전연령_극_저소비지역',
    2: '전연령_소비_비활성지역',
    3: '중장년_고소비지역',
    4: '청년_고소비지역',
}

# 클러스터 번호에 따른 명칭 매핑 딕셔너리
kmeans_cluster_labels = {
    0: '전연령_소비_비활성지역',
    1: '중장년_고소비지역',
    2: '청년_고소비지역',
    3: '전연령_극_저소비지역'
}

# 새로운 컬럼 생성 (기존 숫자 클러스터 유지도 가능)
df_cluster['hier_cluster_label'] = df_cluster['hier_cluster'].map(hier_cluster_labels)
df_cluster['kmeans_cluster_label'] = df_cluster['kmeans_cluster'].map(kmeans_cluster_labels)

# 최종 회귀 데이터 셋 
df_cluster.to_csv('./model/final_cluster.csv',encoding='utf-8-sig', index=False)

# ---------------------------------------
# [Part4. 비교시각화]
# ---------------------------------------

# ▶ 계층적 클러스터링 결과
fig_hier = px.scatter(
    df_cluster,
    x='소비활성도_축',
    y='2030_소비절제_축',
    color=df_cluster['hier_cluster_label'].astype(str),
    title='[계층적 클러스터링] Supervised PCA 기반 군집 결과',
    labels={'hier_cluster_label': 'Cluster'},
    hover_data={'소비활성도_축': ':.2f', '2030_소비절제_축': ':.2f'}
)
fig_hier.show()

# ▶ KMeans 결과
fig_kmeans = px.scatter(
    df_cluster,
    x='소비활성도_축',
    y='2030_소비절제_축',
    color=df_cluster['kmeans_cluster_label'].astype(str),
    title='[K-Means 클러스터링] Supervised PCA 기반 군집 결과',
    labels={'kmeans_cluster_label': 'Cluster'},
    hover_data={'소비활성도_축': ':.2f', '2030_소비절제_축': ':.2f'}
)
fig_kmeans.show()


# ===========================================================
# 5. 시각화 및 clustering 전략제시 
#   [Part1. 그래프 시각화]
#   - 시각화1. 연도별 착한가격업소 증가추이 
#   - 시각화2. 클러스터링 상권별 착한가격업소수 비율 (파이차트)
#   - 시각화3. 클러스터링 상권별 착한가격업소 증가추이 (선그래프 차트)
#   
#   [Part2. 지도 시각화]
#   - 시각화1. 착한가격업소 분포 점 시각화 (mapbox)
#   - 시각화2. 클러스터링(k-means, 4개) - 착한가격업소수 시각화 (mapbox)
# ===========================================================

df_상권_착한가격업소_병합 = pd.read_csv('./model/상권_착한가격업소_병합.csv', encoding='utf-8')

# ------------------------------------------------------------------
#   [Part1. 그래프 시각화]
#   - 시각화1. 연도별 착한가격업소 증가추이 
#   - 시각화2. 클러스터링 상권별 착한가격업소수 비율 (파이차트)
#   - 시각화3. 클러스터링 상권별 착한가격업소 증가추이 (선그래프 차트)
# ------------------------------------------------------------------
# ----------------------------
# 시각화1. 연도별 착한가격업소 증가추이 
# ----------------------------
# 기준_년분기_코드별 전체 업소수 총합 집계
df_trend = df_상권_착한가격업소_병합.groupby('기준_년분기_코드')['업소수'].sum().reset_index()
df_trend['기준_년분기_코드'] = df_trend['기준_년분기_코드'].astype('str')

# 시계열 라인 차트 생성
fig = px.line(
    df_trend,
    x='기준_년분기_코드',
    y='업소수',
    title='기준 분기별 착한가격업소 수 변화',
    labels={'기준_년분기_코드': '기준 분기', '업소수': '총 업소수'},
    markers=True
)

fig.update_layout(
    xaxis_title="기준_년분기_코드",
    yaxis_title="업소수",
    template='plotly_white',
    hovermode='x unified'
)

fig.show()

# --------------------------------------------
# 시각화2. 클러스터링 상권별 착한가격업소수 비율 (파이차트)
# --------------------------------------------
df_final_cluster = pd.read_csv('./model/final_cluster.csv', encoding='utf-8')

df_final_cluster.info()
# 클러스터 기준 선택 ('hc_cluster' 또는 'kmeans_cluster')
cluster_col = 'hier_cluster_label'  # 또는 'hc_cluster'

# 업소수 집계
df_pie_hc_cluster = df_final_cluster.groupby(cluster_col)[['점포_수','업소수']].mean().reset_index()
df_pie_hc_cluster['착한가격_업소수_비중'] = (
    df_pie_hc_cluster['업소수'] / df_pie_hc_cluster['점포_수']
).round(3)


# 파이차트 생성
fig = px.pie(
    df_pie_hc_cluster,
    values='착한가격_업소수_비중',
    names=cluster_col,
    title='클러스터별 업소수 비중',
    hole=0.4  # 도넛형으로 (선택사항)
)

fig.update_traces(textinfo='percent+label')
fig.show()

# --------------------------------------------
# 시각화3. 클러스터링 상권별 착한가격업소수 비율 (파이차트)
# --------------------------------------------

# 1. 군집별-분기별 업소수 합계와 점포수 합계 집계
df_grouped = (
    df_final_cluster
    .groupby(['hier_cluster_label', '기준_년분기_코드'])[['업소수', '점포_수']]
    .sum()
    .reset_index()
)

# 2. 비중 계산
df_grouped['업소수_비중'] = df_grouped['업소수'] / df_grouped['점포_수']
df_grouped['기준_년분기_코드'] = df_grouped['기준_년분기_코드'].astype('str')


# 3. Plotly 라인차트 시각화
fig = px.line(
    df_grouped,
    x='기준_년분기_코드',
    y='업소수_비중',
    color='hier_cluster_label',
    markers=True,
    title='클러스터별 점포수 대비 착한가격 업소 비중 추이',
    labels={
        '기준_년분기_코드': '기준 년분기',
        '업소수_비중': '업소수 비중',
        'hier_cluster_label': '클러스터'
    }
)

fig.update_layout(template='plotly_white')
fig.show()

# ------------------------------------------------------------------
#   [Part2. 지도 시각화]
#   - 시각화1. 착한가격업소 분포 점 시각화 (mapbox)
#   - 시각화2. 클러스터링(k-means, 4개) - 착한가격업소수 시각화 (mapbox)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 시각화1. 착한가격업소 분포 점 시각화 (mapbox)
# -----------------------------------------------------------------
# ------------------------------------------------------------------
# 시각화2. 클러스터링(계층적 클러스터링, 4개) - 착한가격업소수 시각화 (mapbox) 
# ------------------------------------------------------------------
df_final_cluster = pd.read_csv('./model/final_cluster.csv', encoding='utf-8')
df_final_cluster_20244 = df_final_cluster[df_final_cluster['기준_년분기_코드']==20244]

# 행정구역 경계
geometry = gpd.read_file('./data/sig.shp', encoding="utf8")
geometry.rename(columns={'ADSTRD_CD': '행정동_코드'}, inplace=True)
geometry['행정동_코드']= geometry['행정동_코드'].astype('str')
df_final_cluster_20244['행정동_코드']= df_final_cluster_20244['행정동_코드'].astype('str')

# 병합
merged = df_final_cluster_20244.merge(geometry, on= '행정동_코드')

# GeoDataFrame으로 변환
gdf = gpd.GeoDataFrame(merged, geometry='geometry')

gdf.head()
gdf = gdf.set_crs(epsg=5181)
gdf = gdf.to_crs(epsg=4326)

gdf.info()

# 클러스터링 기준 
color_map = {
    1: [255, 0, 0],       # 전연령_극_저소비지역 → 빨강
    2: [255, 165, 0],     # 전연령_소비_비활성지역 → 주황
    3: [0, 128, 0],       # 중장년_고소비지역 → 초록
    4: [0, 0, 255],       # 청년_고소비지역 → 파랑
}

gdf[['r', 'g', 'b']] = pd.DataFrame(
    gdf['hier_cluster'].map(color_map).tolist(),
    index=gdf.index
)

# 핵심 수정사항: geometry를 좌표 리스트로 변환
def polygon_to_coords(geom):
    """Polygon geometry를 pydeck이 이해할 수 있는 좌표 리스트로 변환"""
    if geom.geom_type == 'Polygon':
        # 외부 링의 좌표만 사용 (내부 구멍은 무시)
        return list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        # MultiPolygon인 경우 첫 번째 폴리곤만 사용
        return list(geom.geoms[0].exterior.coords)
    else:
        return []

# geometry를 좌표 리스트로 변환
gdf['coordinates'] = gdf['geometry'].apply(polygon_to_coords)

# 행정동 중심점 계산 (라벨링용)
gdf['centroid'] = gdf['geometry'].centroid
gdf['lon'] = gdf['centroid'].x
gdf['lat'] = gdf['centroid'].y

# 행정동명 매핑
# 1. JSON 매핑 파일 불러오기
with open('./util/map.json', encoding='utf-8') as f:
    dong_map = json.load(f)

# 2. '행정동' 컬럼을 기반으로 영문명을 매핑하여 새 컬럼 생성
gdf['행정동_영문'] = gdf['행정동'].map(dong_map)

# 3. 매핑이 안 된 경우 확인 (선택적)
unmapped = gdf[gdf['행정동_영문'].isna()]['행정동'].unique()
if len(unmapped) > 0:
    print("다음 행정동은 매핑되지 않았습니다:")
    print(unmapped)

# DataFrame으로 변환 (geometry 컬럼 제외)
df_for_pydeck = gdf.drop(columns=['geometry','centroid']).copy()
df_for_pydeck = pd.DataFrame(df_for_pydeck)

# PolygonLayer 생성
polygon_layer = pdk.Layer(
    "PolygonLayer",
    df_for_pydeck,
    get_polygon='coordinates',  # 변환된 좌표 사용
    get_fill_color=['r', 'g', 'b'],  # 클러스터링에 따른 색상
    get_elevation='착한가격_업소수_비중',
    elevation_scale=10000,
    extruded=True,  
    pickable=True,
    auto_highlight=True,
    get_line_color=[255, 255, 255],  # 경계선 색상 (흰색)
    line_width_min_pixels=1,
)


# TextLayer 생성 (행정동 라벨)
text_layer = pdk.Layer(
    "TextLayer",
    df_for_pydeck,
    get_position=['lon', 'lat',1500],
    get_text='행정동_영문',
    get_size=5,
    get_color=[255, 255, 255, 255],  # 흰색 텍스트
    get_angle=0,
    pickable=False,
    billboard=True,
    stroked=True
)

# 서울 중심 좌표
seoul_center = [126.9780, 37.5665]
view_state = pdk.ViewState(
    longitude=seoul_center[0],
    latitude=seoul_center[1],
    zoom=10,
    pitch=45,
    bearing=0
)

r = pdk.Deck(
    layers=[polygon_layer, text_layer],
    initial_view_state=view_state,
    tooltip={"text": "행정동: {행정동}\n착한가격업소 비중: {착한가격_업소수_비중}"},
)

def create_map_with_legend(deck_obj, filename="map_with_legend.html"):
    # HTML 문자열로 받아오기 (중요: as_string=True 추가)
    original_html = deck_obj.to_html(as_string=True)

    legend_html = """
    <div class="legend" style="
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(20, 20, 20, 0.95);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #444;
        font-size: 13px;
        z-index: 1000;
        width: 230px;
        font-family: 'Segoe UI', Arial, sans-serif;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(10px);
    ">
        <h3 style="
            margin: 0 0 18px 0;
            font-size: 18px;
            color: #fff;
            text-align: center;
            border-bottom: 2px solid #555;
            padding-bottom: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
        ">🗺️ 군집별 상권 유형</h3>

        <!-- 군집 1: 전연령_극_저소비지역 -->
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 16px; height: 12px; margin-right: 8px; border-radius: 2px; background-color: rgb(255, 0, 0);"></div>
            <span style="color: #ddd; font-size: 12px;">군집1: 전연령_극_저소비지역</span>
        </div>

        <!-- 군집 2: 전연령_소비_비활성지역 -->
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 16px; height: 12px; margin-right: 8px; border-radius: 2px; background-color: rgb(255, 165, 0);"></div>
            <span style="color: #ddd; font-size: 12px;">군집2: 전연령_소비_비활성지역</span>
        </div>

        <!-- 군집 3: 중장년_고소비지역 -->
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 16px; height: 12px; margin-right: 8px; border-radius: 2px; background-color: rgb(0, 128, 0);"></div>
            <span style="color: #ddd; font-size: 12px;">군집3: 중장년_고소비지역</span>
        </div>

        <!-- 군집 4: 청년_고소비지역 -->
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 16px; height: 12px; margin-right: 8px; border-radius: 2px; background-color: rgb(0, 0, 255);"></div>
            <span style="color: #ddd; font-size: 12px;">군집4: 청년_고소비지역</span>
        </div>

        <div style="
            margin-top: 12px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            border: 1px solid #333;
        ">
            <div style="color: #aaa; font-size: 10px; line-height: 1.3;">
                <strong>분석 기준:</strong><br>
                '외식지출', '폐업률', '20_30_인구비', '상권축소지역여부'<br>
                등 주요 특성을 기반으로 4개 군집으로 분류
            </div>
        </div>

        <div style="
            margin-top: 8px;
            text-align: center;
            color: #666;
            font-size: 10px;
            border-top: 1px solid #333;
            padding-top: 6px;
        ">
            지도를 클릭하면 해당 지역의 세부 정보를 볼 수 있습니다.
        </div>
    </div>
    """

    body_style = """
    <style>
        body {
            background: #1a1a1a !important;
        }
        .legend:hover {
            transform: translateY(-2px);
            transition: transform 0.2s ease;
        }
    </style>
    """

    # 수정된 HTML 삽입
    modified_html = original_html.replace('</body>', legend_html + '\n</body>')
    modified_html = modified_html.replace('</head>', body_style + '\n</head>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write(modified_html)

    print(f"📍 개선된 범례가 포함된 지도가 '{filename}'로 저장되었습니다.")
    return filename


"""
# 분기 코드 → 날짜형으로 변환 함수
def convert_quarter_to_date(code):
    year = int(str(code)[:4])
    quarter = int(str(code)[-1])
    month = (quarter - 1) * 3 + 1
    return pd.to_datetime(f"{year}-{month:02d}-01")

# Prophet을 통한 추세 분석 함수
def trend_analysis(df : pd.DataFrame, target_col : str):
    results = []

    # tqdm으로 진행률 확인
    for adstrd_cd, group in tqdm(df.groupby('행정동_코드')):
        group = group.sort_values('기준_년분기_코드')
        group['ds'] = group['기준_년분기_코드'].apply(convert_quarter_to_date)
        group['y'] = group[target_col]

        if len(group) < 4:  # 최소 분기 수 제한
            continue

        try:
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


sales_slope = trend_analysis(sales_grouped, '당월_매출_금액')
markets_slope = trend_analysis(market_grouped, '점포_수')

df_sales_slope = pd.DataFrame(sales_slope)
df_sales_slope.rename(columns={'추세기울기': 'sales_slope'}, inplace=True)

df_markets_slope = pd.DataFrame(markets_slope)
df_markets_slope.rename(columns={'추세기울기': 'markets_slope'}, inplace=True)

geometry.rename(columns={'ADSTRD_CD': '행정동_코드'}, inplace=True)
geometry['행정동_코드']= geometry['행정동_코드'].astype('str')

df_sales_slope['행정동_코드']= df_sales_slope['행정동_코드'].astype('str')
df_markets_slope['행정동_코드']= df_markets_slope['행정동_코드'].astype('str')

merged = geometry.merge(df_sales_slope, on= '행정동_코드')
merged = merged.merge(df_markets_slope, on= '행정동_코드')

def classify(row):
    if row['sales_slope'] > 0 and row['markets_slope'] >= 0:
        return 1  # Class 1
    elif row['sales_slope'] < 0 and row['markets_slope'] > 0:
        return 2  # Class 2
    elif row['sales_slope'] < 0 and row['markets_slope'] <= 0:
        return 3  # Class 3
    else:
        return 4  # Class 4

def calc_interaction_weight(row):
     return row['sales_norm'] * np.exp(row['markets_norm']-1)

def prepcs_derived_feature(df_base : pd.DataFrame):
    # interact_growth 계산
    scaler1 = MinMaxScaler(feature_range=(1000, 10000))
    scaler2 = MinMaxScaler()
    df_base['sales_norm'] = scaler1.fit_transform(np.abs(df_base[['sales_slope']]))
    df_base['markets_norm'] = scaler2.fit_transform(np.abs(df_base[['markets_slope']]))

    return df_base

# 클래스 분류 
merged['class'] = merged.apply(classify, axis=1)
merged = prepcs_derived_feature(merged)
merged['interact_growth'] = merged.apply(calc_interaction_weight, axis=1)



def create_auto_color_stops(series, n_bins=6, round_base=100):
 
    #Pandas Series로부터 자동 numeric_stops 생성

    #Parameters:
    #- series: 숫자값 Series (예: df['interact_growth'])
    #- n_bins: 구간 개수
    #- round_base: 구간을 몇 단위로 반올림할지

    #Returns:
    #- create_numeric_stops()에서 사용할 수 있는 stops 리스트

    min_val = series.min()
    max_val = series.max()

    # linspace로 구간 생성 → 정수 반올림
    raw_stops = np.linspace(min_val, max_val, n_bins)
    rounded_stops = [round(x / round_base) * round_base for x in raw_stops]

    # 중복 제거 및 정렬
    final_stops = sorted(set(rounded_stops))

    return create_color_stops(final_stops, colors='BuPu')

# 클래스별 데이터프레임 생성
merged_class1 = merged[merged['class'] ==1]
merged_class2 = merged[merged['class'] ==2]
merged_class3 = merged[merged['class'] ==3]
merged_class4 = merged[merged['class'] ==4]

merged_class1_subset = merged_class1[['geometry','행정동_코드', 'ADSTRD_NM','sales_slope','markets_slope','sales_norm','markets_norm','interact_growth']]
gdf_class1 = gpd.GeoDataFrame(merged_class1_subset, geometry='geometry')

gdf_class1 = gdf_class1.set_crs(epsg=5181)
gdf_class1 = gdf_class1.to_crs(epsg=4326)

gdf_class1.to_file('class1-geoj.geojson', driver="GeoJSON")
with open('class1-geoj.geojson', 'rt', encoding='utf-8') as f:
    gj_class1 = geojson.load(f)

# 클래스 컬러 stops
class_color_stops = create_auto_color_stops(gdf_class1['interact_growth'])
# 높이 시각화설정 
numeric_stops = create_auto_numeric_stops(gdf_class1['interact_growth'])


merged_class2_subset = merged_class2[['geometry','행정동_코드', 'ADSTRD_NM','sales_slope','markets_slope','sales_norm','markets_norm','interact_growth']]
gdf_class2 = gpd.GeoDataFrame(merged_class2_subset, geometry='geometry')

gdf_class2 = gdf_class2.set_crs(epsg=5181)
gdf_class2 = gdf_class2.to_crs(epsg=4326)

gdf_class2.to_file('class2-geoj.geojson', driver="GeoJSON")
with open('class2-geoj.geojson', 'rt', encoding='utf-8') as f:
    gj_class2 = geojson.load(f)

# 클래스 컬러 stops
class_color_stops = create_auto_color_stops(gdf_class2['interact_growth'])
# 높이 시각화설정 
numeric_stops = create_auto_numeric_stops(gdf_class2['interact_growth'])

# Choropleth 시각화 객체 생성
viz = ChoroplethViz(
    access_token=token,
    data=gj_class1,
    color_property='interact_growth',
    color_stops=class_color_stops,
    center=seoul_center,
    zoom=10)

# 성장세(매출추세*점포추세의 interaction 을 높이로 설정)
viz.bearing = -15
viz.pitch = 45

viz.height_property = 'interact_growth'
viz.height_stops = numeric_stops
viz.height_function_type = 'interpolate'

viz.show()
"""

gdf = load_clustered_geodataframe()

# 데이터 디버깅 코드
print("=== GeoDataFrame 디버깅 ===")
print(f"GDF 타입: {type(gdf)}")
print(f"전체 행 수: {len(gdf)}")
print(f"컬럼들: {list(gdf.columns)}")

print("\n=== geometry 컬럼 분석 ===")
print(f"geometry 타입: {type(gdf['geometry'])}")
print(f"geometry dtype: {gdf['geometry'].dtype}")
print(f"NaN 개수: {gdf['geometry'].isna().sum()}")
print(f"첫 번째 geometry: {gdf['geometry'].iloc[0] if len(gdf) > 0 else 'No data'}")

print("\n=== 필수 컬럼 확인 ===")
required_cols = ['kmeans_cluster', '착한가격_업소수_비중', '행정동']
for col in required_cols:
    if col in gdf.columns:
        print(f"{col}: OK (타입: {gdf[col].dtype})")
        print(f"  - NaN 개수: {gdf[col].isna().sum()}")
        if col == 'kmeans_cluster':
            print(f"  - 고유값: {gdf[col].unique()}")
    else:
        print(f"{col}: 누락!")

# geometry가 실제로 지오메트리인지 확인
if len(gdf) > 0 and not gdf['geometry'].isna().iloc[0]:
    try:
        first_geom = gdf['geometry'].iloc[0]
        print(f"\n첫 번째 geometry 속성:")
        print(f"  - geom_type: {first_geom.geom_type}")
        print(f"  - is_valid: {first_geom.is_valid}")
        print(f"  - bounds: {first_geom.bounds}")
    except Exception as e:
        print(f"Geometry 검증 중 오류: {e}")