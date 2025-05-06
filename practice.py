from sklearn.linear_model import LinearRegression
import geojson
import pandas as pd
import geopandas as gpd
import plotly.express as px
from prophet import Prophet
from tqdm import tqdm
import numpy as np

from mapboxgl.viz import ChoroplethViz
from mapboxgl.utils import df_to_geojson
from mapboxgl.utils import create_color_stops
from mapboxgl.utils import create_numeric_stops


# 행정구역 경계
geometry = gpd.read_file('./data/sig.shp', encoding="utf8")

sales = pd.read_csv('./data/sales.csv', encoding='cp949')
market = pd.read_csv('./data/market.csv')

# 서비스업종코드 음식업 한정(한식음식점, 일식음식점, 양식음식점, 제과점, 치킨전문점) 
sales_filter = sales[sales['서비스_업종_코드'].isin(['CS100001','CS100003','CS100004','CS100005','CS100007'])]

# 행정동-기준년분기별 매출액 그룹화 
sales_grouped = sales_filter.groupby(['행정동_코드', '기준_년분기_코드'])['당월_매출_금액'].mean().reset_index()

# 분기코드를 시간축으로 변환
sales_grouped['time'] = sales_grouped['기준_년분기_코드'].astype(str).str[:4].astype(int) + (sales_grouped['기준_년분기_코드'].astype(str).str[-1].astype(int) - 1) * 0.25

# 서비스업종코드 음식업 한정(한식음식점, 일식음식점, 양식음식점, 제과점, 치킨전문점) 
market_filter = market[market['서비스_업종_코드'].isin(['CS100001','CS100003','CS100004','CS100005','CS100007'])]

# 행정동-기준년분기별 점포수 그룹화 
market_grouped = market_filter.groupby(['행정동_코드', '기준_년분기_코드'])['점포_수'].sum().reset_index()



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

sales_slope = trend_analysis(sales_grouped, '당월_매출금액_합계')
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
    if row['sales_slope'] > 0 and row['markets_slope'] > 0:
        return 1  # Class 1
    elif row['sales_slope'] < 0 and row['markets_slope'] > 0:
        return 2  # Class 2
    elif row['sales_slope'] < 0 and row['markets_slope'] < 0:
        return 3  # Class 3
    else:
        return 4  # Class 4


merged['class'] = merged.apply(classify, axis=1)
#merged['interact_growth'] = merged['sales_slope'] * merged['store_slope']

# GeoDataFrame으로 변환
gdf = gpd.GeoDataFrame(merged, geometry='geometry')

gdf.head()
gdf = gdf.set_crs(epsg=5181)
gdf = gdf.to_crs(epsg=4326)

print(gdf.geometry.centroid.x.min(), gdf.geometry.centroid.x.max())  # 경도 (126~127)
print(gdf.geometry.centroid.y.min(), gdf.geometry.centroid.y.max())  # 위도 (37~38)

# 서울 중심 좌표
seoul_center = [126.9780, 37.5665]

# Mapbox Access Token 설정
token = "pk.eyJ1Ijoiam9uZ2h3YW5raW0iLCJhIjoiY21hN3Y4ZTUyMTZ5NTJucHV0NWJvY25tMyJ9.lFd9M9VQGqpbWFBGw53ozg"

# GeoDataFrame → GeoJSON 변환
# Note: df_to_geojson()을 쓰기 위해서는 geometry가 반드시 존재해야 함
gdf.to_file('seoul-geoj.geojson', driver="GeoJSON")

with open('seoul-geoj.geojson', 'rt', encoding='utf-8') as f:
    gj = geojson.load(f)

# 색상 지정 (class별)
color_stops = [
    [1, '#a1d99b'],  # 클래스1(매출 증가추세, 점포 증가추세) 연한 초록 (Pastel Green)
    [2, '#fc9272'],  # 클래스2(매출 감소추세, 점포 증가추세) 연한 빨강 (Pastel Red)
    [3, '#9ecae1'],  # 클래스3(매출 감소추세, 점포 감소추세) 연한 파랑 (Pastel Blue)
    [4, '#fdd0a2']   # 클래스4(매출 증가추세, 점포 감소추세) 연한 주황 (Pastel Orange)
]

# Choropleth 시각화 객체 생성
viz = ChoroplethViz(
    access_token=token,
    data=gj,
    color_property='class',
    color_stops=color_stops,
    center=seoul_center,
    zoom=10)

# 성장세(매출추세*점포추세의 interaction 을 높이로 설정)
viz.bearing = -15
viz.pitch = 45

def create_auto_numeric_stops(series, n_bins=10, round_base=100):
    """
    Pandas Series로부터 자동 numeric_stops 생성

    Parameters:
    - series: 숫자값 Series (예: df['interact_growth'])
    - n_bins: 구간 개수
    - round_base: 구간을 몇 단위로 반올림할지

    Returns:
    - create_numeric_stops()에서 사용할 수 있는 stops 리스트
    """
    min_val = series.min()
    max_val = series.max()

    # linspace로 구간 생성 → 정수 반올림
    raw_stops = np.linspace(min_val, max_val, n_bins)
    rounded_stops = [round(x / round_base) * round_base for x in raw_stops]

    # 중복 제거 및 정렬
    final_stops = sorted(set(rounded_stops))

    return create_numeric_stops(final_stops)


# 높이 시각화설정 
numeric_stops = create_auto_numeric_stops(gdf['interact_growth'], n_bins=8, round_base=500)

viz.height_property = 'interact_growth'
numeric_stops = numeric_stops

viz.height_stops = numeric_stops
viz.height_function_type = 'interpolate'

viz.show()

