import geojson
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from mapboxgl.viz import ChoroplethViz
from mapboxgl.utils import df_to_geojson
from mapboxgl.utils import create_color_stops
from mapboxgl.utils import create_numeric_stops

# ==========================
# mapbox, color_stop 생성함수
# ==========================
def generate_color_stops_from_quantiles(series):
    
    #시리즈 값의 quantile 구간에 따라 color_stops 생성 (내장 pastel blue 색상 사용)

    #Parameters:
    #    series (pd.Series): 시각화에 사용할 수치형 데이터

    #Returns:
    #    list: mapboxgl에 사용할 color_stops 리스트
    
    # 기본 pastel blue 색상 팔레트 (연한색 → 진한색)
    color_palette = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']

    # 값이 모두 동일하면 단일 색상 매핑
    if series.nunique() == 1:
        return [[series.iloc[0], color_palette[-1]]]

    # Quantile 기반 구간 생성
    bins = np.quantile(series, np.linspace(0, 1, len(color_palette)))
    return create_color_stops(bins.tolist(), colors=color_palette)


# ==========================
# mapbox 높이 weight 조절함수
# ==========================
def create_auto_numeric_stops(series, n_bins=10, round_base=100):
    
    # Pandas Series로부터 자동 numeric_stops 생성
    
    # Parameters:
    # - series: 숫자값 Series (예: df['interact_growth'])
    # - n_bins: 구간 개수
    # - round_base: 구간을 몇 단위로 반올림할지

    # Returns:
    # - create_numeric_stops()에서 사용할 수 있는 stops 리스트
    
    min_val = series.min()
    max_val = series.max()

    # linspace로 구간 생성 → 정수 반올림
    raw_stops = np.linspace(min_val, max_val, n_bins)
    rounded_stops = [round(x / round_base) * round_base for x in raw_stops]

    # 중복 제거 및 정렬
    final_stops = sorted(set(rounded_stops))

    return create_numeric_stops(final_stops)


# =============
# mapbox 시각화
# =============
def class_visualiztion(df : pd.DataFrame, cond : str):

    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # 좌표변환
    gdf = gdf.set_crs(epsg=5181)
    gdf = gdf.to_crs(epsg=4326)

    # 필터 조건 적용
    if cond in [1, 2, 3, 4]:
        gdf = gdf[gdf['class'] == cond]

    # 서울 중심 좌표
    seoul_center = [126.9780, 37.5665]

    # Mapbox Access Token 설정
    token = "pk.eyJ1Ijoiam9uZ2h3YW5raW0iLCJhIjoiY21hN3Y4ZTUyMTZ5NTJucHV0NWJvY25tMyJ9.lFd9M9VQGqpbWFBGw53ozg"

    # GeoDataFrame → GeoJSON 변환
    # Note: df_to_geojson()을 쓰기 위해서는 geometry가 반드시 존재해야 함
    gdf.to_file('seoul-geoj.geojson', driver="GeoJSON")

    with open('seoul-geoj.geojson', 'rt', encoding='utf-8') as f:
        gj = geojson.load(f)

    # 시각화
    if cond == 0: # 전체
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
        
        viz.show()
    else: # 클래스별 시각화 
        color_stops = generate_color_stops_from_quantiles(gdf['interact_growth'])
        # Choropleth 시각화 객체 생성 (interact_growth 컬러선택)
        viz = ChoroplethViz(
            access_token=token,
            data=gj,
            color_property='interact_growth',
            color_stops=color_stops,
            center=seoul_center,
            zoom=10)
        
        # 높이 시각화설정(매출추세*점포추세의 interaction 을 높이로 설정)
        viz.bearing = -15
        viz.pitch = 45

        numeric_stops = create_auto_numeric_stops(gdf['interact_growth'], n_bins=8, round_base=500)

        viz.height_property = 'interact_growth'
        numeric_stops = numeric_stops

        viz.height_stops = numeric_stops
        viz.height_function_type = 'interpolate'

        viz.show()

    

