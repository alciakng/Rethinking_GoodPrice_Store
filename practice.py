import geojson
import pandas as pd
import geopandas as gpd
import plotly.express as px
import shapely

from mapboxgl.viz import ChoroplethViz
from mapboxgl.utils import df_to_geojson
from mapboxgl.utils import create_color_stops


print("Shapely:", shapely.__version__)

# 행정구역 경계
geometry = gpd.read_file('sig.shp', encoding="utf-8")

resident = pd.read_csv('resident.csv', encoding='cp949')
apartment = pd.read_csv('apartment.csv', encoding='cp949')
income = pd.read_csv('income.csv', encoding='cp949')
workers = pd.read_csv('workers.csv', encoding='cp949')
sales = pd.read_csv('sales.csv', encoding='cp949')
facilities = pd.read_csv('facilities.csv', encoding='cp949')
population = pd.read_csv('population.csv', encoding='cp949')
market = pd.read_csv('market.csv')

resident.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
apartment.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
income.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
workers.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
sales.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
facilities.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
population.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)
market.rename(columns={'행정동_코드':'ADSTRD_CD'},inplace=True)

#sales_20244
sales_20244= sales[sales['기준_년분기_코드']==20244]
sales_20244[sales_20244['ADSTRD_CD']==11500630]

#market_20244
market_20244= market[(market['기준_년분기_코드']==20234) & (market['서비스_업종_코드']=='CS100001')]

geometry['ADSTRD_CD'] = geometry['ADSTRD_CD'].astype(str)
market_20244['ADSTRD_CD'] = market_20244['ADSTRD_CD'].astype(str)

merged = geometry.merge(market_20244, on= 'ADSTRD_CD')

merged['점포_수'].describe()

# GeoDataFrame으로 변환
gdf = gpd.GeoDataFrame(merged, geometry='geometry')

gdf.info()

# CRS 변환
gdf = gdf.set_crs(epsg=5179)
gdf = gdf.to_crs(epsg=4326)

gdf.head()
print(gdf.geometry.centroid.x.min(), gdf.geometry.centroid.x.max())  # 경도 (126~127)
print(gdf.geometry.centroid.y.min(), gdf.geometry.centroid.y.max())  # 위도 (37~38)
print(gdf.crs)
# 서울 중심 좌표
seoul_center = [129.4880, 23.7065]

# Mapbox Access Token 설정
token = "pk.eyJ1Ijoiam9uZ2h3YW5raW0iLCJhIjoiY21hN3Y4ZTUyMTZ5NTJucHV0NWJvY25tMyJ9.lFd9M9VQGqpbWFBGw53ozg"

# GeoDataFrame → GeoJSON 변환
# Note: `df_to_geojson()`을 쓰기 위해서는 geometry가 반드시 존재해야 함
gdf.to_file('seoul-geoj.geojson', driver="GeoJSON")

with open('seoul-geoj.geojson', 'rt', encoding='utf-8') as f:
    gj = geojson.load(f)

color_breaks = [0, 10, 30, 50, 75, 100, 150, 500, 1000]
color_stops = create_color_stops(color_breaks, colors='BuPu')


# Choropleth 시각화 객체 생성
viz = ChoroplethViz(
    access_token=token,
    data=gj,
    color_property='점포_수',
    color_stops=color_stops,
    center=seoul_center,
    zoom=10)

viz.show()