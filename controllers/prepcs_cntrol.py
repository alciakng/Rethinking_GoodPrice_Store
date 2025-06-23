import pandas as pd
import geopandas as gpd
import numpy as np
import requests                      
import Levenshtein
 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
quarters = ['20224','20231','20232','20233','20234','20241','20242','20243','20244']
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
            structure = result.get('structure', {})

            # 행정동 추출: 행정동 없으면 법정동 fallback
            dong_name = structure.get('level4A') or structure.get('level4L')
            dong_code = structure.get('level4AC') or structure.get('level4LC')

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
    "20224": "./data/착한가격업소_20224.csv",
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
GoodPrices_서울특별시.to_csv('./data/착한가격업소.csv',encoding='utf-8-sig', index=False)

# 4. 행정동 매핑 안된 곳 추가정제 (수기)
GoodPrices_서울특별시[GoodPrices_서울특별시['행정동_명'].isna()]

# 수기 수정후 csv 불러오기 
df_착한가격업소_수정후 = pd.read_csv('./data/착한가격업소.csv', encoding='utf-8')
df_착한가격업소_누락분 = df_착한가격업소_수정후[df_착한가격업소_수정후['행정동_명'].isna()]

# 매핑
df_주소_행정동매핑_누락분 = get_dong_info_parallel(df_착한가격업소_누락분['주소'])

# 누락분 확인
df_주소_행정동매핑_누락분[df_주소_행정동매핑_누락분['행정동_명'].isna()]['주소'].value_counts()
df_주소_행정동매핑_누락분_unique = df_주소_행정동매핑_누락분.drop_duplicates(subset='주소')

# 매핑 
mask = df_착한가격업소_수정후['행정동_명'].isna()

map_명 = df_주소_행정동매핑_누락분_unique.set_index('주소')['행정동_명']
map_코드 = df_주소_행정동매핑_누락분_unique.set_index('주소')['행정동_코드']

df_착한가격업소_수정후.loc[mask, '행정동_명'] = df_착한가격업소_수정후.loc[mask, '주소'].map(map_명)
df_착한가격업소_수정후.loc[mask, '행정동_코드'] = df_착한가격업소_수정후.loc[mask, '주소'].map(map_코드)

# 최종파일 export 
df_착한가격업소_수정후.to_csv('./data/착한가격업소_병합.csv',encoding='utf-8-sig', index=False)

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
# ------------------------------------------------------------------
df_상권데이터= pd.read_csv('./data/상권데이터.csv', encoding='utf-8')
df_착한가격업소_요약 = pd.read_csv('./data/착한가격업소_요약.csv', encoding='utf-8')
#df_지역별_임대료 = pd.read_csv('./data/지역별_임대료.csv', encoding='utf-8')

# 컬럼형식 변경 
df_상권데이터[['기준_년분기_코드','행정동_코드']] = df_상권데이터[['기준_년분기_코드','행정동_코드']].astype('str')
df_착한가격업소_요약[['기준_년분기_코드','행정동_코드']] = df_착한가격업소_요약[['기준_년분기_코드','행정동_코드']].astype('str')
#df_지역별_임대료[['기준_년분기_코드','행정동_코드']] = df_지역별_임대료[['기준_년분기_코드','행정동_코드']].astype('str')

# 2023~24년도 데이터만 필터링
quarters = ('20233', '20241', '20242', '20243', '20244')

df_base_2023_2024 = df_상권데이터[df_상권데이터['기준_년분기_코드'].isin(quarters)]
#df_지역별_임대료_2023_2024 = df_지역별_임대료[df_지역별_임대료['기준_년분기_코드'].isin(quarters)]
df_착한가격업소_요약_2023_2024 = df_착한가격업소_요약[df_착한가격업소_요약['기준_년분기_코드'].isin(quarters)]

# 행정동코드 통일 
#df_지역별_임대료_2023_2024['행정동_코드'] = df_지역별_임대료_2023_2024['행정동_코드'].str[:8]
df_착한가격업소_요약_2023_2024['행정동_코드'] = df_착한가격업소_요약_2023_2024['행정동_코드'].str[:8]

# data merge 
merge_keys = ['기준_년분기_코드','행정동_코드']

# 병합
df_GoodPrice = df_base_2023_2024.merge(df_착한가격업소_요약_2023_2024, on=merge_keys, how='left')

# 임시 결측치 제거 
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


# =======================================
# 3. 전략을 위한 소상공인실태조사 데이터 
# =======================================
# 1. 코드 정의 테이블 불러오기
df_소상공인실태조사_코드 = pd.read_csv('./data/소상공인실태조사_코드.csv')
# 2. 실데이터 불러오기
df_소상공인실태조사 = pd.read_csv('./data/소상공인실태조사.csv', encoding='cp949')
# 3. 구분별로 매핑 수행
for category in df_소상공인실태조사_코드['구분'].unique():
    # 매핑 대상이 실데이터셋에 있는 경우만 수행
    if category in df_소상공인실태조사.columns:
        # 해당 구분에 대해 코드 → 코드명 딕셔너리 생성
        submap = df_소상공인실태조사_코드[df_소상공인실태조사_코드['구분'] == category]
        mapping_dict = dict(zip(submap['코드'], submap['코드명']))
        
        # 새로운 컬럼명 지정
        new_col = category + '명'
        
        # 매핑 수행
        df_소상공인실태조사[new_col] = df_소상공인실태조사[category].map(mapping_dict)

# 결과 확인
print(df_소상공인실태조사.head())

# 분석을 위한 상권쇠퇴 그룹, 상위매출(영업이익 >50백만원)그룹 구분 
df_상권쇠퇴 = df_소상공인실태조사[(df_소상공인실태조사['경영_애로사항1코드명']=='상권쇠퇴') & (df_소상공인실태조사['경영_영업이익'] < 50) & (df_소상공인실태조사['산업중분류코드명'] == '음식점 및 주점업')]
df_상위매출 = df_소상공인실태조사[(df_소상공인실태조사['경영_애로사항1코드명']!='상권쇠퇴') & (df_소상공인실태조사['경영_영업이익'] >= 50)& (df_소상공인실태조사['산업중분류코드명'] == '음식점 및 주점업')]

# 그룹명 구분 
df_상권쇠퇴['그룹구분'] = '상권쇠퇴'
df_상위매출['그룹구분'] = '상위매출'

# 데이터 병합
df_merge = pd.concat([df_상권쇠퇴, df_상위매출])

# 데이터 export
df_merge.to_csv('./model/소상공인실태조사_그룹화.csv',encoding='utf-8-sig', index=False)
