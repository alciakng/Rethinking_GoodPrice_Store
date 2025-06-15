import pandas as pd
import plotly.express as px

from linearmodels.panel import PanelOLS
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import silhouette_score, silhouette_samples

from ui.chart_board import display_goodprice_map, plot_grouped_bar, plot_grouped_bar_ratio, save_all_clusters_goodprice_map
from util.common_util import apply_log_transform, check_outliers_std, check_variable_skewness, compute_rmsle_from_result, drop_outlier_rows_std, load_clustered_geodataframe, save_full_model_output
from util.common_util import apply_zscore_scaling

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest


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
df_GoodPrice = pd.read_csv('./model/상권_착한가격업소_병합.csv', encoding='utf-8')

# 점포수_대비_매출액 파생변수 생성
df_GoodPrice['점포수_대비_매출액'] = df_GoodPrice.apply(
    lambda row: 0 if row['점포_수'] == 0 or pd.isna(row['점포_수'])
    else int(np.floor(row['당월_매출_금액'] / row['점포_수'])),
    axis=1
)

# 타입변환 
df_GoodPrice['기준_년분기_코드'] = df_GoodPrice['기준_년분기_코드'].astype(int)
# 총_유동인구 
df_GoodPrice['총_유동인구_수'] = df_GoodPrice['남성_유동인구_수'] + df_GoodPrice['여성_유동인구_수'] 

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

# 임시코드(20234~20244 분기 한정)
df_GoodPrice = df_GoodPrice[df_GoodPrice['기준_년분기_코드'].isin([20233, 20241, 20242, 20243, 20244])]

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
# 12. Model1 회귀식 구성 (가설1,2,3,4)
# -----------------------------------
base_formula = '착한가격_업소수_비중_log ~ 1 + 점포수_대비_매출액_log + 상권_변화_지표_HL + 상권_변화_지표_LH + 상권_변화_지표_LL + 폐업_률 + 음식_지출_총금액_log + 아파트_평균_시가_log + 아파트_단지_수_log + 유동인구_10_30대_log + 유동인구_40_이상_log + 총_직장인구_수_log + 집객시설_수_log + 총_상주인구_수 + 20_30_인구비_log + 31_50_인구비_log'
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
base_formula2 = '착한가격_업소수_비중_log ~ 1 + 점포수_대비_매출액_log + 상권_변화_지표_HL + 상권_변화_지표_LH + 상권_변화_지표_LL + 폐업_률 + 음식_지출_총금액_log + 아파트_평균_시가_log + 아파트_단지_수_log + 유동인구_10_30대_log + 유동인구_40_이상_log + 총_직장인구_수_log + 집객시설_수_log + 총_상주인구_수 + 20_30_인구비_log + 31_50_인구비_log + 상권_변화_지표_HL:20_30_인구비_log + 상권_변화_지표_LH:20_30_인구비_log + 상권_변화_지표_LL:20_30_인구비_log '
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

# 최종데이터셋 export
df_reg_final = df_final.reset_index()
df_reg_final.to_csv('./model/df_reg_final.csv',encoding='utf-8-sig', index=False)


# ===========================================================
# 2. 유의한 변수를 통해 clustering
#  - 공통전처리(스케일링, SPCA)
#  [Part1. 계층적 Clustering]
#  [Part2. K-means 클러스터링]
#  [Part3. 병합 및 클러스터링 명칭부여]
# ===========================================================
df_for_cluster = pd.read_csv('./model/상권_착한가격업소_병합.csv', encoding='utf-8')

# 더미변수 생성
df_for_cluster = pd.get_dummies(df_for_cluster, columns=['상권_변화_지표'], drop_first=False)

# 왜도 확인 & 로그변환 
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

check_variable_skewness(df_for_cluster[skew_test_columns])

skew_columns = skew_test_columns.copy()
skew_columns.remove('총_상주인구_수')       # 상주인구수 대칭 
skew_columns.remove('폐업_영업_개월_차이')
skew_columns.remove('운영_영업_개월_차이')

# 최종확정된 변수를 기준으로 로그변환 
df_for_cluster = apply_log_transform(df_for_cluster, skew_columns)

# 9. 독립변수 최종확정 
# - 아파트_평균_시가_log, 집객시설 수, 운영_영업_개월_차이는 다중공선성이 높아 제거
scale_columns = [
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
    '개업_률_log',
    '폐업_률_log',
    '1인_가구비_log',
    '20_30_인구비_log',
    '31_50_인구비_log'
]

# 독립변수 단위 스케일링
df_for_cluster = apply_zscore_scaling(df_for_cluster, scale_columns)

# ▶ 유의변수 기반 클러스터링용 데이터 추출
features = [
    '점포수_대비_매출액_log',
    '상권_변화_지표_HL',
    '상권_변화_지표_LH',
    '상권_변화_지표_LL',
    '폐업_률_log',
    '음식_지출_총금액_log',
    '아파트_평균_시가_log',
    '아파트_단지_수_log',
    '유동인구_10_30대_log',
    '유동인구_40_이상_log',
    '총_직장인구_수_log',
    '집객시설_수_log',
    '총_상주인구_수',
    '20_30_인구비_log',
    '31_50_인구비_log'
]
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

# PLS(SPC 해석) 
# 이렇게 세 가지를 함께 보고 해석하면:
# 어떤 변수들이 축 구성에 중요했고 (weights)
# 실제 데이터 분산을 얼마나 설명했고 (loadings)
# 결과적으로 Y에 어떤 영향을 주는지(coef) 를 종합적으로 이해할 수 있습니다.
df_pls_weights = pd.DataFrame(pls.x_weights_, columns=['SPC1', 'SPC2'], index=features)
df_pls_loadings = pd.DataFrame(pls.x_loadings_, columns=['SPC1', 'SPC2'], index=features)
df_pls_coef = pd.DataFrame(pls.coef_.T, index=features, columns=['PLS_Coefficient'])

df_pls_weights.index.name = "feature"  # 인덱스 이름 설정
df_pls_loadings.index.name = "feature"  # 인덱스 이름 설정
df_pls_coef.index.name = "feature"  # 인덱스 이름 설정

# pls 해석을 위한 리프토
df_pls_weights.to_csv("./model/pls_weights.csv", index=True, encoding="utf-8-sig")
df_pls_loadings.to_csv("./model/pls_loadings.csv", index=True, encoding="utf-8-sig")
df_pls_coef.to_csv("./model/pls_coef.csv", index=True, encoding="utf-8-sig")

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
    'SPC1': '소비활성도 축',
    'SPC2': '청년밀집도 축'
})

# 클러스터 번호에 따른 명칭 매핑 딕셔너리
hier_cluster_labels = {
    1: '청년 밀집·고소비 지역',
    2: '청년 밀집·저소비 지역',
    3: '중장년 밀집·저소비 지역',
    4: '최대소비 지역',
}

# 클러스터 번호에 따른 명칭 매핑 딕셔너리
kmeans_cluster_labels = {
    0: '최대소비 지역',
    1: '청년 밀집·저소비 지역',
    2: '중장년 밀집·저소비 지역',
    3: '청년 밀집·고소비 지역'
}

# 새로운 컬럼 생성 (기존 숫자 클러스터 유지도 가능)
df_cluster['hier_cluster_label'] = df_cluster['hier_cluster'].map(hier_cluster_labels)
df_cluster['kmeans_cluster_label'] = df_cluster['kmeans_cluster'].map(kmeans_cluster_labels)

# 최종 클러스터링 데이터셋  
df_cluster.to_csv('./model/final_cluster.csv',encoding='utf-8-sig', index=False)


# ===========================================================
#  etc. 클러스터링 된 지도시각화 군집별로 html export
# ===========================================================
# 지역별 클러스터*착한가격업소수 비중 지도시각화
gdf = load_clustered_geodataframe()
save_all_clusters_goodprice_map(gdf)




# ===========================================================
# 3. 소상공인실태조사 전략분석
# ===========================================================
df_소상공인실태조사_그룹 = pd.read_csv('./model/소상공인실태조사_그룹화.csv')

#카이제곱검정 변수리스트
category_cols = [
    '정부지원정책_추진정책1코드명',
    '정부지원정책_추진정책2코드명',
    '사업전환_운영계획코드명',
    '경영_운영활동코드1명'
]

# 결과 저장 리스트
t_results = []

for col in category_cols:
    # 교차표 생성
    ct = pd.crosstab(df_소상공인실태조사_그룹['그룹구분'], df_소상공인실태조사_그룹[col])

    # 결측치로 인한 오류 방지를 위해 열 개수가 2개 이상인지 확인
    if ct.shape[1] < 2:
        t_results.append({
            '변수': col,
            '카이제곱 통계량': None,
            'p값': None,
            '결론': '유효한 비교 불가 (카테고리 부족)'
        })
        continue

    # 카이제곱 독립성 검정
    chi2, p, dof, expected = chi2_contingency(ct)

    # 결과 저장
    t_results.append({
        '변수': col,
        '카이제곱 통계량': round(chi2, 3),
        'p값': round(p, 4),
        '결론': '유의미한 차이 있음' if p < 0.05 else '차이 없음'
    })

# 결과 출력
t_results_df = pd.DataFrame(t_results)
print(t_results_df)



z_results = []

for col in category_cols:
    ct = pd.crosstab(df_소상공인실태조사_그룹['그룹구분'], df_소상공인실태조사_그룹[col])

    # 카테고리가 2개 이상이어야 비교 의미 있음
    if ct.shape[1] < 2:
        z_results.append({
            '변수': col,
            '비교방식': '비율 z-test',
            '결론': '유효한 비교 불가 (카테고리 부족)'
        })
        continue

    for cat in ct.columns:
        count = ct[cat].values
        nobs = ct.sum(axis=1).values

        if len(count) == 2 and all(nobs > 0):
            stat, pval = proportions_ztest(count, nobs)
            z_results.append({
                '변수': col,
                '카테고리': cat,
                'z값': round(stat, 3),
                'p값': round(pval, 4),
                '결론': '유의미한 비율 차이 있음' if pval < 0.05 else '차이 없음'
            })
        else:
            z_results.append({
                '변수': col,
                '카테고리': cat,
                '결론': '비교 불가 (샘플 부족)'
            })

z_results_df = pd.DataFrame(z_results)
print(z_results_df)


z_results_df[z_results_df['결론'] == '유의미한 비율 차이 있음']


plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'정부지원정책_추진정책1코드명')
plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'정부지원정책_추진정책2코드명')
plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'사업전환_운영계획코드명')
plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'경영_운영활동코드1명')

