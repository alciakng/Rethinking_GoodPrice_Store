import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics import mean_squared_log_error

from scipy.stats import zscore
from scipy.stats import skew

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    n_periods = getattr(results, 'time_info', {}).get('total', None) 
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


def load_model_result(csv_path):
    df = pd.read_csv(csv_path)
    separator_idx = df[df.iloc[:, 0] == "---"].index[0]

    df_summary = df.iloc[:separator_idx].copy().reset_index(drop=True)
    df_coef = df.iloc[separator_idx+1:].copy().reset_index(drop=True)

    return df_summary, df_coef

def safe_format(x):
    try:
        return f"{float(x):.4f}"
    except:
        return x
    
def load_clustered_geodataframe(
    cluster_csv_path='./model/final_cluster.csv',
    shapefile_path='./data/sig.shp',
    target_quarter=20244
):
    """
    클러스터링 결과와 행정동 경계 데이터를 병합하여 GeoDataFrame을 생성합니다.

    Parameters:
    - cluster_csv_path (str): 클러스터 결과 CSV 파일 경로
    - shapefile_path (str): 행정동 경계 Shapefile 경로
    - target_quarter (int): 필터링할 기준 연분기 코드 (예: 20244)

    Returns:
    - gdf (GeoDataFrame): 클러스터링 정보가 포함된 공간 데이터
    """

    df_final_cluster = pd.read_csv(cluster_csv_path, encoding='utf-8')
    df_final_cluster_20244 = df_final_cluster[df_final_cluster['기준_년분기_코드']==target_quarter]

    # 행정구역 경계
    geometry = gpd.read_file(shapefile_path, encoding="utf8")
    geometry.rename(columns={'ADSTRD_CD': '행정동_코드'}, inplace=True)
    geometry['행정동_코드']= geometry['행정동_코드'].astype('str')
    df_final_cluster_20244['행정동_코드']= df_final_cluster_20244['행정동_코드'].astype('str')

    # 병합
    merged = df_final_cluster_20244.merge(geometry, on= '행정동_코드')

    # GeoDataFrame으로 변환
    gdf = gpd.GeoDataFrame(merged, geometry='geometry')
    gdf = gdf.set_crs(epsg=5181)
    gdf = gdf.to_crs(epsg=4326)

    return gdf
