import json
import pandas as pd
import streamlit as st
from util.common_util import load_model_result, safe_format
import plotly.express as px
import pydeck as pdk

# ---------------------------
# 테이블 백그라운드 강조 
# ---------------------------
def style_dataframe(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    # Coef: 절댓값 기준 정규화 → 진한 파랑으로 그라데이션
    try:
        max_coef = max(abs(float(c)) for c in df["Coef."] if isinstance(c, (int, float)) or str(c).replace('.', '', 1).replace('-', '', 1).isdigit())
    except:
        max_coef = 1  # fallback to avoid division by zero

    # Coef 색상: 절댓값 클수록 진한 파랑
    for i in df.index:
        try:
            coef = float(df.loc[i, "Coef."])
            norm = min(abs(coef) / max_coef, 1.0)            
            # 절댓값이 작을수록 밝은색 → 반전
            if norm > 0.5:
                styles.loc[i, "Coef."] = "background-color: #0D47A1; color: white"
            elif norm > 0.3:
                styles.loc[i, "Coef."] = "background-color: #1976D2; color: white"
            elif norm > 0.1:
                styles.loc[i, "Coef."] = "background-color: #64B5F6; color: black"
            else:
                styles.loc[i, "Coef."] = "background-color: #BBDEFB; color: black"
        except:
            pass

    # P-Value 조건부 색상
    for i in df.index:
        try:
            pval = float(df.loc[i, "P-Value"])
            if pval < 0.001:
                styles.loc[i, "P-Value"] = "background-color: #0D47A1; color: white"
            elif pval < 0.01:
                styles.loc[i, "P-Value"] = "background-color: #1976D2; color: white"
            elif pval < 0.05:
                styles.loc[i, "P-Value"] = "background-color: #BBDEFB; color: black"
        except:
            pass

    return styles


# ---------------------------
# model 결과 요약출력
# ---------------------------
def display_model_section(title, hypotheses, csv_path):
    st.markdown(f"## 📊 {title}")

    # 가설 표시
    with st.expander("📌 가설", expanded=True):
        for h in hypotheses:
            st.markdown(f"- {h}")

    # 결과 로드
    df_summary, df_coef = load_model_result(csv_path)

    # 요약 통계: 마크다운
    with st.expander("📌 모델 요약 통계", expanded=True):
        col1, col2 = st.columns(2)

        for i, (_, row) in enumerate(df_summary.iterrows()):
            metric = row["Metric"]
            value = safe_format(row["Value"])

            # 홀수 인덱스는 왼쪽, 짝수 인덱스는 오른쪽
            if i % 2 == 0:
                col1.markdown(f"- **{metric}**: {value}")
            else:
                col2.markdown(f"- **{metric}**: {value}")

    # Coef 색 강조 함수
    def highlight_coef(val):
        try:
            val = float(val)
            normalized = min(abs(val) / 2, 1.0)  # ±2 기준으로 최대
            color = f"rgba(30, 136, 229, {normalized})"  # 파란색 계열
            return f"background-color: {color}"
        except:
            return ""

    # 스타일 적용 및 시각화
    styled_df = (
        df_coef.style
        .apply(style_dataframe, axis=None)
        .format({col: safe_format for col in df_coef.columns if col != "Variable"})
    )

    with st.expander("📌 회귀계수", expanded=True):
        st.dataframe(styled_df)

# ---------------------------
# SPC(PLS) 결과출력 
# ---------------------------
def display_spc_analysis_block(csv_path="./model/pls_results.csv"):
    df = pd.read_csv(csv_path)

    # 'feature' 컬럼을 인덱스로 설정 (또는 변수명 컬럼으로 사용)
    if "feature" in df.columns:
        df.set_index("feature", inplace=True)

    # SPC 계수를 숫자로 안전하게 변환 (형식 문제 방지)
    df["SPC1"] = pd.to_numeric(df["SPC1"], errors="coerce")
    df["SPC2"] = pd.to_numeric(df["SPC2"], errors="coerce")

    # 강조 색상 함수
    def highlight_abs_gradient(val):
        try:
            norm = min(abs(val), 1.0)
            return f"background-color: rgba(33, 150, 243, {0.2 + 0.8 * norm}); color: white"
        except:
            return ""

    # 스타일 적용
    styled = df.style \
        .applymap(highlight_abs_gradient, subset=["SPC1", "SPC2"]) \
        .format("{:.4f}")

    # Streamlit 출력
    with st.container():
        # SPC 기법 설명
        st.markdown("""
        <div style="
            border: 1px solid #4A90E2;
            border-radius: 10px;
            padding: 18px 20px;
            background-color: #0f1117;
            color: #f2f2f2;
            margin-bottom: 20px;
        ">
            <h4 style="margin-top: 0;">🧠 SPC(PLS) - Supervised PCA 기법 소개</h4>
            <p style="font-size: 15px; line-height: 1.6;">
            <strong>SPC (Supervised Principal Component Analysis)</strong>는 기존 PCA와 달리 종속변수(Y)의 정보를 활용하여  
            예측력(분류 정확도, 회귀 적합도)을 최대화하는 방향으로 주성분을 구성하는 <strong>지도 기반 차원 축소 기법</strong>입니다.
            </p>
            <p style="font-size: 15px; line-height: 1.6;">
            본 분석에서는 회귀모형의 종속변수인 <code>착한가격업소 비중</code>과의 설명력을 최대화하는 방향으로  
            독립변수 공간을 재구성하였습니다.
            </p>
            <p style="font-size: 15px; line-height: 1.6;">
            사용한 기법은 <strong>PLS (Partial Least Squares)</strong>로, 변수 간 다중공선성을 효과적으로 제어하고  
            핵심 정보만을 압축하여 <u>클러스터링 기반 지역 분류</u>를 수행할 수 있도록 합니다.
            </p>
            <p style="font-size: 15px; line-height: 1.6;">
            이를 통해 해석 가능한 2개의 주성분을 추출하여  
            지역별 클러스터를 시각화하고 특성을 분석하였습니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("SPC 계수영향도 해석", expanded=True):
        # 계수 테이블 시각화
        st.markdown("### SPC 계수 영향도")
        st.dataframe(styled)

        # 주성분 해석
        st.markdown("### 주성분 해석 (SPC1, SPC2)")
        st.markdown("""
        - **SPC1**은 `음식지출총금액_log`와, `상권_변화_지표_HL`은 음의 방향으로 기여하고 있어  
          → **소비력이 좋은 상권**을 반영하는 축으로 해석됩니다.

        - **SPC2**는 `20_30_인구비_log`, `폐업률_log` 계수가 높고 `상권_변화_지표_HL` 계수가 상대적으로 높아  
          → **청년 인구·소비 절제**을 나타내는 축으로 해석됩니다.
        """)


def display_cluster_comparison_with_expander(df_cluster):
    with st.expander("SPC 기반 클러스터링", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📌 계층적 클러스터링 결과")
            fig_hier = px.scatter(
                df_cluster,
                x='소비활성도_축',
                y='2030_소비절제_축',
                color=df_cluster['hier_cluster_label'].astype(str),
                title='[계층적 클러스터링] Supervised PCA 기반 군집 결과',
                labels={'hier_cluster_label': 'Cluster'},
                hover_data={'소비활성도_축': ':.2f', '2030_소비절제_축': ':.2f'}
            )
            st.plotly_chart(fig_hier, use_container_width=True)

        with col2:
            st.markdown("#### 📌 K-Means 클러스터링 결과")
            fig_kmeans = px.scatter(
                df_cluster,
                x='소비활성도_축',
                y='2030_소비절제_축',
                color=df_cluster['kmeans_cluster_label'].astype(str),
                title='[K-MeANS 클러스터링] Supervised PCA 기반 군집 결과',
                labels={'kmeans_cluster_label': 'Cluster'},
                hover_data={'소비활성도_축': ':.2f', '2030_소비절제_축': ':.2f'}
            )
            st.plotly_chart(fig_kmeans, use_container_width=True)

        # 해석
        st.markdown("### 클러스터링 결과 해석")
        st.markdown("""
        - 특히, **K-Means 클러스터링**은 군집 간 중심 거리 기준으로 구획되기 때문에  
          **경계가 보다 명확**하게 나타나는 반면,  
          **계층적 클러스터링**은 유사한 특성끼리의 **서브 군집 형성**이 강조된다는 점에서  
          두 방법 간 차이를 보입니다.

        - 두 방법 모두 **2030 소비절제 축**과 **소비활성도 축**의 조합을 기반으로  
          지역의 소비 성향을 효과적으로 분류하고 있으나,  
          **K-Means 기법이 소비 상권의 특성을 보다 뚜렷하게 반영하는 군집 구조를 형성**하기 때문에  
          본 분석에서는 **K-Means 기반의 클러스터 결과를 중심으로 지역을 분류하고 해석을 진행**하였습니다.
        """)



# -------------------------------------------
# Func. create_map_with_legend
# - pydeck 범례 지도시각화 (클러스터*착한가격업소 비중)
# -------------------------------------------
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

    print(f"📍범례가 포함된 지도가 '{filename}'로 저장되었습니다.")
    return filename


# ------------------------------------------------------------------
#   [Part1. 그래프 시각화]
#   - 연도별 착한가격업소 증가추이 
#   - 클러스터링 상권별 착한가격업소수 비율 (파이차트)
#   - 클러스터링 상권별 착한가격업소 증가추이 (선그래프 차트)
# ------------------------------------------------------------------

# ----------------------------
# 연도별 착한가격업소 증가추이 
# ----------------------------
def plot_goodprice_trend_by_quarter(df):
    # 분기별 업소 수 집계
    df_trend = df.groupby('기준_년분기_코드')['업소수'].sum().reset_index()
    df_trend['기준_년분기_코드'] = df_trend['기준_년분기_코드'].astype(str)

    # 라인 차트 생성
    fig = px.line(
        df_trend,
        x='기준_년분기_코드',
        y='업소수',
        title='기준 분기별 착한가격업소 수 변화',
        labels={'기준_년분기_코드': '기준 분기', '업소수': '총 업소수'},
        markers=True
    )

    # 그래프 스타일 조정
    fig.update_layout(
        xaxis_title="기준 분기",
        yaxis_title="업소수",
        template='plotly_white',
        hovermode='x unified',
        title_x=0.5
    )

    # Streamlit 출력
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# 행정동별 상위 10개 표현하는 함수 
# ----------------------------
# 공통 유틸 함수
def show_top10_chart(df, column, title, sort_ascending=False):
    df_filtered = df[df['기준_년분기_코드'] == 20244]
    df_top10 = df_filtered.sort_values(by=column, ascending=sort_ascending).head(10)
    st.markdown(f"###### {title}")
    st.dataframe(df_top10[['행정동', column]].reset_index(drop=True))
# --------------------------------------------
# 클러스터링 상권별 착한가격업소수 비율 (파이차트)
# --------------------------------------------
def display_clusterwise_goodprice_ratio(df, cluster_col='hier_cluster_label'):
    with st.expander("클러스터별 착한가격업소 비중 (파이차트)", expanded=True):
        
        # 클러스터별 평균 업소 수 및 점포 수 → 비중 계산
        df_grouped = (
            df.groupby(cluster_col)[['점포_수', '업소수']]
            .mean()
            .reset_index()
        )
        df_grouped['착한가격_업소수_비중'] = (df_grouped['업소수'] / df_grouped['점포_수']).round(3)

        # 파이 차트 생성
        fig = px.pie(
            df_grouped,
            values='착한가격_업소수_비중',
            names=cluster_col,
            title='클러스터별 착한가격업소 비중',
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label')

        # 출력
        st.plotly_chart(fig, use_container_width=True)

        # 해석 마크다운 예시 (선택 사항)
        st.markdown("""
        - 각 클러스터의 전체 점포 대비 착한가격업소 평균 비중을 비교한 결과입니다.
        - 극_저소비지역 및 소비_비활성지역에서 전체 점포수대비 착한가격업소 비중이 높게 나타납니다.  
          이는 **소비 여력이 낮은 지역이 착한가격업소 비중이 높다는 것**을 의미하며,
          물가안정을 고려했을 때 지자체 주도의 상향식 선정 및 군집별 차별화된 전략마련이 필요함을 시사합니다.
        """)

# --------------------------------------------
# 클러스터링 상권별 착한가격업소 증가추이 (선그래프)
# --------------------------------------------
def display_clusterwise_goodprice_trend(df):
    with st.expander("📈 클러스터별 착한가격업소 비중 추이", expanded=True):

        # 데이터 전처리
        df_grouped = (
            df.groupby(['kmeans_cluster_label', '기준_년분기_코드'])[['업소수', '점포_수']]
            .sum()
            .reset_index()
        )
        df_grouped['업소수_비중'] = df_grouped['업소수'] / df_grouped['점포_수']
        df_grouped['기준_년분기_코드'] = df_grouped['기준_년분기_코드'].astype(str)

        # Plotly 시각화
        fig = px.line(
            df_grouped,
            x='기준_년분기_코드',
            y='업소수_비중',
            color='kmeans_cluster_label',
            markers=True,
            title='클러스터별 착한가격업소 비중 추이',
            labels={
                '기준_년분기_코드': '기준 분기',
                '업소수_비중': '비중',
                'kmeans_cluster_label': '클러스터'
            }
        )
        fig.update_layout(template='plotly_white')

        # Streamlit 출력
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        ### 클러스터별 착한가격업소 비중 추이 해석

        - 전체적으로 모든 클러스터에서 **시간이 지남에 따라 착한가격업소 비중이 증가하는 경향**이 나타납니다.

        - 특히 `전연령_극_저소비지역`에서는 비중 증가 폭이 가장 크고 가파르며,`전연령_소비_비활성지역` 또한 기울기가 상대적으로 가파른것을 확인가능합니다.
          이는 저소비 지역에서의 상대적 확산 효과가 크게 나타난 것으로 해석됩니다.

        - `청년_고소비지역`과 `중장년_고소비지역`은  
           다른 클러스터 대비 **착한가격업소 비중이 낮게 유지**되고 있으며,  
           이는 **소비 수요가 안정적이거나 가격 민감도가 상대적으로 낮은 지역의 특성**을 나타납니다.

        - 결과적으로, **소비 여력이 낮은 지역이 착한가격업소 확산이 더 뚜렷하게 나타나고 있음**을 보여줍니다.
          이는 물가안정을 위한 착한가격업소 제도의 상향식 선정방식(국민추천, 직접신청)의 개선필요성을 시사합니다.
        """)

# ------------------------------------------------------------------
#   [Part2. 지도 시각화]
#   - 착한가격업소 분포 점 시각화 (mapbox)
# ------------------------------------------------------------------
def display_kmeans_cluster_legend():
    """
    클러스터링 결과의 군집 범례를 Streamlit UI로 표시하는 함수.
    """
    # 클러스터 번호에 따른 명칭 매핑
    kmeans_cluster_labels = {
        3: '전연령_극_저소비지역',
        0: '전연령_소비_비활성지역',
        1: '중장년_고소비지역',
        2: '청년_고소비지역'
    }

    # 색상 매핑 (RGB)
    cluster_colors = {
        0: "rgb(255, 165, 0)",  # 주황
        1: "rgb(0, 128, 0)",    # 초록
        2: "rgb(0, 0, 255)",    # 파랑
        3: "rgb(255, 0, 0)",    # 빨강
    }

    for cid, label in kmeans_cluster_labels.items():
        color = cluster_colors[cid]
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 18px; height: 18px; margin-right: 10px;
                        background-color: {color}; border-radius: 2px;"></div>
            <span style="color: #ddd; font-size: 18px;">군집 {label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-top: 15px; font-size: 18px; color: #bbb;">
            👉 클러스터는 외식지출, 폐업률, 20‒30대 인구비율, 상권변화지표 기반으로 구분되었습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_goodprice_map(gdf, map_json_path='./util/map.json'):
    # 색상 매핑
    color_map = {
        0: [255, 165, 0],     # 전연령_소비_비활성지역 → 주황
        1: [0, 128, 0],       # 중장년_고소비지역 → 초록
        2: [0, 0, 255],       # 청년_고소비지역 → 파랑
        3: [255, 0, 0],       # 전연령_극_저소비지역 → 빨강
    }
    gdf[['r', 'g', 'b']] = pd.DataFrame(gdf['kmeans_cluster'].map(color_map).tolist(), index=gdf.index)

    # Polygon 좌표 추출
    def polygon_to_coords(geom):
        if geom.geom_type == 'Polygon':
            return list(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            return list(geom.geoms[0].exterior.coords)
        return []

    gdf['coordinates'] = gdf['geometry'].apply(polygon_to_coords)

    # 중심점
    gdf['centroid'] = gdf['geometry'].centroid
    gdf['lon'] = gdf['centroid'].x
    gdf['lat'] = gdf['centroid'].y

    # 영문 행정동명 매핑
    with open(map_json_path, encoding='utf-8') as f:
        dong_map = json.load(f)
    gdf['행정동_영문'] = gdf['행정동'].map(dong_map)

    df_for_pydeck = gdf.drop(columns=['geometry', 'centroid']).copy()

    # PolygonLayer
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        df_for_pydeck,
        get_polygon="coordinates",
        get_fill_color=["r", "g", "b"],
        get_elevation="착한가격_업소수_비중",
        elevation_scale=10000,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    # TextLayer
    text_layer = pdk.Layer(
        "TextLayer",
        df_for_pydeck,
        get_position=["lon", "lat", 1500],
        get_text="행정동_영문",
        get_size=5,
        get_color=[255, 255, 255],
        billboard=True,
        pickable=False,
    )

    view_state = pdk.ViewState(
        longitude=126.9780,
        latitude=37.5665,
        zoom=10,
        pitch=45,
        bearing=0
    )

    # pydeck 객체 생성
    deck = pdk.Deck(
        layers=[polygon_layer, text_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>행정동:</b> {행정동}<br><b>착한가격업소 비중:</b> {착한가격_업소수_비중}",
            "style": {"backgroundColor": "#2c3e50", "color": "white"}
        }
    )

    # Streamlit Expander 내 차트 렌더링
    with st.expander("📍 지역별 클러스터 × 착한가격업소 비중 분석", expanded=True):
        display_kmeans_cluster_legend()
        st.pydeck_chart(deck, use_container_width=True)
        st.markdown("""
                    ###  지역별 클러스터별 착한가격업소 분포 해석

                    - 본 지도는 **서울시 행정동 단위**로 K-Means 클러스터링 결과를 시각화한 것입니다.  
                    각 클러스터는 상권 특성에 따라 4가지로 구분되며,  
                    착한가격업소 비중의 높고 낮음은 3D Bar(높이)로 표시됩니다.

                    - 지도에서 확인할 수 있듯이, **빨간색(전연령_극_저소비지역)**과  
                    **주황색(전연령_소비_비활성지역)** 영역에서 **착한가격업소 비중이 상대적으로 높게 나타납니다.**  
                    이는 저소비 지역에서 착한가격업소 제도가 상대적으로 더 확산되고 있음을 의미합니다.

                    - 반면, **고소비 지역(초록색: 중장년 / 파란색: 청년)**에서는  
                    착한가격업소의 비중이 낮게 유지되는 경향이 포착됩니다.  
                    이는 **가격 경쟁력보다는 브랜드/품질 선호가 우선되는 소비 성향**이 반영된 결과일 수 있습니다.

                    ---

                    이러한 결과는 착한가격업소 제도가  
                    **정작 물가 부담이 큰 지역보다는 수요가 낮은 지역에 집중되고 있다는 점**에서  
                    **보다 전략적인 배치 및 선정 기준 개선**이 필요함을 시사합니다.

                    **단순 추천 기반의 상향식 제도 운영을 넘어**,  
                    **지역 소비역동성에 따른 하향식 선정 기준 도입**이 제도 실효성 강화를 위해 필요합니다.
                    """)

