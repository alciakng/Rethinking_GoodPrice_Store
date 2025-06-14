import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from ui.chart_board import display_clusterwise_goodprice_ratio, display_clusterwise_goodprice_trend, display_goodprice_map, display_html_map_in_streamlit, plot_goodprice_trend_by_quarter, save_all_clusters_goodprice_map, show_top10_chart
from ui.chart_board import display_cluster_comparison_with_expander, display_model_section, display_spc_analysis_block, display_cluster_silhouette_plot
from util.common_util import load_clustered_geodataframe

# ==========================
# 분석개요 보드
# ==========================
def overview_board():

    st.markdown("""
    <style>
    /* 탭 영역 전체 스타일 */
    div[data-baseweb="tab-list"] {
        font-size: 20px !important;    /* 글자 크기 */
        height: 60px;                  /* 탭 높이 */
    }

    /* 각 탭 버튼의 스타일 */
    button[role="tab"] {
        padding: 12px 24px !important;  /* 내부 여백 */
        font-size: 18px !important;     /* 글자 크기 */
        font-weight: bold !important;
        color: white !important;
        background-color: #1c1c1c !important;
        border-radius: 8px !important;
        margin-right: 10px !important;
    }

    /* 선택된 탭 스타일 */
    button[aria-selected="true"] {
        background-color: #0e76a8 !important; /* 선택된 탭 배경색 */
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    탭1, 탭2, 탭3 = st.tabs(["🔍 회귀분석 결과", "🔍 기본 통계치", "🔍 클러스터링별 통계"])

    # 첫 번째 탭: 연체율 추이
    with 탭1:
        # 모델1
        model1_hypotheses = [
            "H1. 외식지출비가 높은지역일수록 착한가격업소 수 비중이 감소한다. - 검증",
            "H2. 폐업률이 높은지역일수록 착한가격업소 수 비중이 증가한다 - 검증",
            "H3. 상권축소 지역일수록 착한가격업소 수 비중이 증가한다. - 검증",
            "H4. 20_30대 인구비가 높은지역일수록 착한가격업소 수 비중이 감소한다. - 기각"
        ]

        display_model_section("Model 1", model1_hypotheses, "./model/model1_results.csv")

        # 모델2
    
        model2_hypotheses = [
            "H5. (상호작용항) 20_30대 인구비의 효과는 상권지표에 따라 조절된다. - 검증"
        ]
        display_model_section("Model 2", model2_hypotheses, "./model/model2_results.csv")
        

        # 모델3
        model3_hypotheses = [
            "추가. H2, H3, H4의 lag_1 시차 독립변수를 통해 역인과성에 대한 강건성 검증"
        ]
        display_model_section("Model 3", model3_hypotheses, "./model/model3_results.csv")

    with 탭2:
        st.markdown(f"## 📊 연도별 착한가격 업소수 추이")

        df_상권_착한가격업소_병합 = pd.read_csv('./model/상권_착한가격업소_병합.csv', encoding='utf-8')
        
        with st.expander("📌 시계열그래프", expanded=True):
            st.markdown("'22.3분기, '23.3분기, '24.1~4 분기로 구성된 데이터로 분석진행")
            plot_goodprice_trend_by_quarter(df_상권_착한가격업소_병합)

        st.markdown(f"## 📍 상위 10개 행정동")

        with st.expander("📌 시계열그래프", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                show_top10_chart(df_상권_착한가격업소_병합,'업소수','착한가격업소수 상위10개')
            
            with col2:
                show_top10_chart(df_상권_착한가격업소_병합,'아파트_평균_시가','아파트_평균_시가 상위10개')
            
            with col3:
                show_top10_chart(df_상권_착한가격업소_병합,'20_30_인구비','20_30_인구비 상위10개')
            
            with col4:
                show_top10_chart(df_상권_착한가격업소_병합,'음식_지출_총금액','외식비 상위10개')

            col5, col6, col7, col8 = st.columns(4)

            with col5:
                show_top10_chart(df_상권_착한가격업소_병합,'집객시설_수','집객시설_수 상위10개')
            
            with col6:
                show_top10_chart(df_상권_착한가격업소_병합,'개업_률','개업_률 상위10개')
            
            with col7:
                show_top10_chart(df_상권_착한가격업소_병합,'폐업_률','폐업_률 상위10개')
            
            with col8:
                df_상권_착한가격업소_병합['유동인구_수'] =df_상권_착한가격업소_병합['남성_유동인구_수'] + df_상권_착한가격업소_병합['여성_유동인구_수']
                show_top10_chart(df_상권_착한가격업소_병합,'유동인구_수','유동인구_수 상위10개')

    with 탭3:
        with st.container():
            st.markdown("""
            <div style="
                border: 1px solid #1A73E8;
                border-radius: 10px;
                padding: 16px 20px;
                background-color: #0f1117;
            ">
                <h4 style="margin-top: 0; color: #f2f2f2;">클러스터링 기준</h4>
                <p style="color: #cfcfcf; font-size: 15px; line-height: 1.6;">
                    회귀분석에 사용된 독립변수 및 통제변수를 기반으로<br>
                    <strong>SPC(PLS) - Supervised PCA</strong> 기법을 활용해 2개의 주성분을 도출하고,
                    이를 바탕으로 클러스터링을 수행하였습니다.
                </p>
                <p style="color: #cfcfcf; font-size: 15px; line-height: 1.6;">
                    주성분과 클러스터링의 해석은 회귀분석에서 유의한 변수인<br>
                    <span style="color:#81C784;">상권변화지표</span>,  
                    <span style="color:#81C784;"> 폐업_률</span>,  
                    <span style="color:#81C784;"> 음식지출총금액</span>,  
                    <span style="color:#81C784;"> 20_30인구비</span>,  
                    <span style="color:#81C784;"> 총_직장인구</span>
                    를 기준으로 진행하였습니다.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # 간격 추가 (1줄)
        st.markdown("<br>", unsafe_allow_html=True)
        # spc 해석
        display_spc_analysis_block()

        # 클러스터링 분류 시각화
        df_final_cluster = pd.read_csv('./model/final_cluster.csv', encoding='utf-8')
        display_cluster_comparison_with_expander(df_final_cluster)

        # 실루엣 점수 시각화 
        display_cluster_silhouette_plot(df_final_cluster)

        # 클러스터링 별 업소수 증가추이 
        display_clusterwise_goodprice_trend(df_final_cluster)

        # 클러스터링별  업소수 비중
        display_clusterwise_goodprice_ratio(df_final_cluster)

        # 지역별 클러스터*착한가격업소수 비중 지도시각화
        gdf = load_clustered_geodataframe()
        display_goodprice_map(gdf)



# ==========================
# 분석전략 보드
# ==========================
def strategy_board(cluster_index):
    display_html_map_in_streamlit(cluster_index)