import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from ui.chart_board import display_clusterwise_goodprice_ratio, display_clusterwise_goodprice_trend, display_goodprice_map, display_html_map_in_streamlit, plot_goodprice_trend_by_quarter, save_all_clusters_goodprice_map, show_top10_chart
from ui.chart_board import display_cluster_comparison_with_expander, display_model_section, display_spc_analysis_block, display_cluster_silhouette_plot, display_ztest, plot_grouped_bar_ratio
from util.common_util import load_clustered_geodataframe

# ==========================
# 분석개요 보드
# ==========================
def overview_board():


    st.markdown("""
        <h1 style='font-weight: 700;'>🚀 물가안정을 위한 <span style='color:#4CAF50;'>착한가격업소 정책</span> <span style='color:#4CAF50;'>개선 방안</span> 제안</h1>

        <p style='line-height: 2em; font-weight: 500;'>
        본 프로젝트는 <strong>행정안전부·지자체 주도의 하향식 착한가격업소 정책 도입 필요성</strong>을 문제제기하기 위하여,
        <strong>지역 상권의 특성</strong>을 반영하여 착한가격업소의 <strong>비중 및 증가 추세를 진단</strong>합니다.<br>
        <strong>본 분석의 상권은 서울특별시로 한정하였으며, 서울특별시·행정안전부·통계청 데이터를 기반으로 분석하였습니다.<br>
        <strong>진단을 통해 물가안정을 위한 정책의 목적과 상이하게, 소비력이 낮고 상권이 축소된 지역에서 </strong>
        <strong>착한가격업소수의 비중이 높다는 점을 검증함으로써 정책개선의 필요성을 제기합니다.</strong><br>            
        <strong>비지도 머신러닝 기반의 군집분석(클러스터링)</strong>을 통해 상권을 구분함으로써,
        <strong>군집별 특성에 따른 지자체 주도의 하향식(Top-Down) 정책방식과 선정·제안 전략을</strong>을 제시합니다.
        </p>

        <p style='line-height: 2em; font-weight: 500;'>
        분석은 다음의 4단계 파이프라인을 통해 수행되었습니다.
        </p>

        <ul style='line-height: 1.8em; font-weight: 500;'>
            <li>① <strong>회귀분석</strong>을 통한 착한가격업소 선정에 유의한 요인을 식별</li>
            <li>② <strong>SPC(PLS) 차원축소 분석</strong>을 통한 핵심 변수(축)을 설정</li>
            <li>③ <strong>식별된 주성분(축) 기반 클러스터링</strong>으로 상권 유형 구분</li>
            <li>④ <strong>군집별 정책 방향성 도출 및 전략적 제언</strong></li>
        </ul>

        <p style='line-height: 2em; font-weight: 500;'>
        왼쪽 사이드바에서 분석전략을 선택하여 <strong>상권별 분석 결과</strong>와 <strong>정책 제언</strong>을 확인하실 수 있습니다.
        </p>
        """, unsafe_allow_html=True)

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
def strategy_board():
    st.markdown("""
        <h1 style='font-weight: 700;'>🚀 착한가격업소 <span style='color:#4CAF50;'>군집별 맞춤전략</span> 제안</h1>

        <p style='line-height: 2em; font-weight: 500;'>
        <strong> 데이터 분석을 통해 도출된 군집을 바탕으로, 군집별 맞춤전략을 제안합니다</strong>합니다.<br>
        </p>

        <p style='line-height: 2em; font-weight: 500;'>
        전략제안은 다음의 2단계 파이프라인을 통해 수행되었습니다.
        </p>

        <ul style='line-height: 1.8em; font-weight: 500;'>
            <li>① <strong>소상공인실태조사 데이터를 통해 쇠퇴상권 및 상위매출 점주의 경영활동·애로사항·사업전환계획·필요정책 등을 파악</li>
            <li>② <strong>쇠퇴상권 및 상위매출 점주의 필요정책 및 실태를 참고하여 군집별 맞춤전략을 제안 </li>
        </ul>
        """, unsafe_allow_html=True)

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

    탭1, 탭2, 탭3 = st.tabs(["🔍 Z-test 결과", "🔍 기본통계", "🔍 전략제안"])

    with 탭1:
        df_z_result = pd.read_csv('./model/z_results.csv', encoding='utf-8')
        display_ztest(df_z_result)

    with 탭2:
        df_소상공인실태조사_그룹 = pd.read_csv('./model/소상공인실태조사_그룹화.csv')
        # 차트표시 
        plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'정부지원정책_추진정책1코드명')
        plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'정부지원정책_추진정책2코드명')
        plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'사업전환_운영계획코드명')
        plot_grouped_bar_ratio(df_소상공인실태조사_그룹,'경영_운영활동코드1명')

    with 탭3:
        html_table = """
        <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 15px;
        }
        .custom-table th {
            background-color: #3C8DBC;
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-size: 17px;
            font-weight: bold;
        }
        .custom-table td {
            padding: 10px 8px;
            text-align: center;
            vertical-align: middle;
            font-size: 16px;
            color: #000000;
        }
        .custom-table td.bold {
            font-weight: bold;
        }
        .custom-table td.normal {
            font-weight: normal;
        }
        .custom-table .youth { background-color:  #fff3e0;}
        .custom-table .middle { background-color:  #ffe5e5;}
        .custom-table .high { background-color: #e5f0ff; }
        </style>
        <table class="custom-table">
            <thead>
                <tr>
                    <th>군집구분</th>
                    <th>전략구분</th>
                    <th>주요 전략</th>
                    <th>세부 설명 및 실행 예시</th>
                </tr>
            </thead>
            <tbody>
                <!-- 청년 밀집지역 -->
                <tr class="youth">
                    <td class="bold" rowspan="5">청년 밀집지역</td>
                    <td class="bold" rowspan="2">선정</td>
                    <td class="normal">청년 선호업종 추가 선정</td>
                    <td class="normal">양식, 디저트, 퓨전식 등 대학가 인기 업종 중심</td>
                </tr>
                <tr class="youth">
                    <td class="normal">청년 참여 유도 공모제</td>
                    <td class="normal">대학생 소비자단·동아리 추천 기반 점포 모집</td>
                </tr>
                <tr class="youth">
                    <td class="bold" rowspan="3">지원</td>
                    <td class="normal">대학생 전용 카드 연계</td>
                    <td class="normal">착한카드 출시, 대학생 할인/적립 포인트 제공</td>
                </tr>
                <tr class="youth">
                    <td class="normal">카드사 수수료 감면</td>
                    <td class="normal">지역 카드사 협업을 통한 거래 비용 절감</td>
                </tr>
                <tr class="youth">
                    <td class="normal">SNS 기반 홍보 지원</td>
                    <td class="normal">MZ세대 SNS 마케팅 전문가와 소상공인 매칭</td>
                </tr>
                <!-- 중장년·저소비 밀집지역 -->
                <tr class="middle">
                    <td class="bold" rowspan="5">중장년·저소비 밀집지역</td>
                    <td class="bold" rowspan="2">선정</td>
                    <td class="normal">지역 커뮤니티 추천 기반</td>
                    <td class="normal">주민자치회, 통장추천제, 지역설명회 등</td>
                </tr>
                <tr class="middle">
                    <td class="normal">단골 중심 업소 선정</td>
                    <td class="normal">고객재방문율 높은 소형식당, 골목상권 집중</td>
                </tr>
                <tr class="middle">
                    <td class="bold" rowspan="3">지원</td>
                    <td class="normal">온누리·지역화폐 할인 연계</td>
                    <td class="normal">일정 비율 할인 시 지원금 지급 방식</td>
                </tr>
                <tr class="middle">
                    <td class="normal">공공근로자 할인 혜택</td>
                    <td class="normal">지역 노인일자리/공공근로자 대상 상시 할인</td>
                </tr>
                <tr class="middle">
                    <td class="normal">정책자금·인테리어 패키지</td>
                    <td class="normal">시설 개선, 외관 정비비용 일부 보조</td>
                </tr>
                <!-- 고소비지역 -->
                <tr class="high">
                    <td class="bold" rowspan="5">고소비지역</td>
                    <td class="bold" rowspan="2">선정</td>
                    <td class="normal">물가 앵커점 지정</td>
                    <td class="normal">대형마트 인근, 중심상권 내 대표 식당 지정</td>
                </tr>
                <tr class="high">
                    <td class="normal">지속 성과 모니터링</td>
                    <td class="normal">매출·가격 이력 자동 트래킹 → 재선정에 반영</td>
                </tr>
                <tr class="high">
                    <td class="bold" rowspan="3">지원</td>
                    <td class="normal">손실비용 보전</td>
                    <td class="normal">물가 인하분에 따른 마진 감소 일부 보전</td>
                </tr>
                <tr class="high">
                    <td class="normal">세제 감면, 인건비 보조</td>
                    <td class="normal">부가세·소득세 감면, 인건비 1년 한시 지원</td>
                </tr>
                <tr class="high">
                    <td class="normal">대표메뉴 선정 및 지원</td>
                    <td class="normal">김밥, 칼국수, 자장면 등 대표메뉴 선정·가격감면·손실비용 지원</td>
                </tr>
            </tbody>
        </table>
        """

        st.markdown(html_table, unsafe_allow_html=True)