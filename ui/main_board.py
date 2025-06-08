import streamlit as st
from streamlit_option_menu import option_menu

from ui.analysis_board import overview_board


# ---------------------------
# 사이드바 함수 
# ---------------------------
def sidebar_board():

    with st.sidebar:
        selected = option_menu(
            menu_title='항목선택',
            options=[
                "분석개요",
                "Cluster1 전략: 소비침체지역",
                "Cluster2 전략: 소비비활성지역",
                "Cluster3 전략: 중장년 고소비지역",
                "Cluster4 전략: 청년 고소비지역"
            ],
            icons=["none"] * 5,
            menu_icon=["none"],
            default_index=0,
            styles={
                "nav-link": {"font-size": "15px", "text-align": "left", "font_weight":"bold"},
                "nav-link-selected": {"background-color": "#4CAF50", "font-weight": "bold", "color": "white"}
            }
        )
    
    return selected

# ---------------------------
# 메인화면 조건부 표시
# ---------------------------
def main_board():

    selected = sidebar_board()

    if selected.startswith("분석개요"):
        st.markdown("""
        <h1 style='font-weight: 900;'>🚀 <span style='color:#4CAF50;'>상향식 착한가격업소 정책 설정</span> 및 <span style='color:#4CAF50;'>활성화 방안</span> 제안</h1>

        <p style='line-height: 2em; font-weight: 500;'>
        본 프로젝트는 <strong>행정안전부의 착한가격업소 정책의 효율적인 운영</strong>을 목적으로,
        <strong>지역 상권의 특성</strong>을 반영하여 착한가격업소의 <strong>비중 및 증가 추세를 진단</strong>하고,<br>
        <strong>비지도 머신러닝 기반의 군집분석(클러스터링)</strong>을 통해 상권을 구분함으로써,
        <strong>군집별 특성에 따른 지자체 주도의 상향식 정책설정 필요성과 활성화 전략</strong>을 제안합니다.
        </p>

        <p style='line-height: 2em; font-weight: 500;'>
        분석은 다음의 4단계 파이프라인을 통해 수행되었습니다.
        </p>

        <ul style='line-height: 1.8em; font-weight: 500;'>
            <li>① <strong>회귀분석</strong>을 통한 착한가격업소 결정요인 검증</li>
            <li>② <strong>SPC(PLS) 주성분 분석</strong>을 통한 핵심 변수 축소</li>
            <li>③ <strong>주성분 기반 클러스터링</strong>으로 상권 유형 구분</li>
            <li>④ <strong>군집별 정책 방향성 도출 및 전략적 제언</strong></li>
        </ul>

        <p style='line-height: 2em; font-weight: 500;'>
        왼쪽 사이드바에서 클러스터링 전략을 선택하여 <strong>상권별 분석 결과</strong>와 <strong>정책 제언</strong>을 확인하실 수 있습니다.
        </p>
        """, unsafe_allow_html=True)

        # overview board()
        overview_board()

    elif "Cluster1" in selected:
        st.header("클러스터링1 전략: 소비침체지역")
        # TODO: 시각화1~3 또는 지도 시각화 함수 호출

    elif "Cluster2" in selected:
        st.header("클러스터링2 전략: 전연령 소비비활성지역")
        # TODO: 해당 클러스터 시각화 함수 호출

    elif "Cluster3" in selected:
        st.header("클러스터링3 전략: 중장년 고소비지역")
        # TODO: 해당 클러스터 시각화 함수 호출

    elif "Cluster4" in selected:
        st.header("클러스터링4 전략: 청년 고소비지역")
        # TODO: 해당 클러스터 시각화 함수 호출