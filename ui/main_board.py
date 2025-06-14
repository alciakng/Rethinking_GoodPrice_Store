import streamlit as st
from streamlit_option_menu import option_menu

from ui.analysis_board import overview_board, strategy_board


# ---------------------------
# 사이드바 함수 
# ---------------------------
def sidebar_board():

    with st.sidebar:
        selected = option_menu(
            menu_title='항목선택',
            options=[
                "분석개요",
                "청년 밀집·저소비 지역 전략",
                "중장년 밀집·저소비 지역 전략",
                "청년 밀집·고소비 지역 전략",
                "최대소비 지역 전략"
            ],
            icons=["none"] * 5,
            menu_icon=["none"],
            default_index=0,
            styles={
                "nav-link": {"font-size": "13px", "text-align": "left", "font_weight":"bold"},
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
        <h1 style='font-weight: 700;'>🚀 물가안정을 위한 <span style='color:#4CAF50;'>하향식 착한가격업소 정책 설정</span> 및 <span style='color:#4CAF50;'>활성화 방안</span> 제안</h1>

        <p style='line-height: 2em; font-weight: 500;'>
        본 프로젝트는 <strong>행정안전부·지자체 주도의 착한가격업소 정책의 효율적인 운영</strong>을 목적으로,
        <strong>지역 상권의 특성</strong>을 반영하여 착한가격업소의 <strong>비중 및 증가 추세를 진단</strong>합니다.<br>
        <strong>진단을 통해 물가안정을 위한 정책의 목적과 상이하게, 물가가 다른지역 보다 낮고 상권이 축소된 지역에서 </strong>
        <strong>착한가격업소수의 비중이 높다는 점을 검증함으로써 정책개선의 필요성을 제기합니다.</strong><br>            
        <strong>비지도 머신러닝 기반의 군집분석(클러스터링)</strong>을 통해 상권을 구분함으로써,
        <strong>군집별 특성에 따른 지자체 주도의 하향식(Top-Down) 정책방식과 활성화 전략</strong>을 제안합니다.
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
        왼쪽 사이드바에서 클러스터링 전략을 선택하여 <strong>상권별 분석 결과</strong>와 <strong>정책 제언</strong>을 확인하실 수 있습니다.
        </p>
        """, unsafe_allow_html=True)

        # overview board()
        overview_board()

    elif selected.startswith("최대소비"):
        st.header("최대소비지역 착한가격업소 선정 및 지원전략")
        strategy_board(0)

    elif selected.startswith("청년 밀집·저소비"):
        st.header("청년 밀집·저소비지역 착한가격업소 선정 및 지원전략")
        strategy_board(1)


    elif selected.startswith("중장년 밀집·저소비"):
        st.header("중장년 밀집·저소비지역 착한가격업소 선정 및 지원전략")
        strategy_board(2)

    elif selected.startswith("청년 밀집·고소비"):
        st.header("청년 밀집·고소비 지역 착한가격업소 선정 및 지원전략")
        strategy_board(3)