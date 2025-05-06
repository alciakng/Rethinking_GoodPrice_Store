import streamlit as st
from streamlit_option_menu import option_menu

from ui.analysis_board import trend_board

# ==========================
# 좌측 메뉴 생성 
# ==========================
def left_menu_section():
    
    # 메뉴
    with st.sidebar:
        selected = option_menu(
            menu_title='Menu',  # 메뉴 제목
            options=["지역 트렌드 분석", "착한구역/업체 선정"],
            icons=["bar-chart", "calculator"],
            menu_icon='folder',  # 상단 아이콘
            default_index=0,
            styles={
                "container": {"padding": "10px"},
                "icon": {"color": "green", "font-size": "18px"},
                "nav-link-selected": {"background-color": "#4CAF50", "font-weight": "bold", "color": "white"}
            }
    )

    # 콘텐츠 렌더링
    if selected == "지역 트렌드 분석":
        st.subheader("📊 지역 트렌드 분석")
        trend_board()
    
    elif selected == "착한구역/업체 선정":
        st.subheader("🧮 착한구역/업체 선정")


# ==========================
# trend analysis 조건
# ==========================
def trend_analysis_condition():
    st.title("트렌드 분석조건 선택")

    col1, col2 = st.columns(2)

    # 공통 조회 조건 (위에 삽입)
    st.markdown("###  조회기간 선택")
