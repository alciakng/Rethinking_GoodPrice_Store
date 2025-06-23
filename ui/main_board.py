import streamlit as st
from streamlit_option_menu import option_menu

from ui.analysis_board import overview_board, strategy_board


# ---------------------------
# 사이드바 함수 
# ---------------------------
def sidebar_board():

    with st.sidebar:
        selected = option_menu(
            menu_title='메뉴선택',
            options=[
                "분석개요",
                "전략제안"
            ],
            icons=["none"] * 5,
            menu_icon=["none"],
            default_index=0,
            styles={
                "nav-link": {"font-size": "20px", "text-align": "left", "font_weight":"bold"},
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
        # overview board()
        overview_board()


    elif selected.startswith("전략제안"):
        strategy_board()

