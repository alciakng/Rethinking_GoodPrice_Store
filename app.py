import streamlit as st

from ui.main_board import sidebar_board
from ui.main_board import main_board

# ---------------------------
# 기본 앱 설정
# ---------------------------
st.set_page_config(
    page_title="하향식 착한가격업소 선정 및 활성화 방안",
    layout="wide"
)

# ---------------------------
# 메인보드 호출
# ---------------------------
main_board()