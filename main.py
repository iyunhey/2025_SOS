import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet

st.set_page_config(page_title="응급의료 이송 및 분석 대시보드", layout="wide")
st.title("🚑 응급환자 이송 및 응급실 이용 분석")

# ... (기존 load_transport_data, load_time_data, load_month_data 함수 정의는 동일) ...

# -------------------------------
# 데이터 로드
# -------------------------------
transport_df = load_transport_data(transport_path)

# --- ✨ transport_df 전처리: '시도명' 컬럼 생성 ✨ ---
if not transport_df.empty and '소재지전체주소' in transport_df.columns:
    # '소재지전체주소' 컬럼에서 첫 번째 공백까지의 문자열을 추출하여 '시도명'으로 사용
    # 예: '서울특별시 강남구...' -> '서울특별시'
    # 만약 '도로명전체주소'가 더 적합하다면 '도로명전체주소'를 사용하세요.
    transport_df['시도명'] = transport_df['소재지전체주소'].apply(
        lambda x: str(x).split(' ')[0] if pd.notna(x) and ' ' in str(x) else None
    )
    # 데이터에 따라 "세종특별자치시"처럼 공백 없이 시도명인 경우도 처리
    transport_df['시도명'] = transport_df['시도명'].apply(
        lambda x: '세종특별자치시' if x and '세종' in x else x
    )
    # 추출된 시도명이 비어있거나 불완전할 경우를 대비하여 추가 처리 필요할 수 있음
    
    st.info("'소재지전체주소' 컬럼을 기반으로 '시도명' 컬럼을 생성했습니다.")
elif not transport_df.empty: # 소재지전체주소 컬럼이 없는 경우
    st.warning("'transport_df'에 '소재지전체주소' 컬럼이 없습니다. '시도명' 생성을 건너뜁니다.")
# --- ✨ 전처리 끝 ✨ ---

time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)

# ... (나머지 사이드바, UI 구성 코드는 동일) ...

# -------------------------------
# 1️⃣ 응급환자 이송 현황
# -------------------------------
st.subheader("1️⃣ 응급환자 이송 현황 분석")
if not transport_df.empty:
    st.dataframe(transport_df.head())
    if st.checkbox("📌 이송 데이터 요약 통계 보기"):
        st.write(transport_df.describe(include='all'))
    
    # '시도명' 컬럼이 이제 전처리 과정에서 생성되었으므로, 이 조건문은 그대로 유지
    if '시도명' in transport_df.columns and transport_df['시도명'].notna().any(): # 시도명 컬럼이 있고, 유효한 값이 있을 때만 그래프 그림
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        # '시도명' 컬럼으로 그룹화
        transport_df.groupby('시도명').size().sort_values().plot(kind='barh', ax=ax1, color='skyblue') 
        ax1.set_title("시도별 이송 건수")
        ax1.set_xlabel("건수")
        ax1.set_ylabel("시도")
        st.pyplot(fig1)
    else:
        st.warning("이송 데이터에 '시도명' 컬럼이 없거나 유효한 시도명 값이 없습니다. 데이터 내용을 확인해주세요.")
else:
    st.warning("이송 데이터가 비어있습니다. 파일 경로와 내용을 확인해주세요.")

# ... (나머지 UI 코드 동일) ...
