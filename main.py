import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# 페이지 설정
st.set_page_config(page_title="응급의료 이송 및 응급실 분석", layout="wide")

# 데이터 로드 함수
@st.cache_data
def load_emergency_transport(path):
    return pd.read_csv(path, encoding='cp949')  # 'utf-8-sig'도 가능

@st.cache_data
def load_monthly_er_usage(path):
    return pd.read_csv(path, encoding='cp949')

@st.cache_data
def load_time_er_usage(path):
    return pd.read_csv(path, encoding='cp949')

# 제목
st.title("🚑 응급의료 이송 및 응급실 이용 분석")
st.markdown("#### 📊 응급환자 이송 현황, 월별 및 시간대별 응급실 이용 패턴을 분석하고 시각화합니다.")

# 데이터 경로
transport_df = load_emergency_transport('info_01.csv')
monthly_df = load_monthly_er_usage('info_02.csv')
time_df = load_time_er_usage('info_03.csv')

# -------------------------------
# 1️⃣ 응급환자 이송 현황 분석
# -------------------------------
st.subheader("1️⃣ 응급환자 이송 현황 분석")
st.dataframe(transport_df.head())

if st.checkbox("📌 이송 데이터 요약 통계 보기"):
    st.write(transport_df.describe(include='all'))

if '시도명' in transport_df.columns:
    st.markdown("**시도별 응급환자 이송 건수**")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    transport_df.groupby('시도명').size().sort_values().plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_xlabel("건수")
    ax1.set_ylabel("시도")
    st.pyplot(fig1)

# -------------------------------
# 2️⃣ 월별 응급실 이용 분석
# -------------------------------
st.subheader("2️⃣ 월별 응급실 이용 현황")

if '월' in monthly_df.columns and '시도별' in monthly_df.columns:
    monthly_df['월'] = monthly_df['월'].astype(str)
    selected_region = st.selectbox("지역 선택", monthly_df['시도별'].unique())
    region_data = monthly_df[monthly_df['시도별'] == selected_region]

    fig2, ax2 = plt.subplots()
    sns.lineplot(x='월', y='합계', data=region_data, marker='o', ax=ax2)
    ax2.set_title(f"{selected_region}의 월별 응급실 이용 현황")
    ax2.set_ylabel("이용 건수")
    st.pyplot(fig2)

# -------------------------------
# 3️⃣ 시간대별 응급실 이용 분석
# -------------------------------
st.subheader("3️⃣ 시간대별 응급실 이용 현황")

if '내원시간대' in time_df.columns and '시도별' in time_df.columns:
    selected_region_time = st.selectbox("시간대별 지역 선택", time_df['시도별'].unique())
    region_time = time_df[time_df['시도별'] == selected_region_time]

    fig3, ax3 = plt.subplots()
    sns.barplot(x='내원시간대', y='합계', data=region_time, palette='coolwarm', ax=ax3)
    ax3.set_title(f"{selected_region_time}의 시간대별 응급실 이용")
    ax3.set_ylabel("이용 건수")
    st.pyplot(fig3)

# -------------------------------
# 4️⃣ 스택/큐 모델 시뮬레이션
# -------------------------------
st.subheader("🧠 응급 대기 순서 시뮬레이션 (스택/큐 모델)")

mode = st.radio("대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])
patient_names = st.text_input("환자 이름 (쉼표로 구분)", "환자1,환자2,환자3")

names = [name.strip() for name in patient_names.split(',') if name.strip()]

if names:
    if mode == '큐 (선입선출)':
        queue = deque(names)
        st.markdown("**진료 순서 (큐)**")
        for i, name in enumerate(queue, 1):
            st.write(f"{i}번째: {name}")
    else:
        stack = list(names)
        st.markdown("**진료 순서 (스택)**")
        for i, name in enumerate(reversed(stack), 1):
            st.write(f"{i}번째: {name}")
