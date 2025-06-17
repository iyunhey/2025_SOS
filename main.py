import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import os
import chardet

st.set_page_config(page_title="응급의료 이송 및 응급실 분석", layout="wide")

# 자동 인코딩 감지 및 안전한 CSV 로드
def safe_read_csv(path):
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
    try:
        df = pd.read_csv(path, encoding=result['encoding'], on_bad_lines='skip')
    except Exception as e:
        st.error(f"{path} 파일을 여는 중 오류 발생: {e}")
        df = pd.DataFrame()
    return df

@st.cache_data
def load_emergency_transport(path):
    return safe_read_csv(path)

@st.cache_data
def load_monthly_er_usage(path):
    return safe_read_csv(path)

@st.cache_data
def load_time_er_usage(path):
    return safe_read_csv(path)

# 📁 파일 경로 설정
path_01 = "info_01.csv"
path_02 = "info_02.csv"
path_03 = "info_03.csv"

# 📦 데이터 로드
transport_df = load_emergency_transport(path_01)
monthly_df = load_monthly_er_usage(path_02)
time_df = load_time_er_usage(path_03)

# ---------------- UI 구성 ----------------

st.title("🚑 응급의료 이송 및 응급실 이용 분석")
st.markdown("#### 📊 응급환자 이송 현황, 월별 및 시간대별 응급실 이용 패턴을 분석하고 시각화합니다.")

# 1️⃣ 응급환자 이송 분석
st.subheader("1️⃣ 응급환자 이송 현황 분석")
if not transport_df.empty:
    st.dataframe(transport_df.head())
    if st.checkbox("📌 이송 데이터 요약 통계 보기"):
        st.write(transport_df.describe(include='all'))
    if '시도명' in transport_df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        transport_df.groupby('시도명').size().sort_values().plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_xlabel("건수")
        ax1.set_ylabel("시도")
        st.pyplot(fig1)
else:
    st.warning("이송 데이터가 비어있습니다.")

# 2️⃣ 월별 응급실 이용
st.subheader("2️⃣ 월별 응급실 이용 현황")
if not monthly_df.empty and '월' in monthly_df.columns and '시도별' in monthly_df.columns:
    monthly_df['월'] = monthly_df['월'].astype(str)
    selected_region = st.selectbox("지역 선택", monthly_df['시도별'].unique())
    region_data = monthly_df[monthly_df['시도별'] == selected_region]

    fig2, ax2 = plt.subplots()
    sns.lineplot(x='월', y='합계', data=region_data, marker='o', ax=ax2)
    ax2.set_title(f"{selected_region}의 월별 응급실 이용 현황")
    ax2.set_ylabel("이용 건수")
    st.pyplot(fig2)
else:
    st.warning("월별 이용 데이터가 비어있거나 컬럼이 누락되었습니다.")

# 3️⃣ 시간대별 응급실 이용
st.subheader("3️⃣ 시간대별 응급실 이용 현황")
if not time_df.empty and '내원시간대' in time_df.columns and '시도별' in time_df.columns:
    selected_region_time = st.selectbox("시간대별 지역 선택", time_df['시도별'].unique())
    region_time = time_df[time_df['시도별'] == selected_region_time]

    fig3, ax3 = plt.subplots()
    sns.barplot(x='내원시간대', y='합계', data=region_time, palette='coolwarm', ax=ax3)
    ax3.set_title(f"{selected_region_time}의 시간대별 응급실 이용")
    ax3.set_ylabel("이용 건수")
    st.pyplot(fig3)
else:
    st.warning("시간대별 이용 데이터가 비어있거나 컬럼이 누락되었습니다.")

# 4️⃣ 스택/큐 시뮬레이션
st.subheader("🧠 응급 대기 순서 시뮬레이션 (스택/큐 모델)")
mode = st.radio("대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])
patient_names = st.text_input("환자 이름 (쉼표로 구분)", "환자1,환자2,환자3")

names = [name.strip() for name in patient_names.split(',') if name.strip()]
if names:
    if mode == '큐 (선입선출)':
        q = deque(names)
        st.write("처리 순서:", list(q))
    else:
        stack = list(reversed(names))
        st.write("처리 순서:", stack)
