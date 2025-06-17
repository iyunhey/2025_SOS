import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

st.set_page_config(page_title="응급의료 이송 및 응급실 분석", layout="wide")

@st.cache_data
def load_emergency_transport(path):
    # 파일 경로를 상대 경로로 변경했습니다.
    # 만약 파일이 'data' 폴더 안에 있다면 'data/info_01.csv' 등으로 수정해야 합니다.
    df = pd.read_csv(path, encoding='cp949')
    return df

@st.cache_data
def load_monthly_er_usage(path):
    # 파일 경로를 상대 경로로 변경했습니다.
    df = pd.read_csv(path, encoding='cp949')
    return df

@st.cache_data
def load_time_er_usage(path):
    # 파일 경로를 상대 경로로 변경했습니다.
    df = pd.read_csv(path, encoding='cp949')
    return df

st.title("🚑 응급의료 이송 및 응급실 이용 분석")
st.markdown("#### 📊 응급환자 이송 현황, 월별 및 시간대별 응급실 이용 패턴을 분석하고 시각화합니다.")

# 데이터 로드
# 파일이 main.py와 같은 디렉토리에 있을 경우
transport_df = load_emergency_transport('info_01.csv')
monthly_df = load_monthly_er_usage('info_02.csv')
time_df = load_time_er_usage('info_03.csv')

# 만약 파일이 'data'라는 하위 폴더에 있다면 아래처럼 수정하세요:
# transport_df = load_emergency_transport('data/info_01.csv')
# monthly_df = load_monthly_er_usage('data/info_02.csv')
# time_df = load_time_er_usage('data/info_03.csv')


st.subheader("1️⃣ 응급환자 이송 현황 분석")
st.dataframe(transport_df.head())

if st.checkbox("📌 이송 데이터 요약 통계 보기"):
    st.write(transport_df.describe(include='all'))

# 시도별 이송 환자 수 시각화
if '시도명' in transport_df.columns:
    st.markdown("**시도별 응급환자 이송 건수**")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    transport_df.groupby('시도명').size().sort_values().plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_xlabel("건수")
    ax1.set_ylabel("시도")
    st.pyplot(fig1)

st.subheader("2️⃣ 월별 응급실 이용 현황")

if '월' in monthly_df.columns:
    monthly_df['월'] = monthly_df['월'].astype(str)
    selected_region = st.selectbox("지역 선택", monthly_df['시도별'].unique())
    region_data = monthly_df[monthly_df['시도별'] == selected_region]
    
    fig2, ax2 = plt.subplots()
    sns.lineplot(x='월', y='합계', data=region_data, marker='o', ax=ax2)
    ax2.set_title(f"{selected_region}의 월별 응급실 이용 현황")
    ax2.set_ylabel("이용 건수")
    st.pyplot(fig2)

st.subheader("3️⃣ 시간대별 응급실 이용 현황")

if '내원시간대' in time_df.columns:
    selected_region_time = st.selectbox("시간대별 지역 선택", time_df['시도별'].unique())
    region_time = time_df[time_df['시도별'] == selected_region_time]
    
    fig3, ax3 = plt.subplots()
    sns.barplot(x='내원시간대', y='합계', data=region_time, palette='coolwarm', ax=ax3)
    ax3.set_title(f"{selected_region_time}의 시간대별 응급실 이용")
    ax3.set_ylabel("이용 건수")
    st.pyplot(fig3)

# 스택 및 큐 시각화
st.subheader("🧠 응급 대기 순서 시뮬레이션 (스택/큐 모델)")

mode = st.radio("대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])
patient_names = st.text_input("환자 이름 (쉼표로 구분)", "환자1,환자2,환자3")

names = [name.strip() for name in patient_names.split(',') if name.strip()]

if st.button("대기 순서 시뮬레이션"):
    if mode == '큐 (선입선출)':
        queue = deque(names)
        st.write("🚶‍♀️ 큐 순서:")
        st.write(list(queue))
    else:
        stack = list(names)
        st.write("🚶‍♂️ 스택 순서:")
        st.write(list(reversed(stack)))

st.markdown("---")
st.caption("ⓒ 2025 긴급의료연구 프로젝트 by Streamlit")
