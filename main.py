import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque

st.set_page_config(page_title="응급의료 이송 및 분석 대시보드", layout="wide")
st.title("🚑 응급환자 이송 및 응급실 이용 분석")

# -------------------------------
# 파일 경로
# -------------------------------
transport_path = "data/정보_01_행정안전부_응급환자이송업(공공데이터포털).csv"
time_json_path = "data/정보_SOS_03.json"
month_json_path = "data/정보_SOS_02.json"

# -------------------------------
# 데이터 로딩 함수
# -------------------------------
@st.cache_data
def load_transport_data(path):
    return pd.read_csv(path, encoding='cp949')

@st.cache_data
def load_time_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw[4:]  # 데이터는 5번째 행부터 시작
    time_cols = {
        'col5': '00-03시', 'col6': '03-06시', 'col7': '06-09시', 'col8': '09-12시',
        'col9': '12-15시', 'col10': '15-18시', 'col11': '18-21시', 'col12': '21-24시'
    }
    rows = []
    for row in records:
        region = row['col3']
        if region == "전체" or not region:
            continue
        values = [int(row.get(c, "0").replace(",", "")) for c in time_cols.keys()]
        rows.append([region] + values)
    return pd.DataFrame(rows, columns=['시도'] + list(time_cols.values()))

@st.cache_data
def load_month_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw[4:]
    month_cols = {
        'col7': '1월', 'col8': '2월', 'col9': '3월', 'col10': '4월',
        'col11': '5월', 'col12': '6월', 'col13': '7월', 'col14': '8월',
        'col15': '9월', 'col16': '10월', 'col17': '11월', 'col18': '12월'
    }
    rows = []
    for row in records:
        region = row['col3']
        if region == "전체" or not region:
            continue
        values = [int(row.get(c, "0").replace(",", "")) for c in month_cols.keys()]
        rows.append([region] + values)
    return pd.DataFrame(rows, columns=['시도'] + list(month_cols.values()))

# -------------------------------
# 데이터 로드
# -------------------------------
transport_df = load_transport_data(transport_path)
time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)

# -------------------------------
# 사이드바 사용자 상호작용
# -------------------------------
st.sidebar.title("사용자 설정")
region = st.sidebar.selectbox("지역 선택", sorted(list(set(time_df['시도']) & set(month_df['시도']))))

# -------------------------------
# 1️⃣ 응급환자 이송 현황
# -------------------------------
st.subheader("1️⃣ 응급환자 이송 현황 분석")
if not transport_df.empty:
    st.dataframe(transport_df.head())
    if '시도명' in transport_df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        transport_df.groupby('시도명').size().sort_values().plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title("시도별 이송 건수")
        ax1.set_xlabel("건수")
        st.pyplot(fig1)

# -------------------------------
# 2️⃣ 시간대별 분석
# -------------------------------
st.subheader("2️⃣ 시간대별 응급실 이용 현황 (2023)")
time_row = time_df[time_df['시도'] == region].iloc[0, 1:]
fig2, ax2 = plt.subplots()
time_row.plot(kind='bar', color='deepskyblue', ax=ax2)
ax2.set_ylabel("이용 건수")
ax2.set_xlabel("시간대")
ax2.set_title(f"{region} 시간대별 응급실 이용")
st.pyplot(fig2)

# -------------------------------
# 3️⃣ 월별 분석
# -------------------------------
st.subheader("3️⃣ 월별 응급실 이용 현황 (2023)")
month_row = month_df[month_df['시도'] == region].iloc[0, 1:]
fig3, ax3 = plt.subplots()
month_row.plot(kind='line', marker='o', color='seagreen', ax=ax3)
ax3.set_ylabel("이용 건수")
ax3.set_xlabel("월")
ax3.set_title(f"{region} 월별 응급실 이용")
st.pyplot(fig3)

# -------------------------------
# 4️⃣ 우선순위 큐 시뮬레이션
# -------------------------------
st.subheader("4️⃣ 응급 대기 시뮬레이션 (스택/큐 모델)")
mode = st.radio("대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])
patient_input = st.text_input("환자 이름 (쉼표로 구분)", "홍길동,김영희,이철수")
patients = [p.strip() for p in patient_input.split(',') if p.strip()]

if patients:
    st.write("**진료 순서:**")
    if mode == '큐 (선입선출)':
        queue = deque(patients)
        st.write(list(queue))
    else:
        st.write(list(reversed(patients)))

st.markdown("---")
st.caption("ⓒ 2025 스마트 응급의료 데이터 분석 프로젝트 - SDG 3.8 보건서비스 접근성 개선")
