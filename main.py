import streamlit as st
import pandas as pd
import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import os

# --- 경로 설정 ---
DATA_DIR = '/mnt/data'

# --- 전처리 함수 정의 ---
@st.cache_data
def load_emergency_transport(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='cp949')
    df = df.dropna(subset=['출발지위도', '출발지경도', '도착지위도', '도착지경도'])
    df[['출발지위도','출발지경도','도착지위도','도착지경도']] = df[['출발지위도','출발지경도','도착지위도','도착지경도']].astype(float)
    return df

@st.cache_data
def load_er_usage_monthly(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='cp949')
    df['월'] = pd.to_datetime(df['월'], format='%Y-%m')
    return df

@st.cache_data
def load_er_usage_hourly(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding='cp949')
    df['시간대'] = df['시간대'].astype(str)
    return df

# 우선순위 큐 클래스
class EmergencyQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, severity, patient_id):
        heapq.heappush(self.heap, (-severity, self.counter, patient_id))
        self.counter += 1

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[2]
        return None

    def to_list(self):
        return [item[2] for item in sorted(self.heap, reverse=True)]

# 그래프 모델 생성
@st.cache_data
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = (row['출발지위도'], row['출발지경도'])
        dst = (row['도착지위도'], row['도착지경도'])
        dist = np.hypot(dst[0]-src[0], dst[1]-src[1])
        G.add_edge(src, dst, weight=dist)
    return G

# --- 앱 본문 ---
st.set_page_config(page_title="응급의료 최적화 대시보드", layout="wide")
st.title("응급의료 서비스 최적화")

# 데이터 로드
transport_df = load_emergency_transport('정보_01_행정안전부_응급환자이송업(공공데이터포털).csv')
er_monthly = load_er_usage_monthly('정보_02_월별+응급실+이용(시도별).csv')
er_hourly = load_er_usage_hourly('정보_03_내원시간별+응급실+이용(시도별).csv')

# 그래프 빌드 (데이터 유효 시)
if not transport_df.empty:
    G = build_graph(transport_df)
else:
    G = nx.DiGraph()

# 사이드바: 사용자·모드 선택
user = st.sidebar.selectbox("사용자 유형", ['의료진', '구급대원', '병원 관리자'])
mode = st.sidebar.radio("모드 선택", ['환자 관리', '경로 최적화', '통계 조회'])

# 전역 우선순위 큐
if 'eq' not in st.session_state:
    st.session_state.eq = EmergencyQueue()

# 1) 환자 관리 모드
if mode == '환자 관리':
    st.header("환자 우선순위 큐 관리")
    col1, col2 = st.columns(2)

    with col1:
        pid = st.text_input("환자 ID 입력")
        severity = st.slider("응급도 점수", 1, 10, 5)
        if st.button("추가"): st.session_state.eq.push(severity, pid)

    with col2:
        if st.button("처리"): processed = st.session_state.eq.pop(); st.write(f"처리된 환자: {processed}")

    st.subheader("현재 큐 현황")
    queue_list = st.session_state.eq.to_list()
    st.write(queue_list)
    fig, ax = plt.subplots()
    ax.barh(range(len(queue_list)), list(range(len(queue_list))), tick_label=queue_list)
    ax.set_xlabel("대기 순서")
    st.pyplot(fig)

# 2) 경로 최적화 모드
elif mode == '경로 최적화':
    st.header("이송 경로 최적화")
    if G.number_of_nodes() == 0:
        st.warning("이송 데이터가 없어 경로 최적화를 수행할 수 없습니다.")
    else:
        coords = list(G.nodes)
        src = st.selectbox("출발 좌표", coords)
        dst = st.selectbox("도착 좌표", coords)
        if st.button("계산"):            
            path = nx.dijkstra_path(G, src, dst)
            st.write(path)
            fig, ax = plt.subplots()
            xs, ys = zip(*path)
            ax.plot(ys, xs, marker='o')
            st.pyplot(fig)

# 3) 통계 조회 모드
else:
    st.header("응급실 이용 통계")
    if not er_monthly.empty:
        st.subheader("월별 이용")
        st.line_chart(er_monthly.set_index('월')['이용건수'])
    if not er_hourly.empty:
        st.subheader("시간대별 이용")
        st.bar_chart(er_hourly.set_index('시간대')['이용건수'])

st.markdown("---")
st.write("개발팀 Streamlit 앱")
