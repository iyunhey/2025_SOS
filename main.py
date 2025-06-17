import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import os
import chardet # Make sure you have chardet installed: pip install chardet

st.set_page_config(page_title="응급의료 이송 및 응급실 분석", layout="wide")

# 자동 인코딩 감지 및 안전한 CSV 로드
def safe_read_csv(path):
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    
    try:
        # Read a small chunk to detect encoding
        with open(path, 'rb') as f:
            raw_data = f.read(100000) # Read up to 100KB for detection
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            
            # Fallback to common Korean encodings if confidence is low or detection fails
            if detected_encoding is None or result['confidence'] < 0.8:
                possible_encodings = ['utf-8', 'euc-kr', 'cp949', 'utf-8-sig']
            else:
                possible_encodings = [detected_encoding, 'utf-8', 'euc-kr', 'cp949', 'utf-8-sig']
            
            df = None
            for enc in possible_encodings:
                try:
                    df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
                    st.info(f"'{path}' 파일을 '{enc}' 인코딩으로 성공적으로 로드했습니다.")
                    return df
                except UnicodeDecodeError:
                    continue # Try next encoding
                except Exception as e:
                    st.error(f"'{path}' 파일을 여는 중 오류 발생: {e} (인코딩: {enc})")
                    return pd.DataFrame() # Return empty if other error
            
            st.error(f"'{path}' 파일을 지원되는 어떤 인코딩으로도 로드할 수 없습니다.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"{path} 파일을 여는 중 예기치 않은 오류 발생: {e}")
        return pd.DataFrame()


@st.cache_data
def load_emergency_transport(path):
    return safe_read_csv(path)

@st.cache_data
def load_monthly_er_usage(path):
    return safe_read_csv(path)

@st.cache_data
def load_time_er_usage(path):
    return safe_read_csv(path)

# --- 파일 경로 정의 ---
# 'data' 폴더가 project-root 안에 있고, 그 안에 CSV 파일들이 있다고 가정합니다.
path_01 = "data/정보_01_행정안전부_응급환자이송업(공공데이터포털).csv"
path_02 = "data/정보_02_월별+응급실+이용(시도별).csv"
path_03 = "data/정보_03_내원시간별+응급실+이용(시도별).csv"

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
    st.warning("이송 데이터가 비어있습니다. 파일 경로와 내용을 확인해주세요.")

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
    st.warning("월별 이용 데이터가 비어있거나 필요한 컬럼이 누락되었습니다. 파일 경로와 내용을 확인해주세요.")

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
    st.warning("시간대별 이용 데이터가 비어있거나 필요한 컬럼이 누락되었습니다. 파일 경로와 내용을 확인해주세요.")

# 4️⃣ 스택/큐 시뮬레이션
st.subheader("🧠 응급 대기 순서 시뮬레이션 (스택/큐 모델)")
mode = st.radio("대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])
patient_names = st.text_input("환자 이름 (쉼표로 구분)", "환자1,환자2,환자3")

names = [name.strip() for name in patient_names.split(',') if name.strip()]
if names:
    if mode == '큐 (선입선출)':
        q = deque(names)
        st.write("🚶‍♀️ 큐 처리 순서:")
        st.write(list(q))
    else:
        stack = list(reversed(names))
        st.write("🚶‍♂️ 스택 처리 순서:")
        st.write(stack)
else:
    st.info("환자 이름을 입력하여 시뮬레이션을 시작해주세요.")

st.markdown("---")
st.caption("ⓒ 2025 긴급의료연구 프로젝트 by Streamlit")
