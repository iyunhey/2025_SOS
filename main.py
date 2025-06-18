import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet

# Matplotlib 한글 폰트 설정
# Streamlit Cloud 환경에서는 폰트 설치가 필요할 수 있습니다.
# 예를 들어, .streamlit/config.toml 파일에 다음과 같이 추가 (Streamlit Cloud에서)
# [theme]
# fontFamily = "Malgun Gothic"
# (로컬 환경의 경우)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 사용자
# plt.rcParams['font.family'] = 'AppleGothic' # macOS 사용자
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

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

# CSV 파일을 안전하게 로드하는 함수
@st.cache_data
def load_transport_data(path):
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    
    try:
        # Notepad++에서 EUC-KR로 확인되었으나, 'euc-kr' decode 오류가 발생했으므로
        # 'cp949'를 가장 먼저 시도하고 그 다음 'euc-kr'을 시도합니다.
        # 'cp949'는 'euc-kr'의 확장판으로 더 많은 문자를 포함합니다.
        # 이후 utf-8 및 utf-8-sig도 시도합니다.
        possible_encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig'] 
        possible_seps = [',', ';', '\t', '|']

        df = None
        for enc in possible_encodings:
            for sep in possible_seps:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines='skip', engine='python')
                    
                    if not df.empty and len(df.columns) > 1:
                        st.info(f"'{path}' 파일을 '{enc}' 인코딩, 구분자 '{sep}'로 성공적으로 로드했습니다.")
                        return df
                    else:
                        continue # 빈 DataFrame이거나 컬럼이 하나면 잘못 로드된 것으로 간주
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    # 인코딩 오류 또는 파싱 오류 발생 시 다음 조합 시도
                    # st.warning(f"'{path}' 로드 실패 (인코딩: {enc}, 구분자: {sep}): {e}") # 디버깅용
                    continue
                except Exception as e:
                    # 예상치 못한 다른 오류 발생 시
                    st.error(f"'{path}' 파일을 여는 중 예상치 못한 오류 발생 (인코딩: {enc}, 구분자: {sep}): {e}")
                    continue
        
        st.error(f"'{path}' 파일을 지원되는 어떤 인코딩/구분자로도 로드할 수 없습니다. 파일 내용을 직접 확인해주세요.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"'{path}' 파일을 로드하는 중 최상위 오류 발생: {e}")
        return pd.DataFrame()


@st.cache_data
def load_time_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw[4:]
        time_cols = {
            'col5': '00-03시', 'col6': '03-06시', 'col7': '06-09시', 'col8': '09-12시',
            'col9': '12-15시', 'col10': '15-18시', 'col11': '18-21시', 'col12': '21-24시'
        }
        rows = []
        for row in records:
            region = row.get('col3')
            if region == "전체" or not region:
                continue
            values = [int(row.get(c, "0").replace(",", "")) for c in time_cols.keys()]
            rows.append([region] + values)
        df = pd.DataFrame(rows, columns=['시도'] + list(time_cols.values()))
        st.info(f"'{path}' JSON 파일을 성공적으로 로드했습니다.")
        return df
    except FileNotFoundError:
        st.error(f"JSON 파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        st.error(f"'{path}' JSON 파일 디코딩 오류: {e}. 파일 내용이 올바른 JSON 형식인지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"'{path}' JSON 파일을 로드하는 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data
def load_month_data(path):
    try:
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
            region = row.get('col3')
            if region == "전체" or not region:
                continue
            values = [int(row.get(c, "0").replace(",", "")) for c in month_cols.keys()]
            rows.append([region] + values)
        df = pd.DataFrame(rows, columns=['시도'] + list(month_cols.values()))
        st.info(f"'{path}' JSON 파일을 성공적으로 로드했습니다.")
        return df
    except FileNotFoundError:
        st.error(f"JSON 파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        st.error(f"'{path}' JSON 파일 디코딩 오류: {e}. 파일 내용이 올바른 JSON 형식인지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"'{path}' JSON 파일을 로드하는 중 오류 발생: {e}")
        return pd.DataFrame()

# -------------------------------
# 데이터 로드 및 전처리
transport_df = load_transport_data(transport_path)

# --- ✨ transport_df 전처리: '시도명' 컬럼 생성 및 보정 ✨ ---
if not transport_df.empty and '소재지전체주소' in transport_df.columns:
    def extract_sido(address):
        if pd.isna(address):
            return None
        
        addr_str = str(address).strip() # 앞뒤 공백 제거
        if not addr_str: # 빈 문자열인 경우
            return None

        parts = addr_str.split(' ')
        if not parts: # 공백으로 나눴을 때 빈 리스트인 경우
            return None

        first_part = parts[0]

        # 세종특별자치시와 같이 단일 단어이지만 긴 경우를 먼저 처리
        if '세종' in first_part:
            return '세종특별자치시'
        
        # 일반적인 시/도명 패턴 (2~4글자)
        if len(first_part) <= 4:
            # "서울", "경기", "인천" 등
            # 여기에 시도 목록을 명시적으로 넣어 더 정확하게 필터링할 수도 있습니다.
            korean_sido_list = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
                                 "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도",
                                 "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도",
                                 "제주특별자치도"]
            
            # 주소의 첫 부분이 실제 시도명 목록에 포함되는지 확인
            for sido in korean_sido_list:
                if first_part in sido: # 예: '서울' in '서울특별시'
                    return sido # 정확한 시도명을 반환

        # 특별시, 광역시, 자치시/도 포함하는 경우 처리
        # '서울특별시', '부산광역시', '강원특별자치도', '제주특별자치도' 등
        for part in parts:
            if '특별시' in part or '광역시' in part or '자치시' in part or '자치도' in part:
                # '강원특별자치도'처럼 두 단어일 경우를 위해 조정
                if '강원' in part or '전라' in part or '충청' in part or '경상' in part or '경기' in part:
                    return f"{parts[0]}{part}" # '강원' + '특별자치도'
                return part # '서울특별시', '부산광역시' 등

        return None # 어떤 조건에도 해당하지 않으면 None 반환


    transport_df['시도명'] = transport_df['소재지전체주소'].apply(extract_sido)

    # 시도명 컬럼에 유효하지 않은 (None) 값이 남아있을 경우 제거
    # 혹은 '기타' 등으로 채울 수도 있습니다: transport_df['시도명'].fillna('기타', inplace=True)
    transport_df.dropna(subset=['시도명'], inplace=True)
    
    st.info("'소재지전체주소' 컬럼을 기반으로 '시도명' 컬럼을 생성하고 보정했습니다.")
elif not transport_df.empty: # 소재지전체주소 컬럼이 없는 경우
    st.warning("'transport_df'에 '소재지전체주소' 컬럼이 없습니다. '시도명' 생성을 건너킵니다.")
# --- ✨ 전처리 끝 ✨ ---
time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)


# -------------------------------
# 사이드바 사용자 상호작용
# -------------------------------
st.sidebar.title("사용자 설정")
if not time_df.empty and not month_df.empty:
    common_regions = list(set(time_df['시도']) & set(month_df['시도']))
    if common_regions:
        region = st.sidebar.selectbox("지역 선택", sorted(common_regions))
    else:
        st.sidebar.warning("시간대별 및 월별 데이터에 공통 지역이 없습니다.")
        region = None
else:
    st.sidebar.warning("시간대별 또는 월별 데이터가 로드되지 않았습니다.")
    region = None


# -------------------------------
# 1️⃣ 응급환자 이송 현황
# -------------------------------
st.subheader("1️⃣ 응급환자 이송 현황 분석")
if not transport_df.empty:
    st.dataframe(transport_df.head())
    if st.checkbox("📌 이송 데이터 요약 통계 보기"):
        st.write(transport_df.describe(include='all'))
    
    # '시도명' 컬럼이 이제 전처리 과정에서 생성되었으므로, 이 조건문은 그대로 유지
    # 시도명 컬럼이 있고, 유효한 값이 있을 때만 그래프를 그립니다.
    if '시도명' in transport_df.columns and transport_df['시도명'].notna().any(): 
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        # '시도명' 컬럼으로 그룹화
        transport_df.groupby('시도명').size().sort_values(ascending=False).plot(kind='barh', ax=ax1, color='skyblue') # 내림차순 정렬
        ax1.set_title("시도별 이송 건수")
        ax1.set_xlabel("건수")
        ax1.set_ylabel("시도")
        plt.tight_layout() # 레이아웃 자동 조정
        st.pyplot(fig1)
    else:
        st.warning("이송 데이터에 '시도명' 컬럼이 없거나 유효한 시도명 값이 없습니다. 데이터 내용을 확인해주세요.")
else:
    st.warning("이송 데이터가 비어있습니다. 파일 경로와 내용을 확인해주세요.")

# -------------------------------
# 2️⃣ 시간대별 분석
# -------------------------------
st.subheader("2️⃣ 시간대별 응급실 이용 현황 (2023)")
if not time_df.empty and region:
    # time_df에서 해당 지역의 데이터 행 선택
    time_row = time_df[time_df['시도'] == region].iloc[0, 1:]
    
    fig2, ax2 = plt.subplots()
    time_row.plot(kind='bar', color='deepskyblue', ax=ax2)
    ax2.set_ylabel("이용 건수")
    ax2.set_xlabel("시간대")
    ax2.set_title(f"{region} 시간대별 응급실 이용")
    st.pyplot(fig2)
else:
    st.warning("시간대별 데이터 로드에 문제가 있거나 지역이 선택되지 않았습니다.")

# -------------------------------
# 3️⃣ 월별 분석
# -------------------------------
st.subheader("3️⃣ 월별 응급실 이용 현황 (2023)")
if not month_df.empty and region:
    # month_df에서 해당 지역의 데이터 행 선택
    month_row = month_df[month_df['시도'] == region].iloc[0, 1:]
    
    fig3, ax3 = plt.subplots()
    month_row.plot(kind='line', marker='o', color='seagreen', ax=ax3)
    ax3.set_ylabel("이용 건수")
    ax3.set_xlabel("월")
    ax3.set_title(f"{region} 월별 응급실 이용")
    st.pyplot(fig3)
else:
    st.warning("월별 데이터 로드에 문제가 있거나 지역이 선택되지 않았습니다.")

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
