import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet

# 공간 데이터 및 그래프 처리를 위한 라이브러리
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter 

# Matplotlib 한글 폰트 설정
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

# ... (load_transport_data 함수는 동일) ...
@st.cache_data
def load_transport_data(path):
    if not os.path.exists(path):
        st.error(f"파일을 찾을 수 없습니다: {path}")
        return pd.DataFrame()
    
    try:
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
                        continue 
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    continue
                except Exception as e:
                    st.error(f"'{path}' 파일을 여는 중 예상치 못한 오류 발생 (인코딩: {enc}, 구분자: {sep}): {e}")
                    continue
        
        st.error(f"'{path}' 파일을 지원되는 어떤 인코딩/구분자로도 로드할 수 없습니다. 파일 내용을 직접 확인해주세요.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"'{path}' 파일을 로드하는 중 최상위 오류 발생: {e}")
        return pd.DataFrame()

# ... (load_time_data 함수는 동일) ...
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

# ... (load_month_data 함수는 동일) ...
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

# ... (load_road_network_from_osmnx 함수는 동일) ...
@st.cache_data
def load_road_network_from_osmnx(place_name):
    try:
        st.info(f"'{place_name}' 지역의 도로망 데이터를 OpenStreetMap에서 가져오는 중입니다. 잠시 기다려주세요...")
        G = ox.graph_from_place(place_name, network_type='drive', simplify=True, retain_all=True)
        st.success(f"'{place_name}' 도로망을 NetworkX 그래프로 변환했습니다. 노드 수: {G.number_of_nodes()}, 간선 수: {G.number_of_edges()}")
        return G

    except Exception as e:
        st.error(f"'{place_name}' 도로망 데이터를 OpenStreetMap에서 가져오고 그래프로 변환하는 중 오류 발생: {e}")
        st.warning("네트워크 연결을 확인하거나, 지역 이름이 정확한지 확인해주세요. 너무 큰 지역을 지정하면 메모리 부족이나 타임아웃이 발생할 수 있습니다.")
        return None

# ... (geocode_address 함수는 동일) ...
@st.cache_data
def geocode_address(address, user_agent="emergency_app"):
    geolocator = Nominatim(user_agent=user_agent)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) 

    try:
        if pd.isna(address) or not isinstance(address, str) or not address.strip():
            return None, None 
        
        location = geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        return None, None

# -------------------------------
# ✨ 중증도 맵핑 정의 ✨
severity_scores = {
    "경증": 1,
    "중등증": 3,
    "중증": 5,
    "응급": 10,
    "매우_응급": 20 
}

# ✨ 우선순위 큐 클래스 수정 ✨
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = [] # (우선순위 점수, 삽입 순서, 환자 정보) 튜플 저장
        self.counter = 0 # 삽입 순서 (고유성 및 선입선출 보장용)

    # ✨ queue_type 인자 추가 ✨
    def insert(self, patient_info, priority_score, queue_type="큐 (선입선출)"):
        # heapq는 기본적으로 최소 힙이므로, 높은 응급도를 높은 숫자로 정의했다면
        # 음수로 변환하여 저장하면 가장 높은 응급도(큰 양수)가 가장 작은 음수가 되어 최상위로 옴
        adjusted_score = -priority_score
        
        # ✨ 동일 중증도 내 선입선출/후입선출 로직 적용 ✨
        if queue_type == "큐 (선입선출)":
            # 삽입 순서가 낮은(먼저 들어온) 것이 우선
            entry = [adjusted_score, self.counter, patient_info]
        elif queue_type == "스택 (후입선출)":
            # 삽입 순서가 높은(나중에 들어온) 것의 음수 값이 더 작아지므로 우선
            entry = [adjusted_score, -self.counter, patient_info]
        else:
            # 기본값은 선입선출
            entry = [adjusted_score, self.counter, patient_info]

        heapq.heappush(self.heap, entry)
        self.counter += 1 # 카운터 증가 (삽입 순서 추적)
        
        # 환자 히스토리는 필요하다면 남겨두세요 (여기서는 사용하지 않음)
        # st.session_state.get('patient_queue_history', []).append(patient_info['이름']) 

    def get_highest_priority_patient(self):
        if not self.heap:
            return None  
        adjusted_score, _, patient_info = heapq.heappop(self.heap)
        original_score = -adjusted_score
        return patient_info, original_score

    def is_empty(self):
        return not bool(self.heap)

    def peek(self):
        if not self.heap:
            return None
        adjusted_score, _, patient_info = self.heap[0]
        original_score = -adjusted_score
        return patient_info, original_score
        
    def get_all_patients_sorted(self):
        # 현재 힙의 모든 항목을 복사하여 정렬된 형태로 반환 (실제 힙 변경 없음)
        # 힙은 내부적으로 순서가 보장되지만, 전체 리스트로 볼 때는 정렬이 필요
        # 튜플의 첫 번째 요소(우선순위 점수), 두 번째 요소(삽입 순서) 순으로 정렬됨
        temp_heap = sorted(self.heap) 
        sorted_patients = []
        for adjusted_score, _, patient_info in temp_heap:
            sorted_patients.append({
                '이름': patient_info.get('이름', '알 수 없음'),
                '중증도': patient_info.get('중증도', '알 수 없음'),
                '응급도 점수': -adjusted_score
            })
        return sorted_patients

# Streamlit session_state에 우선순위 큐 인스턴스 저장
if 'priority_queue' not in st.session_state:
    st.session_state.priority_queue = PriorityQueue()


# ... (데이터 로드 및 전처리 부분은 동일) ...

# ... (4️⃣ 도로망 그래프 정보 섹션까지 동일) ...


# -------------------------------
# 5️⃣ 응급 대기 시뮬레이션 (간이 진단서 기반)
# -------------------------------
st.subheader("5️⃣ 응급환자 진단 및 대기열 관리 시뮬레이션")

# 대기 방식 선택 라디오 버튼 (이제 이 값이 큐 동작에 영향을 미침)
# 이 라디오 버튼을 여기에 옮겨야 진단서 작성 시점에 값을 참조할 수 있습니다.
mode = st.radio("동일 중증도 내 대기 방식 선택", ['큐 (선입선출)', '스택 (후입선출)'])


# 진단서 작성 섹션
with st.expander("📝 환자 진단서 작성", expanded=True):
    st.write("환자의 상태를 입력하여 응급도를 평가합니다.")

    patient_name = st.text_input("환자 이름", value="")

    q1 = st.selectbox("1. 의식 상태", ["명료", "기면 (졸림)", "혼미 (자극에 반응)", "혼수 (자극에 무반응)"])
    q2 = st.selectbox("2. 호흡 곤란 여부", ["없음", "가벼운 곤란", "중간 곤란", "심한 곤란"])
    q3 = st.selectbox("3. 주요 통증/출혈 정도", ["없음", "경미", "중간", "심함"])
    q4 = st.selectbox("4. 외상 여부", ["없음", "찰과상/멍", "열상/골절 의심", "다발성 외상/심각한 출혈"])

    submit_diagnosis = st.button("진단 완료 및 큐에 추가")

    if submit_diagnosis and patient_name:
        current_priority_score = 0
        current_severity_level = "경증" 

        if q1 == "기면 (졸림)": current_priority_score += 3
        elif q1 == "혼미 (자극에 반응)": current_priority_score += 7
        elif q1 == "혼수 (자극에 무반응)": current_priority_score += 15

        if q2 == "가벼운 곤란": current_priority_score += 4
        elif q2 == "중간 곤란": current_priority_score += 9
        elif q2 == "심한 곤란": current_priority_score += 20

        if q3 == "경미": current_priority_score += 2
        elif q3 == "중간": current_priority_score += 6
        elif q3 == "심함": current_priority_score += 12

        if q4 == "찰과상/멍": current_priority_score += 3
        elif q4 == "열상/골절 의심": current_priority_score += 8
        elif q4 == "다발성 외상/심각한 출혈": current_priority_score += 18
        
        if current_priority_score >= 35:
            current_severity_level = "매우_응급"
        elif current_priority_score >= 20:
            current_severity_level = "응급"
        elif current_priority_score >= 10:
            current_severity_level = "중증"
        elif current_priority_score >= 3:
            current_severity_level = "중등증"
        else:
            current_severity_level = "경증"

        final_priority_score = severity_scores.get(current_severity_level, 1)

        patient_info = {
            "이름": patient_name,
            "중증도": current_severity_level,
            "의식 상태": q1,
            "호흡 곤란": q2,
            "통증/출혈": q3,
            "외상": q4,
            "계산된 점수": final_priority_score 
        }
        
        # ✨ 큐 타입(mode)을 insert 함수에 전달 ✨
        st.session_state.priority_queue.insert(patient_info, final_priority_score, queue_type=mode)
        st.success(f"'{patient_name}' 환자가 '{current_severity_level}' (점수: {final_priority_score}) 상태로 큐에 추가되었습니다.")
        st.rerun() # 큐 현황을 즉시 업데이트하기 위해

    elif submit_diagnosis and not patient_name:
        st.warning("환자 이름을 입력해주세요.")

# 대기열 현황 및 진료 섹션
st.markdown("#### 🏥 현재 응급 대기열 현황")

if not st.session_state.priority_queue.is_empty():
    st.dataframe(pd.DataFrame(st.session_state.priority_queue.get_all_patients_sorted()))
    
    col1, col2 = st.columns(2)
    with col1:
        process_patient = st.button("환자 진료 시작 (가장 응급한 환자)")
        if process_patient:
            processed_patient, score = st.session_state.priority_queue.get_highest_priority_patient()
            if processed_patient: # None이 아닌지 확인
                st.info(f"**{processed_patient['이름']}** 환자 진료를 시작합니다. (중증도: {processed_patient['중증도']}, 점수: {score})")
            else:
                st.warning("진료할 환자가 없습니다.")
            st.rerun() 
    with col2:
        # 이 부분은 이제 실제 로직에 반영되므로, 안내 문구는 불필요하거나 변경 가능
        st.markdown(f"현재 선택된 대기 방식: **{mode}** (동일 중증도 내 적용)")
else:
    st.info("현재 응급 대기 환자가 없습니다.")

st.markdown("---")
st.caption("ⓒ 2025 스마트 응급의료 데이터 분석 프로젝트 - SDG 3.8 보건서비스 접근성 개선")
