import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque
import os
import chardet

# 공간 데이터 및 그래프 처리를 위한 라이브러리 추가 (geopandas 대신 osmnx 사용)
# import geopandas as gpd # geopandas는 더 이상 직접 사용하지 않습니다.
import networkx as nx
# from shapely.geometry import Point, LineString # shapely도 osmnx 내부에서 처리됩니다.
import osmnx as ox # ✨ osmnx 라이브러리 추가 ✨

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
# 🛣️ 도로망 SHP 파일 경로는 이제 필요 없습니다. (osmnx가 직접 다운로드)

# -------------------------------
# 데이터 로딩 함수
# -------------------------------

# ... (load_transport_data 함수는 그대로 유지) ...
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

# ... (load_time_data 함수는 그대로 유지) ...
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

# ... (load_month_data 함수는 그대로 유지) ...
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

# 🛣️ osmnx를 사용하여 도로망 그래프를 로드하고 networkx 그래프로 반환하는 함수
@st.cache_data
def load_road_network_from_osmnx(place_name):
    try:
        # osmnx를 사용하여 특정 지역의 도로망 그래프를 가져옵니다.
        # network_type='drive'는 차량이 통행 가능한 도로만 가져오도록 합니다.
        # simplify=True는 복잡한 지오메트리를 단순화하여 그래프 크기를 줄입니다.
        # retain_all=True는 그래프의 모든 연결 요소를 유지합니다. (큰 그래프의 경우 False 고려)
        # custom_settings는 필요에 따라 추가적인 Overpass API 설정 가능
        st.info(f"'{place_name}' 지역의 도로망 데이터를 OpenStreetMap에서 가져오는 중입니다. 잠시 기다려주세요...")
        
        G = ox.graph_from_place(place_name, network_type='drive', simplify=True, retain_all=True)
        
        # 그래프의 좌표계가 위도/경도(EPSG:4326)인지 확인 (osmnx는 기본적으로 이 형식을 따름)
        # ox.add_edge_speeds(G) # 필요시 도로별 기본 속도 추가
        # ox.add_edge_travel_times(G) # 필요시 각 간선의 통행 시간 계산 (속도 기반)
        
        st.success(f"'{place_name}' 도로망을 NetworkX 그래프로 변환했습니다. 노드 수: {G.number_of_nodes()}, 간선 수: {G.number_of_edges()}")
        return G

    except Exception as e:
        st.error(f"'{place_name}' 도로망 데이터를 OpenStreetMap에서 가져오고 그래프로 변환하는 중 오류 발생: {e}")
        st.warning("네트워크 연결을 확인하거나, 지역 이름이 정확한지 확인해주세요. 너무 큰 지역을 지정하면 메모리 부족이나 타임아웃이 발생할 수 있습니다.")
        return None

# -------------------------------
# 데이터 로드 및 전처리
# -------------------------------
transport_df = load_transport_data(transport_path)

# --- ✨ transport_df 전처리: '시도명' 컬럼 생성 및 보정 ✨ ---
if not transport_df.empty and '소재지전체주소' in transport_df.columns:
    def extract_sido(address):
        if pd.isna(address):
            return None
        
        addr_str = str(address).strip() 
        if not addr_str: 
            return None

        parts = addr_str.split(' ')
        if not parts: 
            return None

        first_part = parts[0]

        if '세종' in first_part:
            return '세종특별자치시'
        
        korean_sido_list = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
                                 "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도",
                                 "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도",
                                 "제주특별자치도"]
            
        for sido in korean_sido_list:
            if first_part in sido: 
                return sido 
        
        for part in parts:
            if isinstance(part, str) and ('특별시' in part or '광역시' in part or '자치시' in part or '자치도' in part):
                if '강원' in part or '전라' in part or '충청' in part or '경상' in part or '경기' in part or '제주' in part:
                    if len(parts) > 1:
                        return f"{parts[0]}{part}"
                    else:
                        return part
                return part

        return None 


    transport_df['시도명'] = transport_df['소재지전체주소'].apply(extract_sido)

    transport_df.dropna(subset=['시도명'], inplace=True)
    
    st.info("'소재지전체주소' 컬럼을 기반으로 '시도명' 컬럼을 생성하고 보정했습니다.")
elif not transport_df.empty:
    st.warning("'transport_df'에 '소재지전체주소' 컬럼이 없습니다. '시도명' 생성을 건너킵니다.")
# --- ✨ 전처리 끝 ✨ ---

time_df = load_time_data(time_json_path)
month_df = load_month_data(month_json_path)

# 🛣️ 도로망 그래프 로드 (osmnx 함수로 변경)
# 'Yongin-si, Gyeonggi-do, South Korea' 또는 '서울특별시' 등으로 변경 가능
road_graph = load_road_network_from_osmnx("Yongin-si, Gyeonggi-do, South Korea") 


# -------------------------------
# 사이드바 사용자 상호작용
# -------------------------------
st.sidebar.title("사용자 설정")
if not time_df.empty and not month_df.empty:
    # transport_df의 시도명도 추가하여 공통 지역 선택에 활용
    all_regions = set(time_df['시도']) | set(month_df['시도'])
    if not transport_df.empty and '시도명' in transport_df.columns:
        all_regions |= set(transport_df['시도명'].unique()) # transport_df의 시도명도 추가
    
    if all_regions:
        region = st.sidebar.selectbox("지역 선택", sorted(list(all_regions)))
    else:
        st.sidebar.warning("데이터에 공통 지역이 없습니다.")
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
    
    if '시도명' in transport_df.columns and transport_df['시도명'].notna().any(): 
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        # 특정 지역이 선택되었을 경우 해당 지역만 필터링하여 이송 건수 시각화
        if region and region in transport_df['시도명'].unique():
            transport_df[transport_df['시도명'] == region].groupby('시도명').size().plot(kind='barh', ax=ax1, color='skyblue') 
            ax1.set_title(f"{region} 시도별 이송 건수") # 제목 변경
        else:
            transport_df.groupby('시도명').size().sort_values(ascending=False).plot(kind='barh', ax=ax1, color='skyblue') 
            ax1.set_title("시도별 이송 건수")
        
        ax1.set_xlabel("건수")
        ax1.set_ylabel("시도")
        plt.tight_layout() 
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
    # 선택된 지역에 대한 시간대별 데이터를 찾음
    time_row = time_df[time_df['시도'] == region]
    if not time_row.empty:
        time_row_data = time_row.iloc[0, 1:]
        fig2, ax2 = plt.subplots()
        time_row_data.plot(kind='bar', color='deepskyblue', ax=ax2)
        ax2.set_ylabel("이용 건수")
        ax2.set_xlabel("시간대")
        ax2.set_title(f"{region} 시간대별 응급실 이용")
        st.pyplot(fig2)
    else:
        st.warning(f"'{region}' 지역에 대한 시간대별 데이터가 없습니다.")
else:
    st.warning("시간대별 데이터 로드에 문제가 있거나 지역이 선택되지 않았습니다.")

# -------------------------------
# 3️⃣ 월별 분석
# -------------------------------
st.subheader("3️⃣ 월별 응급실 이용 현황 (2023)")
if not month_df.empty and region:
    # 선택된 지역에 대한 월별 데이터를 찾음
    month_row = month_df[month_df['시도'] == region]
    if not month_row.empty:
        month_row_data = month_row.iloc[0, 1:]
        fig3, ax3 = plt.subplots()
        month_row_data.plot(kind='line', marker='o', color='seagreen', ax=ax3)
        ax3.set_ylabel("이용 건수")
        ax3.set_xlabel("월")
        ax3.set_title(f"{region} 월별 응급실 이용")
        st.pyplot(fig3)
    else:
        st.warning(f"'{region}' 지역에 대한 월별 데이터가 없습니다.")
else:
    st.warning("월별 데이터 로드에 문제가 있거나 지역이 선택되지 않았습니다.")


# -------------------------------
# 4️⃣ 도로망 그래프 정보 (osmnx로 변경)
# -------------------------------
st.subheader("🛣️ 도로망 그래프 정보")
if road_graph:
    st.write(f"**로드된 도로망 그래프 (`{road_graph.graph['place']}`):**")
    st.write(f"  - 노드 수: {road_graph.number_of_nodes()}개")
    st.write(f"  - 간선 수: {road_graph.number_of_edges()}개")
    
    # 맵 위에 그래프 시각화 (간단한 예시)
    st.write("간단한 도로망 지도 시각화 (노드와 간선):")
    fig, ax = ox.plot_graph(road_graph, show=False, close=False, bgcolor='white', node_color='red', node_size=5, edge_color='gray', edge_linewidth=0.5)
    st.pyplot(fig) # Streamlit에 Matplotlib 그림 표시
    st.caption("참고: 전체 도로망은 복잡하여 로딩이 느릴 수 있습니다.")

else:
    st.warning("도로망 그래프 로드에 실패했습니다. 지정된 지역을 확인해주세요.")


# -------------------------------
# 5️⃣ 우선순위 큐 시뮬레이션
# -------------------------------
st.subheader("5️⃣ 응급 대기 시뮬레이션 (스택/큐 모델)")
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
