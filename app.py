import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- 웹 페이지 기본 설정 ---
st.set_page_config(page_title="암표어사 비정상행동 군집화 및 감지 알고리즘", page_icon="🛡️", layout="wide")

st.title("암표상 행동 탐지 AI 데모")
st.markdown("""
이 대시보드는 **DBSCAN(밀도 기반 군집화) 알고리즘**을 활용하여, 사람(진짜 팬)과 기계(매크로 암표상)의 행동 데이터를 시각적으로 분리해 내는 데모입니다.
* **진짜 팬:** 예매 소요 시간, 마우스 클릭 패턴 등이 사람마다 달라 넓게 퍼져 있습니다. (노이즈로 분류)
* **조직적 암표상(매크로):** 기계적으로 세팅된 값을 사용하므로, 특정 행동 좌표에 뭉쳐 있습니다. (군집으로 분류)
""")

# --- 1. 가상 데이터 생성 (Mock Data) ---
np.random.seed(42)

# 진짜 팬 (정상 유저) - 불규칙하고 넓게 퍼진 데이터 (300명)
# X축: 예매 소요 시간 (초), Y축: 단기간 리셀 페이지 접근 횟수
normal_users_x = np.random.uniform(15, 300, 300)
normal_users_y = np.random.uniform(0, 20, 300)
normal_users = np.column_stack((normal_users_x, normal_users_y))

# 조직적 암표상 그룹 1 (초고속 매크로 봇 - 50개 계정)
bot_group1_x = np.random.normal(4, 1.1, 50)  # 4초 만에 예매 완벽 완료
bot_group1_y = np.random.normal(45, 1.2, 50)   # 리셀 페이지 접근 빈도 매우 높음
bot_group1 = np.column_stack((bot_group1_x, bot_group1_y))

# 조직적 암표상 그룹 2 (아옮 시도 봇 - 40개 계정)
bot_group2_x = np.random.normal(12, 1.2, 40)
bot_group2_y = np.random.normal(30, 1.8, 40)
bot_group2 = np.column_stack((bot_group2_x, bot_group2_y))

# 데이터 병합
X = np.vstack((normal_users, bot_group1, bot_group2))
df = pd.DataFrame(X, columns=['예매 소요 시간 (초)', '단기 리셀 시도 횟수'])

# --- 2. 사이드바 (파라미터 조절) ---
st.sidebar.header("⚙️ AI 모델 파라미터 조절")
st.sidebar.markdown("DBSCAN 알고리즘의 민감도를 조절해 보세요.")
eps = st.sidebar.slider("Epsilon (탐색 반경)", min_value=0.05, max_value=1.0, value=0.15, step=0.05)
min_samples = st.sidebar.slider("Min Samples (최소 군집 크기)", min_value=3, max_value=30, value=12, step=1)

# --- 3. DBSCAN 모델 적용 ---
# 데이터를 스케일링하여 거리 계산을 정확하게 만듦
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# AI 군집화 실행
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

# 라벨 매핑 (-1: 정상 유저/노이즈, 0 이상: 암표상 군집)
df['Cluster_ID'] = labels
df['유저 분류'] = df['Cluster_ID'].apply(lambda x: '진짜 팬 (정상 유저)' if x == -1 else f'🚨 암표상 의심 그룹 {x+1}')

# --- 4. 데이터 시각화 (Plotly 인터랙티브 차트) ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 유저 행동 데이터 분포 및 탐지 결과")
    
    # 색상 맵 설정 (정상 유저는 연한 파란색, 암표상은 강렬한 빨간색/주황색 계열)
    color_discrete_map = {'진짜 팬 (정상 유저)': '#AEC6CF'}
    for i in range(10):
        color_discrete_map[f'🚨 암표상 의심 그룹 {i+1}'] = '#FF6961' if i%2==0 else '#FFB347'

    fig = px.scatter(
        df, 
        x='예매 소요 시간 (초)', 
        y='단기 리셀 시도 횟수', 
        color='유저 분류',
        color_discrete_map=color_discrete_map,
        hover_data=['예매 소요 시간 (초)', '단기 리셀 시도 횟수'],
        title="DBSCAN 알고리즘 기반 비정상 행동 패턴 감지"
    )
    
    # 차트 디자인 다듬기
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(xaxis_title="예매 완료까지 걸린 시간 (초)", yaxis_title="양도(리셀) 시스템 접근 빈도")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 탐지 요약")
    total_users = len(df)
    normal_count = len(df[df['Cluster_ID'] == -1])
    scalper_count = total_users - normal_count
    
    st.metric("총 접속 계정 수", f"{total_users}개")
    st.metric("진짜 팬 (정상)", f"{normal_count}명", delta="통과", delta_color="normal")
    st.metric("🚨 암표상 의심 적발", f"{scalper_count}개", delta="차단 필요", delta_color="inverse")
    
    st.markdown("---")
    st.markdown("**AI 판단 로직:**\n인간은 매번 클릭 속도나 행동이 다릅니다. 하지만 기계(매크로)는 0.1초의 오차도 없이 동일하게 행동합니다. AI는 이를 **'비정상적인 밀도(군집)'**로 인식하여 즉시 솎아냅니다.")
