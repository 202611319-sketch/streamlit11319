import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 / 스케일러 불러오기
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# 데이터 불러오기
df = pd.read_csv("lung.csv")

# 제목
st.title("폐 질환 군집 예측 프로그램")

st.write("사용자의 정보를 입력하면 어떤 군집에 속하는지 예측합니다.")

# 사용자 입력
Age = st.number_input("나이", min_value=0.0, max_value=120.0, value=30.0)
Smokes = st.number_input("흡연량", min_value=0.0, value=0.0)
Alkhol = st.number_input("음주량", min_value=0.0, value=0.0)

# 버튼
if st.button("예측하기"):

    # 새 데이터 생성
    new_patient = pd.DataFrame(
        [[Age, Smokes, Alkhol]],
        columns=['나이', '흡연', '음주']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 군집 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 군집 번호 저장
    cluster_num = int(pred_cluster[0])

    # 결과 출력
    st.success(f"이 환자는 {cluster_num}번 군집에 속합니다.")

    # 상태별 메시지
    if cluster_num == 0:
        st.success("폐 질환 위험도가 높은 상태입니다.")

    elif cluster_num == 1:
        st.warning("폐 관련 위험이 높은 상태입니다.")

    else:
        st.error("건강 위험이 낮은 상태입니다.")

    # 그래프
    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        df['나이'],
        df['흡연'],
        c=df['cluster'],
        alpha=0.5
    )

    # 새 환자 표시
    ax.scatter(
        Age,
        Smokes,
        c='black',
        s=300,
        marker='X'
    )

    ax.set_xlabel('나이')
    ax.set_ylabel('흡연')
    ax.set_title('폐 질환 군집 결과')

    st.pyplot(fig)