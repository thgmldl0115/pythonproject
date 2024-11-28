# 필요한 라이브러리 불러오기
import cx_Oracle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Oracle 데이터베이스 연결
conn = cx_Oracle.connect("team1", "team1", "192.168.0.42:1521/xe")

# SQL 쿼리
query = """
SELECT a.mjr_prps_cd,  -- 주요용도코드  
       b.STNDD_YR,    -- 기준년도 
       b.USE_MM,      -- 사용월
       ROUND(AVG(b.ELRW_USQNT)) as ELRW_USQNT   -- 전력 에너지 사용량
FROM building_usage1 a, daejeon_energy1 b
WHERE a.lotno_addr = b.lotno_addr
AND a.mjr_prps_nm IS NOT NULL
AND b.stndd_yr between 2015 and 2022
GROUP BY mjr_prps_cd, STNDD_YR, USE_MM
ORDER BY mjr_prps_cd, STNDD_YR, TO_NUMBER(USE_MM)
"""

# 데이터 가져오기
df = pd.read_sql(query, conn)

# 연결 종료
conn.close()
# 데이터 준비 및 전처리
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# 주요 용도 코드 라벨링
df['MJR_PRPS_CD_LABEL'] = label_encoder.fit_transform(df['MJR_PRPS_CD'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# 전력 사용량 정규화
df[['ELRW_USQNT']] = scaler.fit_transform(df[['ELRW_USQNT']])

# 12개월 데이터를 입력으로 변환
sequence_length = 12  # 최근 12개월 데이터 사용
features = []
targets = []

for i in range(len(df) - sequence_length):
    # 12개월치 데이터를 입력으로 사용
    sequence = df.iloc[i:i+sequence_length][['ELRW_USQNT', 'MJR_PRPS_CD_LABEL', 'STNDD_YR', 'USE_MM']].values.flatten()
    target = df['ELRW_USQNT'].iloc[i+sequence_length]  # 다음 달 데이터가 타겟
    features.append(sequence)
    targets.append(target)

# 배열 변환
X = np.array(features)
y = np.array(targets)

# 학습 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor 모델 학습
model = RandomForestRegressor(
    random_state=42,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=4,
    min_samples_split=2,
    bootstrap=True
)

model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = model.predict(X_test)

# 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("모델 평가")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R² Score: {r2}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual", marker='o', linestyle='--')
plt.plot(y_pred, label="Predicted", marker='x', linestyle='-')
plt.legend()
plt.title("Actual vs Predicted (Test Set)")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Usage")
plt.show()

# 특정 시점의 다음 달 예측 (2020년 12월 기준)
# 학습 데이터 생성 방식 재확인
sequence_length = 12
features = []

for i in range(len(df) - sequence_length):
    # 학습 데이터 생성 방식
    sequence = df.iloc[i:i + sequence_length][
        ['ELRW_USQNT', 'MJR_PRPS_CD_LABEL', 'STNDD_YR', 'USE_MM']].values.flatten()
    features.append(sequence)

# 학습 데이터의 특성 수 확인
feature_count = len(features[0])
print(f"학습 데이터의 특성 수: {feature_count}")

# 특정 시점의 다음 달 예측 (2020년 12월 기준)
december_data = df[(df['STNDD_YR'] == '2022') & (df['USE_MM'] == '1') & (df['MJR_PRPS_CD'] == '01000')]
if not december_data.empty:
    december_index = december_data.index[0]
    if december_index >= sequence_length:
        input_sequence = []
        for i in range(sequence_length):
            # 1개월치 데이터 생성
            usage = df['ELRW_USQNT'].iloc[december_index - sequence_length + i]  # 사용량
            mjr_prps_cd = df['MJR_PRPS_CD_LABEL'].iloc[december_index - sequence_length + i]  # 주요 용도 코드
            year = df['STNDD_YR'].iloc[december_index - sequence_length + i]  # 연도
            month = df['USE_MM'].iloc[december_index - sequence_length + i]  # 월

            # 1개월 데이터를 리스트에 추가 (사용량 → 구분 코드 → 연도 → 월 순서)
            input_sequence.extend([usage, mjr_prps_cd, year, month])

        # 입력 데이터 배열로 변환
        input_sequence = np.array(input_sequence).reshape(1, -1)

        # 디버깅 용도: 입력 데이터 확인
        print("Input Sequence Shape:", input_sequence.shape)
        print("Input Sequence Data:", input_sequence)

        # 예측 수행
        predicted_normalized = model.predict(input_sequence)
        predicted_actual = scaler.inverse_transform([[predicted_normalized[0]]])[0][0]
        print(f"2021년 1월 예측 사용량: {predicted_actual:.2f}")
    else:
        print("2020년 12월 이전의 데이터가 부족하여 예측 불가능")
