# 필요한 라이브러리 불러오기
import cx_Oracle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib  # 모델 저장/로드 라이브러리

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
AND b.stndd_yr between 2015 and 2021
GROUP BY mjr_prps_cd, STNDD_YR, USE_MM
ORDER BY mjr_prps_cd, STNDD_YR, TO_NUMBER(USE_MM)
"""

# 데이터 가져오기
df = pd.read_sql(query, conn)

# 연결 종료
conn.close()
model_dir = "saved_models"

# 데이터 준비 및 전처리
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# 주요 용도 코드 라벨링
df['MJR_PRPS_CD_LABEL'] = label_encoder.fit_transform(df['MJR_PRPS_CD'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# 전력 사용량 정규화
df[['ELRW_USQNT']] = scaler.fit_transform(df[['ELRW_USQNT']])


# 모델 저장 디렉토리 생성

os.makedirs(model_dir, exist_ok=True)

# label_mapping 저장 파일 경로
label_mapping_path = os.path.join(model_dir, "label_mapping.joblib")

# label_mapping 저장
joblib.dump(label_mapping, label_mapping_path)
print("label_mapping 저장 완료:", label_mapping_path)

# 구분 코드별로 데이터 분리 및 모델 학습
models = {}  # 구분 코드별 모델 저장
unique_codes = df['MJR_PRPS_CD_LABEL'].unique()  # 고유 구분 코드 목록

for code in unique_codes:
    # 구분 코드에 해당하는 데이터만 필터링
    code_data = df[df['MJR_PRPS_CD_LABEL'] == code].reset_index(drop=True)
    # 스케일러 저장
    scaler_path = os.path.join(model_dir, f"scaler_code_{code}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"스케일러 저장 완료 (코드 {code}):", scaler_path)

    # 12개월 데이터를 입력으로 변환
    sequence_length = 12  # 최근 12개월 데이터 사용
    features = []
    targets = []

    # 데이터 길이가 충분하지 않으면 건너뛰기
    if len(code_data) <= sequence_length:
        print(f"\n구분 코드 {code} 데이터가 부족하여 모델을 생성할 수 없습니다. (데이터 수: {len(code_data)})")
        continue


    for i in range(len(code_data) - sequence_length):
        # 12개월치 데이터를 입력으로 사용
        sequence = code_data.iloc[i:i + sequence_length][['ELRW_USQNT', 'STNDD_YR', 'USE_MM']].values.flatten()
        target = code_data['ELRW_USQNT'].iloc[i + sequence_length]  # 다음 달 데이터가 타겟
        features.append(sequence)
        targets.append(target)

    # 배열 변환
    X = np.array(features)
    y = np.array(targets)

    # 데이터가 충분한지 확인
    if len(X) == 0 or len(y) == 0:
        print(f"\n구분 코드 {code} 데이터가 부족하여 모델을 생성할 수 없습니다.")
        continue

    # 학습 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"\n구분 코드 {code} 데이터가 부족하여 모델을 생성할 수 없습니다. (학습/테스트 데이터 부족)")
        continue

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

    # 모델 저장
    model_path = os.path.join(model_dir, f"model_code_{code}.joblib")
    joblib.dump(model, model_path)  # 모델 저장
    models[code] = model

    # 테스트 데이터 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n구분 코드 {code} 모델 평가")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R² Score: {r2}")

# 특정 시점의 다음 달 예측 (구분 코드별 예측)
# label_mapping 로드
label_mapping = joblib.load(label_mapping_path)
print("\nlabel_mapping 로드 완료:", label_mapping)

target_code = label_mapping['01000']  # 예측할 구분 코드
code_data = df[df['MJR_PRPS_CD_LABEL'] == target_code].reset_index(drop=True)

# 특정 시점(2020년 12월 기준)의 데이터
december_data = code_data[(code_data['STNDD_YR'] == '2020') & (code_data['USE_MM'] == '12')]
if not december_data.empty:
    december_index = december_data.index[0]
    if december_index >= sequence_length:
        input_sequence = []
        for i in range(sequence_length):
            # 1개월치 데이터 생성
            usage = code_data['ELRW_USQNT'].iloc[december_index - sequence_length + i]  # 사용량
            year = code_data['STNDD_YR'].iloc[december_index - sequence_length + i]  # 연도
            month = code_data['USE_MM'].iloc[december_index - sequence_length + i]  # 월

            # 1개월 데이터를 리스트에 추가 (사용량 → 연도 → 월 순서)
            input_sequence.extend([usage, year, month])

        # 입력 데이터 배열로 변환
        input_sequence = np.array(input_sequence).reshape(1, -1)

        # 디버깅 용도: 입력 데이터 확인
        print("\nInput Sequence Shape:", input_sequence.shape)
        print("Input Sequence Data:", input_sequence)

        # 저장된 모델 로드 후 예측 수행
        model_path = os.path.join(model_dir, f"model_code_{target_code}.joblib")
        model = joblib.load(model_path)  # 모델 로드
        predicted_normalized = model.predict(input_sequence)
        predicted_actual = scaler.inverse_transform([[predicted_normalized[0]]])[0][0]
        print(f"2021년 1월 예측 사용량 (구분 코드 {target_code}): {predicted_actual:.2f}")
    else:
        print("2020년 12월 이전의 데이터가 부족하여 예측 불가능")
else:
    print("2020년 12월 데이터가 없습니다.")

