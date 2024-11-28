# 필요한 라이브러리 불러오기
import cx_Oracle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Oracle 데이터베이스 연결
conn = cx_Oracle.connect("team1", "team1", "192.168.0.42:1521/xe")

# SQL 쿼리
query = """
SELECT mjr_prps_cd,  -- 주요용도코드  
       BLDG_ARCH_AREA, -- 건물건축면적
       STNDD_YR,    -- 기준년도 
       USE_MM,      -- 사용월
       ROUND(AVG(ELRW_USQNT)) as ELRW_USQNT   -- 전력 에너지 사용량
FROM ML_DATA
WHERE STNDD_YR BETWEEN 2015 AND 2021
GROUP BY mjr_prps_cd, STNDD_YR, USE_MM, BLDG_ARCH_AREA
ORDER BY mjr_prps_cd, STNDD_YR, TO_NUMBER(USE_MM)
"""

# 데이터 가져오기
df = pd.read_sql(query, conn)

# 연결 종료
conn.close()

# 모델 저장 디렉토리 생성
model_dir = "saved_models_with_area"
os.makedirs(model_dir, exist_ok=True)

# 주요 용도 코드 라벨링
label_encoder = LabelEncoder()
df['MJR_PRPS_CD_LABEL'] = label_encoder.fit_transform(df['MJR_PRPS_CD'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# 라벨 매핑 저장
label_mapping_path = os.path.join(model_dir, "label_mapping.joblib")
joblib.dump(label_mapping, label_mapping_path)
print(f"라벨 매핑 저장 완료: {label_mapping_path}")

# 용도별 면적 그룹 나누기
bins_by_usage = {}  # 용도별 그룹 경계값 저장

for usage in df['MJR_PRPS_CD_LABEL'].unique():
    usage_data = df[df['MJR_PRPS_CD_LABEL'] == usage]  # 특정 용도 데이터만 필터링

    # 값이 중복되어 qcut 실행이 불가능한 경우 처리
    if len(usage_data['BLDG_ARCH_AREA'].unique()) < 10:
        print(f"용도 {usage}의 BLDG_ARCH_AREA 값이 중복되어 그룹화를 생략합니다.")
        continue

    # qcut으로 그룹화, duplicates='drop'을 설정하여 중복 경계값 제거
    df.loc[df['MJR_PRPS_CD_LABEL'] == usage, 'AREA_GROUP'], bins = pd.qcut(
        usage_data['BLDG_ARCH_AREA'], q=10, labels=False, retbins=True, duplicates='drop'
    )
    bins_by_usage[usage] = bins  # 용도별 그룹 경계값 저장

# 그룹 번호를 1부터 시작하도록 조정
df['AREA_GROUP'] = df['AREA_GROUP'] + 1

# 용도별 그룹 경계값 저장
bins_path = os.path.join(model_dir, "area_bins_by_usage.joblib")
joblib.dump(bins_by_usage, bins_path)
print(f"용도별 면적 그룹 경계값 저장 완료: {bins_path}")

# 전력 사용량 및 건축면적 정규화
scaler = MinMaxScaler()
df[['ELRW_USQNT', 'BLDG_ARCH_AREA']] = scaler.fit_transform(df[['ELRW_USQNT', 'BLDG_ARCH_AREA']])

# 스케일러 저장
scaler_path = os.path.join(model_dir, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"스케일러 저장 완료: {scaler_path}")

# 데이터 축소: 동일한 용도, 그룹, 연도, 월 기준으로 평균 사용량 계산
df = df.groupby(['MJR_PRPS_CD_LABEL', 'AREA_GROUP', 'STNDD_YR', 'USE_MM']).agg({
    'ELRW_USQNT': 'mean',  # 사용량 평균
    'BLDG_ARCH_AREA': 'mean'  # 면적 평균
}).reset_index()

# 데이터 검증
print(f"축소된 데이터 크기: {df.shape}")
print(df.head())

# AREA_GROUP별 모델 학습
models = {}
area_groups = df['AREA_GROUP'].unique()

for group in area_groups:
    group_data = df[df['AREA_GROUP'] == group].reset_index(drop=True)
    unique_codes = group_data['MJR_PRPS_CD_LABEL'].unique()

    for code in unique_codes:
        code_data = group_data[group_data['MJR_PRPS_CD_LABEL'] == code].reset_index(drop=True)

        # 12개월 데이터를 입력으로 변환
        sequence_length = 12
        features = []
        targets = []

        if len(code_data) <= sequence_length:
            print(f"AREA_GROUP {group}, CODE {code} 데이터 부족. (데이터 수: {len(code_data)})")
            continue

        for i in range(len(code_data) - sequence_length):
            sequence = code_data.iloc[i:i + sequence_length][['ELRW_USQNT', 'BLDG_ARCH_AREA', 'STNDD_YR', 'USE_MM']].values.flatten()
            target = code_data['ELRW_USQNT'].iloc[i + sequence_length]
            features.append(sequence)
            targets.append(target)

        X = np.array(features)
        y = np.array(targets)

        if len(X) == 0 or len(y) == 0:
            print(f"AREA_GROUP {group}, CODE {code} 데이터가 부족합니다.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RandomForestRegressor 학습
        model = RandomForestRegressor(
            random_state=42,
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=4,
            min_samples_split=2,
            bootstrap=True
        )
        print(f"AREA_GROUP {group}, CODE {code} 모델 학습 중...")
        model.fit(X_train, y_train)

        # 모델 저장
        model_path = os.path.join(model_dir, f"model_area_{int(group)}_code_{code}.joblib")
        joblib.dump(model, model_path)

        # 테스트 데이터 예측
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"AREA_GROUP {group}, CODE {code} 평가")
        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

print("모델 학습 및 저장 완료.")