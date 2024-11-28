from flask import Flask, request, jsonify
import cx_Oracle
import numpy as np
import pandas as pd
import joblib
import os

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# DB 연결 설정
db_config = {
    "user": "team1",
    "password": "team1",
    "dsn": "192.168.0.42:1521/xe"
}

# 모델 및 매핑 정보 로드
model_dir = "saved_models_with_area"
label_mapping_path = os.path.join(model_dir, "label_mapping.joblib")
label_mapping = joblib.load(label_mapping_path)

# 용도별 면적 그룹 경계값 로드
bins_path = os.path.join(model_dir, "area_bins_by_usage.joblib")
bins_by_usage = joblib.load(bins_path)

@app.route('/predict', methods=['GET'])
def predict():
    # GET 요청에서 입력 데이터 받기
    code = str(request.args.get('code'))  # 코드값 (예: '01000')
    bldg_arch_area = float(request.args.get('bldg_arch_area'))  # 건물 면적
    year = int(request.args.get('year'))  # 기준 연도
    month = int(request.args.get('month'))  # 기준 월

    # if code not in label_mapping:
    #     return jsonify({"error": f"Invalid code: {code}"}), 400

    # 코드에 따른 라벨 찾기
    target_label = label_mapping[code]

    # 해당 코드의 그룹 경계값 확인
    if target_label not in bins_by_usage:
        return jsonify({"error": f"No group bins found for code {code}"}), 400

    # 면적 그룹 계산
    bins = bins_by_usage[target_label]
    area_group = np.digitize([bldg_arch_area], bins, right=True)[0]  # 면적에 따른 그룹 계산


    # 데이터베이스 연결
    try:
        conn = cx_Oracle.connect(**db_config)
        cursor = conn.cursor()

        # 12개월 데이터 가져오기
        query = f"""
    
        SELECT 
        BLDG_ARCH_AREA, -- 건물건축면적
        STNDD_YR,    -- 기준년도 
        USE_MM,      -- 사용월
        ROUND(AVG(ELRW_USQNT)) as ELRW_USQNT   -- 전력 에너지 사용량
        FROM ML_DATA
        WHERE mjr_prps_cd = :code
        AND ((STNDD_YR = :year AND USE_MM <= :month) OR (STNDD_YR = :prev_year AND USE_MM > :prev_month))
        GROUP BY  STNDD_YR, USE_MM, BLDG_ARCH_AREA
        ORDER BY  STNDD_YR, TO_NUMBER(USE_MM)
        """
        prev_year = year - 1
        prev_month = (month + 1) % 12
        cursor.execute(query,
                       {"code": code, "year": year, "month": month, "prev_year": prev_year, "prev_month": prev_month})
        rows = cursor.fetchall()

        # 데이터프레임 생성
        df = pd.DataFrame(rows, columns=['ELRW_USQNT', 'STNDD_YR', 'USE_MM', 'BLDG_ARCH_AREA'])
        df.sort_values(by=['STNDD_YR', 'USE_MM'], inplace=True)

        if len(df) < 12:
            return jsonify({"error": "Not enough data to predict"}), 400

        # 면적 그룹 계산 및 면적 그룹의 평균값으로 대체
        df['AREA_GROUP'] = np.digitize(df['BLDG_ARCH_AREA'], bins, right=True)
        group_means = df.groupby('AREA_GROUP')['BLDG_ARCH_AREA'].mean().to_dict()
        df['BLDG_ARCH_AREA'] = df['AREA_GROUP'].map(group_means)  # 그룹의 평균값으로 대체

        # 최근 12개월 데이터를 사용
        recent_data = df.iloc[-12:]
        usage_sequence = recent_data['ELRW_USQNT'].values
        area_sequence = recent_data['BLDG_ARCH_AREA'].values
        year_month_sequence = recent_data[['STNDD_YR', 'USE_MM']].values.flatten()

        # 입력 데이터 생성
        input_sequence = []
        for i in range(12):
            input_sequence.extend(
                [usage_sequence[i], area_sequence[i], year_month_sequence[i * 2], year_month_sequence[i * 2 + 1]])
        input_sequence = np.array(input_sequence).reshape(1, -1)

        # 해당 코드와 그룹의 모델 로드
        model_path = os.path.join(model_dir, f"model_area_{area_group}_code_{target_label}.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model for code {code} in group {area_group} not found"}), 404
        model = joblib.load(model_path)

        # 정규화 스케일러 로드
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        scaler = joblib.load(scaler_path)

        # 6개월 예측
        predictions = []
        for i in range(6):
            # 모델 예측
            predicted_normalized = model.predict(input_sequence)

            # scaler.inverse_transform을 적용하려면 scaler가 학습한 입력 차원을 맞춰야 합니다.
            # 예측된 값에 기존 차원 맞추기 (예: [predicted_value, placeholder_value])
            placeholder = 0  # 추가 차원을 채우기 위한 임시 값 (BLDG_ARCH_AREA 자리)
            predicted_normalized_full = [[predicted_normalized[0], placeholder]]

            # 스케일링 복원 (inverse_transform)
            predicted_actual_full = scaler.inverse_transform(predicted_normalized_full)[0]  # 복원된 전체 데이터
            predicted_actual = predicted_actual_full[0]  # ELRW_USQNT 값만 사용

            # 결과 저장
            predictions.append({"year": year, "month": month, "predicted_usage": predicted_actual})

            # 다음 입력 데이터 업데이트
            # predicted_actual을 스케일링하여 next_usage로 변환
            next_usage = scaler.transform([[predicted_actual, placeholder]])[0][0]  # 다시 정규화된 값으로 변환
            input_sequence = np.roll(input_sequence, -4)  # 4개의 특징 이동 (usage, area, year, month)
            input_sequence[0, -4:] = [next_usage, area_sequence[-1], year, month]

            # 월/년도 업데이트
            month += 1
            if month > 12:
                month = 1
                year += 1

        # 결과 JSON 반환
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if 'conn' in locals():
            conn.close()

# Flask 서버 실행
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=5500, host="0.0.0.0")  # 로컬 호스트로도, 휴대폰 ip로도 접근 가능. 즉, 배포가능
