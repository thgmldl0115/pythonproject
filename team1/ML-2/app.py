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
model_dir = "saved_models"
label_mapping_path = os.path.join(model_dir, "label_mapping.joblib")
label_mapping = joblib.load(label_mapping_path)



@app.route('/predict', methods=['GET'])
def predict():
    # GET 요청에서 입력 데이터 받기
    code = request.args.get('code')  # 코드값 (예: '01000')
    year = int(request.args.get('year'))  # 기준 년도
    month = int(request.args.get('month'))  # 기준 월

    if code not in label_mapping:
        return jsonify({"error": f"Invalid code: {code}"}), 400

    # 데이터베이스 연결
    try:
        conn = cx_Oracle.connect(**db_config)
        cursor = conn.cursor()

        # 12개월 데이터 가져오기
        query = f"""
        SELECT b.ELRW_USQNT, b.STNDD_YR, b.USE_MM
        FROM daejeon_energy1 b
        JOIN building_usage1 a ON a.lotno_addr = b.lotno_addr
        WHERE a.mjr_prps_cd = :code
        AND ((b.STNDD_YR = :year AND b.USE_MM <= :month) OR (b.STNDD_YR = :prev_year AND b.USE_MM > :prev_month))
        ORDER BY b.STNDD_YR, b.USE_MM
        """
        prev_year = year - 1
        prev_month = (month + 1) % 12
        cursor.execute(query, {"code": code, "year": year, "month": month, "prev_year": prev_year, "prev_month": prev_month})
        rows = cursor.fetchall()

        # 데이터프레임 생성
        df = pd.DataFrame(rows, columns=['ELRW_USQNT', 'STNDD_YR', 'USE_MM'])
        df.sort_values(by=['STNDD_YR', 'USE_MM'], inplace=True)

        if len(df) < 12:
            return jsonify({"error": "Not enough data to predict"}), 400

        # 최근 12개월 데이터를 사용
        recent_data = df.iloc[-12:]
        usage_sequence = recent_data['ELRW_USQNT'].values
        year_month_sequence = recent_data[['STNDD_YR', 'USE_MM']].values.flatten()

        # 입력 데이터 생성
        input_sequence = []
        for i in range(12):
            input_sequence.extend([usage_sequence[i], year_month_sequence[i * 2], year_month_sequence[i * 2 + 1]])
        input_sequence = np.array(input_sequence).reshape(1, -1)

        # 해당 코드의 모델 로드
        target_label = label_mapping[code]
        model_path = os.path.join(model_dir, f"model_code_{target_label}.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model for code {code} not found"}), 404
        model = joblib.load(model_path)

        # 정규화 스케일러 (미리 저장되어 있다고 가정)
        scaler_path = os.path.join(model_dir,  f"scaler_code_{target_label}.joblib")
        scaler = joblib.load(scaler_path)

        # 6개월 예측
        predictions = []
        for i in range(6):
            predicted_normalized = model.predict(input_sequence)
            predicted_actual = scaler.inverse_transform([[predicted_normalized[0]]])[0][0]
            predictions.append({"year": year, "month": month, "predicted_usage": predicted_actual})

            # 다음 입력 데이터 업데이트
            next_usage = scaler.transform([[predicted_actual]])[0][0]
            input_sequence = np.roll(input_sequence, -3)
            input_sequence[0, -3:] = [next_usage, year, month]

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
