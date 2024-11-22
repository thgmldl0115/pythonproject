import cx_Oracle
import pandas as pd

# 파일 경로와 필요한 컬럼
file_path = '건물에너지DB_좌표매칭_최종(15-18).csv'
columns_to_read = ['LOTNO_ADDR', 'ROAD_NM_ADDR', 'SGNG_CD', 'STDG_CD', 'LOTNO_MNO', 'LOTNO_SNO', 'GPS_LOT', 'GPS_LAT', 'STNDD_YR', 'USE_MM', 'ELRW_USQNT', 'CTY_GAS_USQNT', 'SUM_NRG_USQNT', 'ELRW_TOE_USQNT', 'CTY_GAS_TOE_USQNT', 'SUM_NRG_TOE_USQNT', 'ELRW_GRGS_DSAMT', 'CTY_GAS_GRGS_DSAMT', 'SUM_GRGS_DSAMT']

# CSV 파일 읽기 및 필요한 열 선택
data = pd.read_csv(file_path, usecols=columns_to_read)


# 30230 : 대덕구
# 30110 : 동구
# 30170 : 서구
# 30200 : 유성구
# 30140 : 중구

# 포함하고자 하는 문자열 리스트 생성
dj_list = ['30230', '30110', '30170', '30200', '30140']

# 데이터프레임 생성
df = pd.DataFrame(data)

# join함수를 이용하여 이어주고 contains 함수에 넣기
test = '|'.join(dj_list)

# '대전광역시'가 포함된 데이터 필터링
filtered_data = data[data['SGNG_CD'].astype(str).str.contains(test, na=False)]

# Oracle DB 연결 설정
oracle_connection = cx_Oracle.connect(
    user="team1",         # Oracle 사용자 이름
    password="team1",     # 비밀번호
    dsn="192.168.0.42:1521/xe"  # Oracle DSN 정보
)

# 커서 생성
cursor = oracle_connection.cursor()

# 컬럼 이름 수동 할당
data.columns = [
    'LOTNO_ADDR', 'ROAD_NM_ADDR', 'SGNG_CD', 'STDG_CD', 'LOTNO_MNO', 'LOTNO_SNO',
    'GPS_LOT', 'GPS_LAT', 'STNDD_YR', 'USE_MM', 'ELRW_USQNT', 'CTY_GAS_USQNT',
    'SUM_NRG_USQNT', 'ELRW_TOE_USQNT', 'CTY_GAS_TOE_USQNT', 'SUM_NRG_TOE_USQNT',
    'ELRW_GRGS_DSAMT', 'CTY_GAS_GRGS_DSAMT', 'SUM_GRGS_DSAMT'
]

# 모든 데이터를 문자열로 변환 및 NaN 값 처리
data = data.fillna('').astype(str)

# 테이블에 삽입할 SQL 쿼리
insert_query = """
    INSERT INTO DAEJEON_ENERGY (

        LOTNO_ADDR, ROAD_NM_ADDR, SGNG_CD, STDG_CD, LOTNO_MNO, LOTNO_SNO,
        GPS_LOT, GPS_LAT, STNDD_YR, USE_MM, ELRW_USQNT, CTY_GAS_USQNT,
        SUM_NRG_USQNT, ELRW_TOE_USQNT, CTY_GAS_TOE_USQNT, SUM_NRG_TOE_USQNT,
        ELRW_GRGS_DSAMT, CTY_GAS_GRGS_DSAMT, SUM_GRGS_DSAMT, PK_CD
    ) VALUES (
        :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20
    )
"""

idx = 1305177
# 데이터프레임 데이터를 테이블에 삽입
try:
    for index, row in data.iterrows():
        idx += 1
        try:
            cursor.execute(insert_query, (
                row['LOTNO_ADDR'],
                row['ROAD_NM_ADDR'],
                row['SGNG_CD'],
                row['STDG_CD'],
                row['LOTNO_MNO'],
                row['LOTNO_SNO'],
                row['GPS_LOT'],
                row['GPS_LAT'],
                row['STNDD_YR'],
                row['USE_MM'],
                row['ELRW_USQNT'],
                row['CTY_GAS_USQNT'],
                row['SUM_NRG_USQNT'],
                row['ELRW_TOE_USQNT'],
                row['CTY_GAS_TOE_USQNT'],
                row['SUM_NRG_TOE_USQNT'],
                row['ELRW_GRGS_DSAMT'],
                row['CTY_GAS_GRGS_DSAMT'],
                row['SUM_GRGS_DSAMT'], idx
            ))
        except Exception as row_error:
            print(f"오류 발생 (행 {index}): {row_error}")
            print(f"문제 데이터: {row.to_dict()}")
    # 커밋
    oracle_connection.commit()
    print(f"{len(data)}개의 행이 성공적으로 삽입되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
    oracle_connection.rollback()
finally:
    # 커서 및 연결 종료
    cursor.close()
    oracle_connection.close()