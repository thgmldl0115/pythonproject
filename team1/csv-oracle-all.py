import cx_Oracle
import pandas as pd

# Oracle DB 연결 설정
oracle_connection = cx_Oracle.connect(
    user="team1",         # Oracle 사용자 이름
    password="team1",     # 비밀번호
    dsn="192.168.0.42:1521/xe"  # Oracle DSN 정보
)

# 커서 생성
cursor = oracle_connection.cursor()

# # 테이블 생성 쿼리 (테스트용 테이블 생성)
# create_table_query = """
#     CREATE TABLE DAEJEON_ENERGY (
#         PK_CD NUMBER,
#         LOTNO_ADDR VARCHAR2(4000),
#         ROAD_NM_ADDR VARCHAR2(4000),
#         SGNG_CD VARCHAR2(4000),
#         STDG_CD VARCHAR2(4000),
#         LOTNO_MNO VARCHAR2(4000),
#         LOTNO_SNO VARCHAR2(4000),
#         GPS_LOT VARCHAR2(4000),
#         GPS_LAT VARCHAR2(4000),
#         STNDD_YR VARCHAR2(4000),
#         USE_MM VARCHAR2(4000),
#         ELRW_USQNT VARCHAR2(4000),
#         CTY_GAS_USQNT VARCHAR2(4000),
#         SUM_NRG_USQNT VARCHAR2(4000),
#         ELRW_TOE_USQNT VARCHAR2(4000),
#         CTY_GAS_TOE_USQNT VARCHAR2(4000),
#         SUM_NRG_TOE_USQNT VARCHAR2(4000),
#         ELRW_GRGS_DSAMT VARCHAR2(4000),
#         CTY_GAS_GRGS_DSAMT VARCHAR2(4000),
#         SUM_GRGS_DSAMT VARCHAR2(4000)
#     )
# """
#
# # 테이블 생성
# try:
#     cursor.execute("DROP TABLE test")
# except cx_Oracle.DatabaseError:
#     pass  # 테이블이 없으면 넘어감

# cursor.execute(create_table_query)

# CSV 파일 경로 및 데이터 로드
file_path = '건물에너지DB_좌표매칭_최종(15-18)(대전).csv'
data = pd.read_csv(file_path, header=None, encoding='utf-8', dtype=str, low_memory=False)

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
