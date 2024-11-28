import pandas as pd
import cx_Oracle
from tqdm import tqdm

# Oracle DB 연결 설정
oracle_connection = cx_Oracle.connect(
    user="team1",
    password="team1",
    dsn="192.168.0.42:1521/xe"
)

# 커서 생성
cursor = oracle_connection.cursor()

# CSV 파일 경로
file_path = './대전광역시_중구_건물용도.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='utf-8', low_memory=False)

# 필요한 열만 추출
columns_to_extract = ['법정동코드','법정동명','지번', '주요용도코드', '주요용도명', '건물대지면적', '건물건축면적',
                        '건물용도분류코드', '건물용도분류명', '건물높이', '지상층수', '지하층수']
filtered_data = data[columns_to_extract]

# 컬럼 이름 수동 할당
filtered_data.columns = ['STDG_CD','STDG_NM', 'LOTNO_NO', 'MJR_PRPS_CD', 'MJR_PRPS_NM', 'BLDG_ST_AR', 'BLDG_ARCH_AREA', 'BLDG_USG_CLSF_CD',
                         'BLDG_USG_CLSF_NM', 'BLDG_HGT', 'GRND_NOFL', 'UDGD_NOFL']

# 모든 데이터를 문자열로 변환 및 NaN 값 처리
filtered_data = filtered_data.fillna('').astype(str)

# SQL UPDATE 문
insert_query = """
    INSERT INTO BUILDING_USAGE (

        "STDG_CD", "STDG_NM", "LOTNO_NO", "MJR_PRPS_CD", "MJR_PRPS_NM", 
        "BLDG_ST_AR", "BLDG_ARCH_AREA", "BLDG_USG_CLSF_CD", "BLDG_USG_CLSF_NM", 
        "BLDG_HGT", "GRND_NOFL", "UDGD_NOFL"
        
    ) VALUES (
        :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12
    )
"""

# 데이터프레임 데이터를 테이블에 삽입
try:
    for index, row in filtered_data.iterrows():
        try:
            cursor.execute(insert_query, (
                row['STDG_CD'],
                row['STDG_NM'],
                row['LOTNO_NO'],
                row['MJR_PRPS_CD'],
                row['MJR_PRPS_NM'],
                row['BLDG_ST_AR'],
                row['BLDG_ARCH_AREA'],
                row['BLDG_USG_CLSF_CD'],
                row['BLDG_USG_CLSF_NM'],
                row['BLDG_HGT'],
                row['GRND_NOFL'],
                row['UDGD_NOFL']
            ))
        except Exception as row_error:
            print(f"오류 발생 (행 {index}): {row_error}")
            print(f"문제 데이터: {row.to_dict()}")

    # 커밋
    oracle_connection.commit()
    print(f"{len(filtered_data)}개의 행이 성공적으로 삽입되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
    oracle_connection.rollback()
finally:
    # 커서 및 연결 종료
    cursor.close()
    oracle_connection.close()