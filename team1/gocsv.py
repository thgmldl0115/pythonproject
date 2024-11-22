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

# 데이터 개수 확인
row_count = len(filtered_data)
# 대전 전체 데이터 갯수 : 1,199,696 개,  개로 출력된 코드 개수와 비교 필요
# 추출된 데이터 개수(19-22): 1,305,175개
# 추출된 데이터 개수(15-18): 1,710,373개
print(f"추출된 데이터 개수: {row_count}개")


# 결과를 새로운 CSV 파일로 저장
filtered_data.to_csv('건물에너지DB_좌표매칭_최종(15-18)(대전).csv', index=False, encoding='utf-8-sig')