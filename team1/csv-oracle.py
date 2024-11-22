import pandas as pd
import numpy as np
import json as js
import datetime
import csv
from tqdm.auto import tqdm
import cx_Oracle as cx
import logging
import datetime

# oracle 설정
host_name = 'l92.168.0.42'
oracle_port = 1521
service_name = 'xe'

dsn = cx.makedsn(host_name, oracle_port)
conn = cx.connect('test1', 'test1')
cursor = conn.cursor()


# oracle db 연결하기
def ExecuteQuery(Query):
    try:
        dsn = cx.makedsn(host_name, oracle_port)
        conn = cx.connect('test1', 'test1')
        cursor = conn.cursor()

        cursor.execute(Query)
        cursor.close()
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logging.error(RuntimeError(f'[ERROR][ExecuteQuery] : {Query} {e}'))
        return False


# 데이터 전처리
def data_reverse(index, row):
    data_json = {
        "ex1": str(datetime.datetime.now().strftime('%Y%m%d%H%M%S') + lpad(index, 6, '0')),
        "ex2": row['ex2'],
        "FIRST_NAME": str(row['First Name']).replace('"', '').replace("'", " "),
        "LAST_NAME": str(row['Last Name']).replace('"', '').replace("'", " "),
        "EMAIL": str(row['Email Address']).replace('"', '').replace("'", " "),
        "ACCOUNT": str(nvl(row.Company, "N/A")).replace('"', '').replace("'", " "),
        "CREATED_DATE": row['Date Created'],
        "UPDATED_DATE": row['Date Modified']
    }
    return data_json


def lpad(i, width, fillchar='0'):
    return str(i).rjust(width, fillchar);


def nvl(value, defaultValue):
    return defaultValue if value == "" else value


print("start")

# csv 파일 읽기 및 그 전의 str 변경
load = pd.read_csv("건물에너지DB_좌표매칭_최종(19-22).csv", encoding='UTF8', dtype={"ex1": str, "ex2": str}, low_memory=False)
load = load.fillna('')

# iterrows -> 데이터의 행-열/데이터 정보를 튜플 형태의 generator 객체로 반환하는 매서드
# tqdm ->  python 에서 progreess bar을 통하여 알 수 있음 (Library 설치 필요!)
# 사용 방법은 본문 참고, 기본적으로 순회가능한 객체(리스트, 튜플, 이터레이터등), for문에 삽입

for index, row in tqdm(load.iterrows()):
    try:
        # json 형식 이용 -> row 정리
        data = data_reverse(index, row)

        # 다시 dataframe으로 변환
        dfitem = pd.json_normalize(data)

        # type 맞춰주기 -> str & 날짜
        dfitem = dfitem.astype({'x1': 'string'})
        dfitem['ex2'] = dfitem['ex2'].astype(str)

        # insert 할  table명 입력
        tbl_name = 'indegration_VER2'

        # dataframe을 numpy() -> tuple로 고정 -> list로 감싸기
        data_to_insert = [tuple(x) for x in dfitem.to_numpy()]

        # 컬럼명 지정
        db_columns = ['ex1', 'ex2', 'FIRST_NAME', 'LAST_NAME', 'EMAIL', 'ACCOUNT', 'CREATED_DATE', 'UPDATED_DATE']


        # 한줄씩 insert 함수
        def insert_data_by_query(tbl_name, data_to_insert):
            seperator = ","

            try:
                # insert 쿼리 생성 및 실행
                insert_query = f' INSERT INTO indegration_VER2 ({seperator.join(db_columns)}) VALUES {seperator.join(map(str, data_to_insert))}'  # str(lst)[1:-1]

                # print(seperator.join(map(str,data_to_insert))) -> list를 tuple로 변환

                # insert query 부분에서 형이 맞지 않음 -> 문제 ([('')]) 형식으로 value값 인식 불가
                insert_result = ExecuteQuery(insert_query)

                if (insert_result == False):
                    print(f''' [PROCESS] insert fail {insert_query} ''')

            except Exception as ex:
                print(f''' [PROCESS] insert error [data_to_insert]''')


        insert_data_by_query(tbl_name, data_to_insert)

    except Exception as e:
        print(e)


