import csv
import cx_Oracle
conn = cx_Oracle.connect("test1","test1","192.168.0.42:1521/xe")
print(conn)
cursor = conn.cursor()
print(cursor)

f = open('건물에너지DB_좌표매칭_최종(19-22).csv', 'r', encoding='utf-8')
# 오류 메세지: UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 233: illegal multibyte sequence
# 인코딩 문제로 encoding='utf-8' 추가
# 2024.11.20 11:45 강민호

csvReader = csv.reader(f)

for row in csvReader:
    LOTNO_ADDR = (row[0])

    LOTNO_MNO = (row[4])

    LOTNO_SNO = (row[5])


    sql = """insert into lotno_addr (lotno_addr, lotno_mno, lotno_sno) values (:1, :2, :3)"""
    # 잘못된 변수명/번호 오류: SQL 구문에서 플레이스홀더와 실제 변수의 매핑이 올바르지 않을 때 발생
    # values((%s, %s, %s) 형식을 values (:1, :2, :3)으로 변경
    # 2024.11.20 11:51 강민호

    cursor.execute(sql, (LOTNO_ADDR, LOTNO_MNO, LOTNO_SNO))

# db의 변화 저장

conn.commit()

f.close()

conn.close()