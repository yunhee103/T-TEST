import pandas as pd
import mysql.connector    # pip install mysql-connector-python 설치
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


# MariaDB 연결 정보 설정
config = {
    'user': 'root',
    'password': '1234',
    'host': '127.0.0.1',
    'database': 'mydb'
}

try:
    conn = mysql.connector.connect(**config) # conn 변수에 MariaDB에 연결된 객체를 할당
    cursor = conn.cursor(dictionary=True)    # cursor 변수에 SQL 쿼리를 실행하고 결과를 가져올 커서 객체를 dictionary로 할당
    
    query = "SELECT jikwonjik, jikwonpay FROM jikwon"    # SQL 쿼리를 정의
    cursor.execute(query)                                # cursor를 이용해 SQL 쿼리를 데이터베이스에 보냄
    
    df = pd.DataFrame(cursor.fetchall())  # 쿼리 결과로 나온 모든 데이터를 가져와 pandas DataFrame으로 만듬
    print(" 데이터베이스 연결 성공 및 데이터 로드 완료")

except mysql.connector.Error as e:
    print(f" 데이터베이스 연결 또는 쿼리 실행 실패: {e}")
    exit()

# try-except-else 문의 성공/실패 여부와 관계없이 항상 실행
finally:
    if cursor is not None:
        cursor.close()
    if conn is not None and conn.is_connected():
        conn.close()

if df is None:
    print("오류: 데이터를 불러오지 못했습니다. 프로그램을 종료합니다.")
    exit()

# NA가 있는 행은 pd.crosstab에서 자동으로 제외되므로, 명시적으로 dropna를 사용하지 않음


# jikwon_pay의 범주를 지정 조건
#  jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#  jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)

# 연봉을 범주화하는 bins와 labels 정의
bins = [1000, 3000, 5000, 7000, df['jikwonpay'].max() + 1]
labels = ["1000~2999", "3000~4999", "5000~6999", "7000~"]

# pd.cut 함수를 사용하여 연봉을 범주화하고, 교차표 생성

ctab = pd.crosstab(
    index=df['jikwonjik'],
    columns=pd.cut(df['jikwonpay'], 
            bins=bins, labels=labels, right=False)
)
# 교차표 인덱스 가독성 향상
ctab.index = ["이사", "부장", "과장", "대리", "사원"]

print("\n[직급과 연봉 범주의 교차표]")
print(ctab)

# 카이제곱 검정
chi2, p, dof, _ = stats.chi2_contingency(ctab)

print("\n직급과 연봉 범주의 교차표:")

# 가설 설정
# 귀무가설(H0): 직급과 연봉은 서로 관련이 없다.
# 대립가설(H1): 직급과 연봉은 서로 관련이 있다.

# 카이제곱 검정
chi2, p_value, dof, expected = stats.chi2_contingency(ctab)

chi2, p , dof, _ = stats.chi2_contingency(ctab)
msg = "test statics:{}, p-value:{}, df:{}"
format(msg.format(chi2, p , dof))
print(msg.format(chi2,p,dof))
#  test statics:120.0, p-value:0.04917667372448821, df:96
"""
결론 : p-value(0.0491)  < 0.05 이므로 귀무가설 기각.
직급과 연봉은 서로 관련이 있다.
"""

# 추가 그래프 
plt.rc('font', family='malgun gothic')
sns.heatmap(ctab, annot=True, fmt='d', cmap='Reds')   
# data-> ctab는 숫자형 데이터(빈도수)만 포함하고 있어 heatmap이 정상적으로 작동되게함.
plt.title('직급과 연봉 관계 확인 ')
plt.xlabel('연봉 범위')
plt.ylabel('직급')
plt.show()