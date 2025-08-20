#  비(눈) 여부 ( 두 개의 집단)에 따른 매출액의 평균차이 검정
#  공통 칼람이 년월일인 두개의 파일을 조합을 해서 작업
#  귀무 : 강수량에 따른 음식점 매출액 평균 차이는 없다.
#  대립 : 강수량에 따른 음식점 매출액 평균 차이는 있다.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

#  매출 자료 일긱
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(3)) # 328 entries, 0 to 327
print(sales_data.info())

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(3)) #  702 entries,  9 columns
print(wt_data.info())

# sales 데이터의 날짜를 기준으로 두개의 자료를 병합 작업 진행
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-',''))
print(wt_data.head(3))
frame = sales_data.merge(wt_data, how='left',left_on='YMD',right_on='tm')
print(frame.head(3), '', len(frame))
print(frame.columns) #(['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes']
data = frame.iloc[:,[0,1,7,8]]
print(data.head(3))
print(data.isnull().sum())  # 결측치 확인

print('강수 여부에 따른 매출액 평균 차이가 유의미한지 확인하기----------')
# data['rain_yn'] = (data['sumRn'] > 0).astype(int)            #칼람 만들기  astype숫자화 비안옴 true=0, 비옴 false=1
data['rain_yn'] = (data.loc[:,('sumRn')] > 0 ) * 1              # 같은 뜻
print(data.head(6))

sp = np.array(data.iloc[:, [1,4]]) #AMT, rain_yn
tg1 = sp[sp[:,1]==0,0] #집단 1  : 비안올때 매출액을 가짐
tg2 = sp[sp[:,1]==1,0] #집단 2  : 비올때 매출액을 가짐
print('tg1', tg1[:3])
print('tg2', tg2[:3])

plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()

print('두 집단 평균 :', np.mean(tg1), 'vs', np.mean(tg2))
#  두 집단 평균 : 761040.2542372881 vs 757331.5217391305
#  정규성 검정
print(len(tg1), '', len(tg2))
print('tg1_pvalue : ', stats.shapiro(tg1).pvalue)
print('tg2_pvalue : ', stats.shapiro(tg2).pvalue)
# tg1_pvalue :  0.056050644029515644 > 0.05 정규성 만족
# tg2_pvalue :  0.8827503155277691 > 0.05 정규성 만족

#  등분산성 검정 -  등분산성 검정은 두 개 이상의 집단이 같은 분산을 가지고 있는지 통계적으로 확인하는 절차
print('등분산성:', stats.levene(tg1,tg2).pvalue) 
#  등분산성: 0.7123452333011173  > 0.05 만족


print(stats.ttest_ind(tg1,tg2, equal_var=True))
#  TtestResult(statistic=np.float64(0.10109828602924716), pvalue=np.float64(0.919534587722196), df=np.float64(326.0))
#  pvalue : 0.9195345>0.05 이므로 귀무가설 채택
#  강수 여부에 따른 매출액 평균은 차이가 없다.
"""
[two-sample t 검정 : 문제1] 
다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
""" 
blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

#  귀무 : 포장지 색상에 따른 제품의 매출액에 차이는 없다.
#  대립 : 포장지 색상에 따른 제품의 매출액에 차이는 있다.
print('두 집단 평균 :', np.mean(blue), 'vs', np.mean(red))
print(len(blue), '', len(red))
# 샤피로-윌크(Shapiro-Wilk) 검정을 통해 정규성을 확인  
print('blue_pvalue : ', stats.shapiro(blue).pvalue)  
print('red_pvalue : ', stats.shapiro(red).pvalue) # blue_pvalue :  0.5102310078114559  red_pvalue :  0.5347933246260025  > 0.05 만족   / 정규분포를 따른다고 볼 수 있음
print(stats.ttest_ind(blue,red))  
#TtestResult(statistic=np.float64(2.9280203225212174), pvalue=np.float64(0.008316545714784402), df=np.float64(20.0))
# 0.00831654 <  0.05 이므로 귀무가설 기각
# 포장지 색상에 따른 제품의 매출액에 차이는 있다.

"""
[two-sample t 검정 : 문제2]  
아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.

  남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
  여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4
"""
print()

import random
from numpy import random
male = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

male_sample = np.random.choice(male, 15, replace=False) # 비복원 추출
female_sample = np.random.choice(female, 15, replace=False)
# print(male_sample)
print("무작위 남자 평균:" , np.mean(male_sample))
# print(female_sample)
print("무작위 여자 평균:" , np.mean(female_sample))
# print(len(male_sample), '', len(female_sample))
# 정규성 검사
print('male_sample_pvalue : ', stats.shapiro(male_sample).pvalue)   # 남자 표본 정규성 p-value
print('female_sample_pvalue : ', stats.shapiro(female_sample).pvalue)  # 여자 표본 정규성 p-value

# 등분산성 검정 (Levene)
levene_pvalue = stats.levene(male_sample, female_sample).pvalue
print(f"Levene p-value: {levene_pvalue:.4f}")

# Levene 검정 결과에 따라 t-검정 옵션 선택
alpha = 0.05
if levene_pvalue < alpha:
    # p-value가 작으면 분산이 다르다고 판단
    equal_variances = False
    print("등분산성 가정 기각: 이분산 t-검정(Welch's t-test)을 사용합니다.")
else:
    # p-value가 크면 분산이 같다고 판단
    equal_variances = True
    print("등분산성 가정 채택: 등분산 t-검정을 사용합니다.")

# t-검정 수행
ttest_result = stats.ttest_ind(male_sample, female_sample, equal_var=equal_variances)

# 최종 결과 출력
print(f"t-통계량: {ttest_result.statistic:.4f}")
print(f"t-검정 p-value: {ttest_result.pvalue:.4f}")

# p-value를 이용한 최종 결론
if ttest_result.pvalue < alpha:
    print("결론: 귀무가설 기각 → 남녀 콜레스테롤 양에 유의미한 차이가 있습니다.")
else:
    print("결론: 귀무가설 채택 → 남녀 콜레스테롤 양에 유의미한 차이가 없습니다.")



""" 
[two-sample t 검정 : 문제3]
DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.
"""
import mysql.connector  # pip install mysql-connector-python 설치

# MariaDB 연결 정보 설정
config = {
    'user': 'root',
    'password': '1234',
    'host': '127.0.0.1',
    'database': 'mydb'
}

# 데이터베이스 연결
try:
    conn = mysql.connector.connect(**config)
    
    # SQL 쿼리를 이용해 'jikwon'과 'buser' 테이블을 결합하여 총무부, 영업부 직원의 연봉을 가져옴
    sql = """
        SELECT b.busername, j.jikwonpay
        FROM jikwon AS j
        JOIN buser AS b ON j.busernum = b.buserno
        WHERE b.busername IN ('총무부', '영업부');
    """

    # 쿼리 결과를 DataFrame으로 로드
    df = pd.read_sql(sql, conn)
    
    print("데이터베이스 연결 성공 및 데이터 로드 완료")
    print(df.head()) # 데이터 확인용

except mysql.connector.Error as e:
    print(f"데이터베이스 연결 또는 쿼리 실행 실패: {e}")
    exit()

finally:
    if conn and conn.is_connected():
        conn.close()

# 연봉이 없는 값(NaN)을 해당 부서의 평균으로 채우기
df['jikwonpay'] = df.groupby('busername')['jikwonpay'].transform(lambda x: x.fillna(x.mean()))

# 총무부와 영업부의 연봉 데이터 추출
pay_chongmu = df[df['busername'] == '총무부']['jikwonpay']
pay_sales = df[df['busername'] == '영업부']['jikwonpay']

print("\n--- 분석 데이터 ---")
print("총무부 직원 연봉:\n", pay_chongmu.tolist())
print("총무부 직원 연봉 평균:", pay_chongmu.mean())
print("영업부 직원 연봉:\n", pay_sales.tolist())
print("영업부 직원 연봉 평균:", pay_sales.mean())

# 등분산성 검정 (Levene's test)
# H0: 두 그룹의 분산은 같다.
levene_pvalue = stats.levene(pay_chongmu, pay_sales).pvalue

alpha = 0.05
if levene_pvalue < alpha:
    equal_variances = False
    print("\nLevene 검정 p-value: {:.4f} (< 0.05). 등분산성 가정을 기각 -> 이분산 t-검정을 수행.".format(levene_pvalue))
else:
    equal_variances = True
    print("\nLevene 검정 p-value: {:.4f} (>= 0.05). 등분산성 가정을 채택 -> 등분산 t-검정을 수행.".format(levene_pvalue))

# 두 독립 표본 t-검정
# H0: 두 부서의 평균 연봉은 같다.
t_stat, p_val = stats.ttest_ind(pay_chongmu, pay_sales, equal_var=equal_variances)

# 최종 결과 출력 및 해석
print("\n--- t-검정 결과 ---")
print("t-통계량: {:.4f}".format(t_stat))
print("p-value: {:.4f}".format(p_val))

if p_val < alpha:
    print("\n결론: p-value가 0.05보다 작으므로 귀무가설을 기각합니다.")
    print("통계적으로 총무부와 영업부의 평균 연봉에는 유의미한 차이가 존재합니다.")
else:
    print("\n결론: p-value가 0.05보다 크므로 귀무가설을 채택합니다.")
    print("통계적으로 총무부와 영업부의 평균 연봉에는 유의미한 차이가 존재하지 않습니다.")