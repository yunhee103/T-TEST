# 집단간차이분석: 평균 또는 비율 차이를 분석: 
# 모집단에서 추출한 표본 정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.
#  * T-test와 ANOVA의 차이
# - 두 집단 이하의 변수에 대한 평균차이를 검정할 경우 T-test를 사용하여 검정통계량 T값을 구해 가설 검정을 한다.
# - 세 집단이상의 변수에 대한 평균 차이를 검정할 경우에는 ANOVA를 이용하여 검정통계량 F값을 구해 가설검정을 한다.

# 핵심 아이디어:
# 집단 평균 차이(분자)와 집단 내 변동성(표준오차, 표준편차 등 분모)을 비교하여, 차이가 데이터의 불확실성(변동성)에 비해 얼마나 큰지를 계산한다.

# t 분포는 표본평균을 잉용해 정규분포의 평균을 해석할 때 많이 사용한다.
# 대개의 경우 표본의 크기는 30개 이하일 때 t분포를 따른다.




# t 검정은 두개 이하 집단의 평균의 차이가 우연에 의한것인지통계적으로 유의한 차이를 판단하는 통계적 절차다.

# 단일 모집단의 평균에 대한 가설정검(one samples t-test)
# 어느 남성 집단의 평균 키 검정
# 귀무 : 집단의 평균 키가 177이다. (모수)
# 대립 : 집단의 평균 키가 177이 아니다.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
"""
one_sample = [167.0, 182.7, 160.6, 176.8,185.0]
print(np.array(one_sample).mean()) 

result = stats.ttest_1samp(one_sample, popmean=199)
print('statistic:%5.f, pvalue:%.5f'%result)
# pvalue:0.60847 > 0.05 이므로 귀무가설 채택
# plt.boxplot(one_sample)
"""
"""plt.boxspolt(one_sampele, bins=10, kde=True, color='blue')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
plt.xlabel('data')
plt.ylabel('count')
plt.show()
plt.close()"""
"""


# 실습예제1)  A중학교 1학년 1반 학생들의 시험결과가 담긴파일을 읽어 처리(국어점수평균검정) 
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
print(data.head(3))
print(data.describe())
# 정규성 검정 : one-sample t-test는 옵션 
print('정규성 검정 : ', stats.shapiro(data.국어))  #  pvalue=np.float64(0.01295975332132026)) < 0.05 보다 작으므로 정규성을 만족 못함
#  정규성 위배는 데이터 재가공 추천, wilcoxon signed-rank test를 써야 더 안전
#  wilcoxon signed-rank test는 정규성을 가정하지 않음
from scipy.stats import wilcoxon 
wilcox_res = wilcoxon(data.국어 - 80)  #평균 80과 비교
print('wilcox_res : ' , wilcox_res)
# pvalue=np.float64(0.39777620658898905) > 0.05이므로 귀무가설 채택

res = stats.ttest_1samp(data.국어, popmean=80)
print('statistic:%5.f, pvalue:%.5f'%res)  #statistic:   -1, pvalue:0.19856 > 0.05 이므로 귀무가설 채택

# 해석 : 정규성은 부족하지만 t-test와 wilcoxon은 같은 결과를 얻었다. 표본수가 커지면 결과는 달라질 수 있다.
# 정규성 위배가 있어도 t-test결과는 신뢰 할 수 있다..

# 실습예제2) 여아 신생아 몸무게의 평균 검정수행 babyboom.csv
# # 여아 신생아의 몸무게는 평균이2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# # 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해보자

# 귀무 :  여아 신생아의 몸무게는 평균이 2800(g)이다.
# 대립 :  여아 신생아의 몸무게는 평균이 2800(g)이 아니다.

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv')
print(data2.head(3))
print(data2.describe())
fdata = data2[data2.gender == 1]
print(fdata.head(2))

print(len(fdata))
print(np.mean(fdata.weight), '', np.std(fdata.weight))

# 3132.4444  vs 2800  
# 정규성 검정( 하나의 집단일 때는 option)

print(stats.shapiro(fdata.iloc[:,2]))  # p 0.0179 < 0.05 정규성 위배


#  정규성 시각화
#  1) histogram으로 확인
sns.displot(fdata.iloc[:,2], kde=True)
plt.show()
plt.close()

#  2) Q-Q PLOT으로 확인
stats.probplot(fdata.iloc[:,2], plot=plt)
plt.show()
plt.close()

print()
wilcox_resBaby = wilcoxon(fdata.weight - 2800)  #평균 2800과 비교
print('wilcox_resBaby : ' , wilcox_resBaby)  # WilcoxonResult(statistic=np.float64(37.0), pvalue=np.float64(0.03423309326171875))
# 0.034233 < 0.05 이므로 귀무 기각
print()

resBaby = stats.ttest_1samp(fdata.weight, popmean=2800)
print('statistic:%5.f, pvalue:%.5f'%resBaby)   # statistic:    2, pvalue:0.03927
# pvalue:0.03927 < 0.05이므로 귀무기각
# 즉, 여아 신생아의 평균 체중은 2800g보다 증가 하였다.



"""

# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
#    305 280 296 313 287 240 259 266 318 280 325 295 315 278

# 귀무가설(H0​): 새롭게 개발된 백열전구의 수명은 300시간과 같거나 작다.
# 대립가설(H1): 새롭게 개발된 백열전구의 수명은 300시간보다 크다.

one_sample1 = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print(np.array(one_sample1).mean()) 

result = stats.ttest_1samp(one_sample1, popmean=300)
print('statistic:%5.f, pvalue:%.5f'%result)
# pvalue:0.14361 > 0.05 이므로 귀무가설 채택


# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.

data3 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
# print(data3)
# print(data3.describe())
# # print(len(data3))
df = data3.dropna(subset=['time'])
df = df['time'].str.replace(r'\s+', ' ', regex=True)
# 1. replace()를 사용하여 문자열 내의 불필요한 공백을 하나로 만듦
# '\s+'는 정규 표현식으로, '하나 이상의 공백'을 의미합니다.
# ' '는 공백 하나를 의미합니다.
df = df.str.split(expand=True).stack().reset_index(drop=True)
df = pd.to_numeric(df)

"""
df['time'].str.split(expand=True): 코드는 time 열의 각 문자열을 공백을 기준으로 분리하여 여러 개의 새로운 열을 만듭니다. 
expand=True 옵션을 사용하면 분리된 값들이 DataFrame 형태로 확장됩니다.
.stack(): 이렇게 확장된 DataFrame을 stack()하면 여러 열에 걸쳐 있던 데이터가 하나의 Series로 재구성됩니다. 이 과정에서 결측값(NaN)은 자동으로 제거됩니다.
.reset_index(drop=True): stack()으로 인해 생성된 계층적 인덱스를 제거하고, 0부터 시작하는 새로운 인덱스를 만듭니다.
pd.to_numeric(df): 마지막으로, 이렇게 정리된 Series의 모든 요소들을 숫자형(float)으로 변환합니다.
"""

print(df)
print(f"데이터의 총 개수: {len(df)}")
print(f"평균: {np.mean(df):.4f}", f"표준편차: {np.std(df):.4f}")

print("\n--- 검정 결과 ---")

resT = stats.ttest_1samp(df, popmean=5.2)
print('statistic:%5.f, pvalue:%.5f'%resT)  
#  pvalue:0.00014 < 0.05 이므로 귀무 기각
#  결론 : 5.2 시간과 차이가 있다.


# [one-sample t 검정 : 문제3] 
# https://www.price.go.kr/tprice/portal/main/main.do 에서 
# 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
# 정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오. (월별)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon


data = pd.read_excel('testdata_1.xls')
df = data.drop(['번호', '품목', 'Unnamed: 2'],axis=1)
df = df.transpose()
df.rename(columns={0 :'금액'}, inplace=True)
df.dropna(inplace=True)
print(df.mean())
print('정규성 검정:',stats.shapiro(df.금액)) # pvalue=0.05814403680264911 0.05보다 크므로 정규성 만족

# wilcox_res = wilcoxon(df.금액 - 15000) # 평균 15000과 비교
# print('wilcox_res: ',wilcox_res) # pvalue=3.0517578125e-05 0.05보다 작으므로 귀무가설 기각

# 정규성을 만족하면 T-검정 을사용 그렇지 않으면 윌콕슨 검정 사용

res = stats.ttest_1samp(df.금액, popmean=5.15000) # pvalue:0.00000: 0.05보다 작으므로 귀무 가설 기각
print('statistic:%.5f,pvalue:%.5f'%res)

# 결론 전국 평균 미용 요금이 15000이 아니다.

sns.displot(df.금액, kde=True)
plt.show()
plt.close()
 