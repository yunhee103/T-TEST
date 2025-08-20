# 독립 표본 검정 : 두 집단의 평균의 차이 검정
# 서로 다른 두 집단의 평균에 대한 통계 검정에 주로 사용된다.
# 비교를 위해 평균과 표준 편차 통계량을 사용한다.
# 평균값의 차이가 얼마인지, 표준편차는 얼마나 다른지 확인해 분석 대상인 두 자료가 같을 가능성이 우연의 범위 5%에 들어가는지를 판별한다.
# 결국 t-test (독립표본 기본) 는 두 집단의 평균과 표준편차 비율에 대한 대조검정법이다.
# t-value는 두 집단의 차이를 불확실성으로 나눈 비율 t- value  q-value 반비례   t가 커져야함 = 두 집단간 차이가 커야함
"""
실습1) 남녀 두 집단간 파이썬 시험의 평균 차이 검정

"""
# 귀무 : 남녀 두 집단간 파이썬 평균 차이는 없다.
# 대립 : 남녀 두 집단간 파이썬 평균 차이는 있다.
# 95% 신뢰 수준에서 우연히 발생할 확률이 5% 보다 작냐? 그렇다 -> 귀무기각 
# 선행 조건 : 두 집단의 자료는 정규 분포를 따른다. 분산이 동일하다(등분산성) 
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
from scipy import stats
import pandas as pd
import numpy as np

print(np.mean(male), ' ' , np.mean(female)) # 83.8   72.24
two_sample = stats.ttest_ind(male, female)  #t-test independent
two_sample = stats.ttest_ind(male, female, equal_var=True)  # equal_var 분산 기본값
print(two_sample)  #tatistic=np.float64(1.233193127514512), pvalue=np.float64(0.2525076844853278), df=np.float64(8.0))
#  pvalue= 0.2525076 > 0.05 귀무 채택

print('등분산 검정 ------')
"""  
- bartlett : scipy.stats.bartlett

- fligner : scipy.stats.fligner

- levene : scipy.stats.levene
"""
from scipy.stats import levene
levene_stat, levene_p = levene(male, female)
print(f"통계량 : {levene_stat:.4f}, p-value:{levene_p:4f}")
if levene_p > 0.05:
    print('분산이 같다고 할 수 있다.')
else:
    print('분산이 같다고 할 수 없다. 등분산 가정이 부적절하다')

# 등분산성 가정이 부적절한 경우 welch's t-test사용을 권장

welch_result = stats.ttest_ind(male, female, equal_var=False)
print(welch_result) 
#TtestResult(statistic=np.float64(1.233193127514512), pvalue=np.float64(0.2595335362303284), df=np.float64(6.613033864755501))

