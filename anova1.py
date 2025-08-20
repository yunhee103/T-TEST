# 분산분석(ANOVA, Analysis of Variance)은 여러 집단의 평균을 비교할 때 사용하는 통계적 방법. 
# 특히 세 집단 이상의 평균을 비교할 때 유용하며, 단순한 평균 비교를 반복하는 것보다 오류 가능성을 줄여줍니다.
# 두 집단의 평균 비교는 t-검정으로 충분하지만,세 집단 이상을 비교할 때 t-검정을 반복하면 제1종 오류(잘못된 귀무가설 기각)가 누적되어 신뢰도가 떨어집니다.
# 이를 해결하기 위해 Fisher가 개발한 방법이 바로 ANOVA입니다.
# 집단 간 분산 (요인에 의한 분산) → 서로 다른 집단 간 평균 차이에서 발생하는 분산
# 집단 내 분산 (오차에 의한 분산) → 동일 집단 내 개별 값들의 차이에서 발생하는 분산
# 이 두 분산을 비교하여, 집단 간 분산이 통계적으로 유의미한지를 검정합니다.
# F값 = 집단 간 분산 / 집단 내 분산
# F값이 크면 집단 간 차이가 크다는 뜻 -> 귀무가설(집단 간 평균 차이가 없다)을 기각할 수 있음  
# F값이 작으면 귀무가설을 기각할 수 없음

""" 
* 서로 독립인 세 집단의 평균 차이검정
실습) 세 가지 교육방법을 적용하여 1개월 동안 교육 받은 교육생 80명을 대상으로 실기시험을 실시.   three_sample.csv
독립변수 : 교육방법 (세가지 방법), 종속변수 : 시험점수
일원분산분석(oneway)
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols  #기울기 직선 = 회기분석
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv')
print(data.head(3))
print(data.shape)
print(data.describe())

# 이상치를 차트로 확인 
# plt.hist(data.score)
# plt.boxplot(data.score)
# plt.show()
# plt.close()

# 이상치 제거
data = data.query('score <= 100')
print(len(data))

result = data[['method', 'score']]
print(result)
m1 = result[result['method'] == 1 ]
m2 = result[result['method'] == 2 ]
m3 = result[result['method'] == 3 ]
print(m1[:3])
print(m2[:3])
print(m3[:3])
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']

#  정규성
print('score1 : ', stats.shapiro(score1).pvalue)
print('score2 : ', stats.shapiro(score2).pvalue)
print('score3 : ', stats.shapiro(score3).pvalue)

"""
score1 :  0.17467355591727662
score2 :  0.3319001150712364
score3 :  0.11558564512681252

0.05보다 크므로 정규성 만족
"""
print(stats.ks_2samp(score1,score2))  # 두 표본이 동일한 분포를 따르는지 확인하는 **콜모고로프-스미르노프 검정(Kolmogorov-Smirnov test)**을 수행하는 함수
# 등분산성(복수 집단 분산의 치우침 정도)
print('levene:',stats.levene(score1, score2, score3).pvalue)
print('fligner:',stats.fligner(score1, score2, score3).pvalue)
print('bartlett:',stats.bartlett(score1, score2, score3).pvalue)

# 교차표 등 작성 가능..

import statsmodels.api as sm
reg = ols("data['score'] ~ C(data['method'])", data=data).fit() #단일회귀모델 작성
# 분산 분석표를 이용해 분산 결과 작성 
table = sm.stats.anova_lm(reg, type=2)
print(table)

# 사후검정(post hoc test)
# 분산분석은 집단의 평균에 차이 여부만 알려 줄 뿐 각 집단 간의 평균 차이는 알려주지 않는다.
# 각 집단 간의 평균 차이를 확인하기 위해 사후검정 실시

from statsmodels.stats.multicomp import pairwise_tukeyhsd
turResult = pairwise_tukeyhsd(endog=data.score, groups=data.method)
print(turResult)
turResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()

"""
터키 사후 검정에서 두 그룹의 신뢰 구간이 겹치면 그 차이는 통계적으로 유의미하지 않습니다.
제공된 그래프에서는 그룹 1, 2, 3의 신뢰 구간이 모두 겹칩니다. 
(예: 그룹 1의 신뢰 구간(62~72.5)과 그룹 2의 신뢰 구간(63~73)은 겹치는 부분이 매우 넓습니다.)     
따라서, 세 그룹의 평균 사이에는 통계적으로 유의미한 차이가 없다고 결론 내릴 수 있습니다.

"""