# 추론 통계 분석 중 비율검정 
# - 비율검정특징
# : 집단의 비율이 어떤 특정한 값과 같은지를 검증.
# : 비율 차이 검정 통계량을 바탕으로 귀무가설의 기각여부를 결정
# one-sample
# A회사에는 100명 중에 45명이 흡연을 한다.국가 통계를 보니 국민 흡연율은 35%라고한다.
# 비율이같냐?
# 귀무 :A회사 직원들의 흡연율과 국민 흡연율의 비율이 같다.
# 대립 :A회사 직원들의 흡연율과 국민 흡연율의 비율이 같지 않다.

import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count = np.array([45])
nobs = np.array([100])
val = 0.35

z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)         #[2.01007563]
print(p)         #[0.04442318]   < 0.05 귀무 기각 : 비율이 다르다.

# two-sample
# A회사 사람들 300명 중 100명이 커피를마시고, B회사 사람들 400명 중 170명이 커피를 마셨다.
# 비율이같냐?

count = np.array([100, 170])
nobs = np.array([300, 400])

z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)         # -11.88004694268173
print(p)         # 1.5028294294082938e-32  < 0.05 귀무 기각 : 비율이 다르다.


print('---------이항검정-------------')

# 결과가 두 가지 값을 가지는 확률 변수의 분포를 판단 하는데 효과적
# 예) 10명 중 자격증 시험 합격자 중 여성 6명이었다고 할 때 '여성이 남성보다 합격률이 높다.'라고 할 수 있는가?

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
print(data.head(3))

ctab = pd.crosstab(index=data['survey'], columns='count')
ctab.index = ['불만족', '만족']
print(ctab)

# 귀무 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80% 이다. 
# 대립 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80% 아니다. 

#  양측 검정 : 방향성이 없다.  n=150번 시행 중 성공 136
result = stats.binomtest(k=136, n= 150, p=0.8, alternative='two-sided')
print(result)

# BinomTestResult(k=136, n=150, alternative='two-sided', statistic=0.9066666666666666, pvalue=0.0006734701362867024)  < 0.05 귀무 기각 

# 단측 검정 : 방향성이 있다. (80% 보다 크다 라고 가정하고 검증)
result = stats.binomtest(k=136, n= 150, p=0.8, alternative='greater') # less 
print(result)
# BinomTestResult(k=136, n=150, alternative='greater', statistic=0.9066666666666666, pvalue=0.00031794019219854805) < 0.05 귀무 기각 
