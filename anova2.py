#일원분산분석 연습
# 강남구에 있는 gs편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있는가?

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols  #기울기 직선 = 회기분석
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"
# data = pd.read_csv(url, header=None)
# print(data)
data = np.genfromtxt(url,delimiter=',')
print(data, type(data), data.shape)

# 3개 집단의 월급, 평균 얻기
gr1 = data[data[:,1]==1, 0]
gr2 = data[data[:,1]==2, 0]
gr3 = data[data[:,1]==3, 0]

print(gr1, '', np.mean(gr1))
print(gr2, '', np.mean(gr2))
print(gr3, '', np.mean(gr3))

# 정규성
print(stats.shapiro(gr1).pvalue)
print(stats.shapiro(gr2).pvalue)
print(stats.shapiro(gr3).pvalue)
# 등분산성
print(stats.levene(gr1, gr2, gr3).pvalue)
print(stats.bartlett(gr1, gr2, gr3).pvalue)

plt.boxplot([gr1, gr2, gr3], showmeans=True)
plt.show()
plt.close()

# anova 검정 방법1: anova_lm
df = pd.DataFrame(data, columns=['pay', 'group'])
print(df)
lmodel = ols('pay~C(group)', data=df).fit()
print(anova_lm(lmodel, type=2))
# anova 검정 방법2: f_oneway
f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3)
print('f_statistic :', f_statistic)
print('p_value : ', p_value)

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(turkyResult)

turkyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()