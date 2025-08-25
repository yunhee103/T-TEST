# 공분산/상관계수 확인
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')
from pandas.plotting import scatter_matrix

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')
print(data.head(3))
print(data.describe())
print()
print(np.std(data.친밀도)) #0.968505126935272
print(np.std(data.적절성)) #0.8580277077642035
print(np.std(data.만족도)) #0.8271724742228969
# plt.hist([np.std(data.친밀도),np.std(data.적절성),np.std(data.만족도)])
# plt.show()
# plt.close()

# 공분산 -상관 계수 연습 
print('공분산')
print(np.cov(data.친밀도,data.적절성))  # numpy는 두개씩만 확인 가능
print(np.cov(data.친밀도,data.만족도))
print(data.cov())  #DataFrame 으로 공분산 출력


print('상관계수')
print(np.corrcoef(data.친밀도,data.적절성))  # numpy는 두개씩만 확인 가능
print(np.corrcoef(data.친밀도,data.만족도))
print(data.corr())  # Pandas를 활용한 상관계수
print(data.corr(method='pearson'))  # pearson 기본 : 변수가 등간, 비율 척도일 때 
print(data.corr(method='spearman'))  # 스피어만 : 변수가 서열 척도일 때
print(data.corr(method='kendall'))  # 켄달 : 스피어만과 유사함
"""
상관관계 분석 시에는 먼저 데이터의 척도를 파악한 후, 그에 맞는 상관계수 계산 방법을 선택하는 것이 중요. 
연속형 데이터의 선형 관계를 보려면 피어슨, 순위나 비정규 분포 데이터를 다룰 때는 스피어만이나 켄달을 사용
"""
# 예) 만족도 대한 다른 특성(변수) 사이의 상관관계 보기
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))
# 만족도    1.000000
# 적절성    0.766853
# 친밀도    0.467145

# 시각화
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()
attr = ['친밀도', '적절성', '만족도']
scatter_matrix(data[attr], figsize=(10,6))  #산점도, 히스토그램
plt.show()

import seaborn as sns
sns.heatmap(data.corr())
plt.show()