#  ols가 제공하는 표에 대해 알아보기
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')

print(df.head(3))
print(df.corr())

print('회귀분석 수행-------------------')
import statsmodels.formula.api as smf
model = smf.ols(formula='만족도~ 적절성', data=df).fit()
print(model.summary())
print('parameters :', model.params)  #회귀계수
print('R squared :', model.rsquared)  #결정계수
print('p value :', model.pvalues)  
# print('predicted value :', model.predict())  
print('실제값 : ', df.만족도[0], '\n예측값 : ', model.predict()[0])

import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='Malgun Gothic')

plt.scatter(df.적절성, df.만족도)
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)
plt.plot(df.적절성, df.적절성 * slope + intercept, 'b')
plt.show()



