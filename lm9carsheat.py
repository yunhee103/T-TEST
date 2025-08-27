import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm # statsmodels.api는 sm으로 별칭을 지정했습니다.
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import linear_reset
import scipy.stats as stats

plt.rc('font', family='malgun gothic')

# 1. 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv')
print(df.head(2))
print(df.info())

# 2. 불필요한 열 제거
# 'df.columns[6]', 'df.columns[9]', 'df.columns[10]'은 각각 'Urban', 'US', 'ShelveLoc' 열을 의미합니다.
# 이 열들은 숫자형이 아니거나, 종속변수와 관련성이 적어 제거합니다.
df = df.drop(df.columns[[6, 9, 10]], axis=1)
# 3. 상관관계 분석
print(df.corr())

# 4. 다중 선형 회귀 모델
# 'Sales'를 종속변수로, 'Income', 'Advertising', 'Price', 'Age'를 독립변수로 설정합니다.
lmodel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=df).fit()

# 5. 모델 요약 결과 출력
print('요약결과 : ', lmodel.summary())
# 6. 2가지 모듈을 사용해서 저장 읽기
"""
# pickle 모듈 사용
import pickle
# 저장
with open('mymodel.pickle', mode='wb') as obj:
    pickle.dump(mymodel, obj) 
# 읽기 
with open('mymodel.pickle', mode='rb') as obj:
    mymodel = pickle.load(obj)
mymodel.predict('~~~')
"""


"""
# joblib모듈 사용
import joblib
# 저장
joblib.dump(lmodel, 'mymodel.model')
# 읽기
mymodel = joblib.load('mymodel.model')
mymodel.predict('~~~')
"""


# 선형회귀분석의 기본 충족 조건
print(df)
df_lm = df.iloc[:,[0,2,3,5,6]]
# 잔차항 얻기
fitted = lmodel.predict(df_lm)
residual = df_lm['Sales']-fitted
print(residual[:3])
print('잔차의 평균 :', np.mean(residual))

print('\n 선형성 : 잔차가 일정하게 분포되어야 함')

#시각화로 확인
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()],[0,0], '--', color='gray')
plt.show()  #잔차가 일정하게 분포되어 선형성 만족

print('\n 정규성 : 잔차가 정규 분포를 따라야 함')

sr = stats.zscore(residual)
(x,y), _ = stats.probplot(sr)
sns.scatterplot(x=x,y=y)
plt.plot([-3,3], [-3,3], '--', color='gray')
plt.show()

print('shapiro test',stats.shapiro(residual))  
#shapiro test ShapiroResult(statistic=np.float64(0.9949221268962878), pvalue=np.float64(0.21270047355487404)) > 0.05 이므로 정규성 만족

print('\n 독립성 : 독립변수 값이 서로 관련되지 않아야 한다.')
# 듀빈 왓슨 검정 값으로 확인
#    Durbin-Watson:                   1.931  -- 2에 근사하면 자기상관 없음 -> 독립적이다.
import statsmodels.api as sm
print('Durbin-Watson: ', sm.stats.stattools.durbin_watson(residual))   # summary를 보거나 이런 방법으로 확인 가능하다.

print('\n등분산성 (Homoscedasticity) :')
# # 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# 등분산성은 회귀 모델의 잔차(Residuals) 또는 오차항의 분산이 독립 변수들의 값과 관계없이 일정해야 한다는 가정
# 분산성 (Homoscedasticity): 잔차의 분산이 일정한 경우.
# 이분산성 (Heteroscedasticity): 잔차의 분산이 독립 변수의 특정 값에서 커지거나 작아지는 등 일정하지 않은 경우.
sr = stats.zscore(residual)
sns.regplot(x=fitted, y=np.sqrt(abs(sr)), lowess=True, line_kws={'color':'red'})
plt.show()  #잔차가 일정하게 분포되어 선형성 만족
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, lmodel.model.exog)  #exog 독립변수
print(f'통계량 : {bp_test[0]:.4f}, p-value : {bp_test[1]:.4f}')
#   p-value : 0.8899 > 0.05 등분산성 만족
print('\n다중공선성 :')
# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.
# 다중공선성(Multicollinearity)은 회귀 분석에서 독립 변수들 사이에 강한 선형 상관관계가 존재할 때 발생하는 문제
#  **분산팽창지수(VIF, Variance Inflation Factor)**는 이러한 다중공선성의 정도를 측정하는 지표 / 연속형의 경우 10을 넘으면 의심
from statsmodels.stats.outliers_influence import variance_inflation_factor
imsidf = df[['Income','Advertising','Price','Age']]
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(imsidf.values, i) for i in range(imsidf.shape[1])]

print(vifdf) # 모든 독립변수가 다중공선성에 해당하지 않음 
vifdf['vif_value'] = [variance_inflation_factor(imsidf.values, i) for i in range(imsidf.shape[1])]

print(vifdf) # 모든 독립변수가 값 10미만이므로 다중공선성 문제 발생하지 않음 

import joblib

ourmodel = joblib.load('mymodel.model')
new_df = pd.DataFrame({'Income' :[35,63,25], 'Advertising':[6,3,11], 'Price':[105,88,77], 'Age':[33,55,22]})
new_pred = ourmodel.predict(new_df)
print('Sales 예측결과 :\n', new_pred)