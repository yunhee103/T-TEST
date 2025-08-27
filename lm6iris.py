# 단순선형회귀 : ols 사용
# 상관관계가 선형회귀모델에 미치는 영향에 대해
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


iris = sns.load_dataset('iris')
print(iris.head(2))
print(iris.iloc[:,0:4].corr())

#  연습1 : 상관관계가 약한(  -0.117570  ) 두 변수를 사용( sepal_width, sepal_length)를 사용
results1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('검정 결과1 ',  results1.summary())
print('결정계수: ',  results1.rsquared)       # 0.013822654141080859
print('p-value 결과 ',  results1.pvalues[1]) #0.15189826071144785 > 0.05 유의하지않다. 
#
# plt.scatter(iris.sepal_width,iris.sepal_length)
# plt.plot(iris.sepal_width, results1.predict(), color='r')
# plt.show()

# -> 그래프를 보니 의미가 없음

#  연습2 : 상관관계가 강한( 0.871754 ) 두 변수를 사용( petal_length, sepal_length)를 사용
result2 = smf.ols(formula='sepal_length ~ petal_length ', data=iris).fit()
print('검정 결과2 ',  result2.summary())
print('결정계수: ',  result2.rsquared)       #  0.759954645772515
print('p-value 결과 ',  result2.pvalues.iloc[1]) #1.0386674194498124e-47 < 0.05 유의한 모델. 

plt.scatter(iris.sepal_length,iris.petal_length)
plt.plot(iris.petal_length, result2.predict(), color='r')
plt.show()
print()
# 입부의 실제값과 예측값 비교
print(' 실제값 : ', iris.sepal_length[:5].values)
print(' 예측값 : ', result2.predict()[:5])



#새로운 값으로 예측
new_data = pd.DataFrame({'petal_length':[1.1, 0.5, 5.0]})
y_pred = result2.predict(new_data)
print('예측결과 <sepal_length> :\n', y_pred)


print('------ 다중 선형 회귀 : 독립변수 복수 -------')
# result3 = smf.ols(formula='sepal_length ~ petal_length+petal_width+sepal_width ', data=iris).fit()

# 독립변수의 개수가 많을때 줄여서 쓰는 방법
column_select = "+".join(iris.columns.difference(['sepal_length','species']))
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
print(result3.summary())    # 독립변수가 여러개일 때는 R-squared보다 Adj. R-squared를 살펴야함 