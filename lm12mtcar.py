# LinearRegression으로 선형회귀 모델 작성 - mtcars

import statsmodels.api
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars.corr(method='pearson'))
print()

x = mtcars[['hp']].values
print(x[:3])
y = mtcars['mpg'].values
print(y[:3])

lmodel = LinearRegression().fit(x,y)
print('slope : ', lmodel.coef_)
print('intercept :' , lmodel.intercept_)
# plt.scatter(x,y)
# plt.plot(x,lmodel.coef_*x + lmodel.intercept_, c = 'r')
# plt.show()

pred = lmodel.predict(x)
print('예측값 :' , np.round(pred[:5], 1))  #소수 첫째자리까지
print('실제값 :' , y[:5])  
print()

print('MSE : ', mean_squared_error(y, pred))
print('r2_score : ', r2_score(y, pred))
# MSE :  13.989822298268805
# r2_score :  0.602437341423934

# 새로운 마력수에 대한 연비는 ? 
new_hp = [[123]]
new_pred = lmodel.predict(new_hp)
print('%s 마력인 경우 연비는 약 %s 입니다'%(new_hp[0][0], new_pred[0]))
