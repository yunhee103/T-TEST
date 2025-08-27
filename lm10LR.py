# sklearn 모듈의 linearRegression 클래스 사용 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
sample_size = 100
np.random.seed(1)

# 1) 편차가 없는 데이터 생성
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관계수 : ', np.corrcoef(x,y))
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled : ', x_scaled)
plt.scatter(x_scaled,y)
plt.show()

model = LinearRegression().fit(x_scaled,y)
print(model)
print('계수(slope)' , model.coef_) # 회귀계수 (각 독립변수)
print('절편(intercept)' , model.intercept_)
print('결정계수(R²)' , model.score(x_scaled,y)) #설명력 : 훈련 데이터 기준
# y = wx + b    <== 1350.4161554 * x + -691.1877661754081
# 종속변수에 스케일링하는것 아니다~ 독립변수에만 하는것~ model.summary()  x -> ols쓰세요~
y_pred = model.predict(x_scaled)
print('예측값(ŷ) : ' , y_pred[:5])  # [ 490.32381062 -182.64057041 -157.48540955 -321.44435455  261.91825779]
print('실제값(y) : ' , y[:5])       # [ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]

print()
# 선형 회귀 모델의 주요 평가지표는 
# MAE (Mean Absolute Error, 평균 절대 오차), 
# MSE (Mean Squared Error, 평균 제곱 오차), 
# RMSE (Root Mean Squared Error, 평균 제곱근 오차), 
# R² (결정계수) 
# MAE, MSE, RMSE는 값이 작을수록 모델의 성능이 좋으며, 실제값과 예측값 간의 오차를 나타냅니다. 
# 반면, R²는 값이 클수록(1에 가까울수록) 모델의 설명력이 높아 더 좋은 성능을 의미합니다

# 모델 성능 파악용 함수 작성
def RegScoreFunc(y_true, y_pred):
    print('R²(결정계수):{}'.format(r2_score(y_true,y_pred)))
    print('(설명분산점수):{}'.format(explained_variance_score(y_true,y_pred)))
    print('mean_squared_error(평균제곱오차):{}'.format(mean_squared_error(y_true,y_pred)))

RegScoreFunc(y, y_pred)
# R²(결정계수):0.9987875127274646
# (설명분산점수):0.9987875127274646
# mean_squared_error(평균제곱오차):86.14795101998747
print('------------------------------------------------------------------')
# 2) 편차가 있는 데이터 생성
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관계수 : ', np.corrcoef(x,y))  #     0.00401167
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled : ', x_scaled)
plt.scatter(x_scaled,y)
plt.show()

model = LinearRegression().fit(x_scaled,y)
print(model)

y_pred = model.predict(x_scaled)
print('예측값(ŷ) : ' , y_pred[:5]) #[-10.75792685  -8.15919008 -11.10041394  -5.7599096  -12.73331002]
print('실제값(y) : ' , y[:5])   # [1020.86531436 -710.85829436 -431.95511059 -381.64245767 -179.50741077]

RegScoreFunc(y, y_pred)
# R²(결정계수):1.6093526521765433e-05
# (설명분산점수):1.6093526521765433e-05
# mean_squared_error(평균제곱오차):282457.9703485092

