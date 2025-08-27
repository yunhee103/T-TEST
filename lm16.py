# 보스톤 집값 데이터를 이용해 단순, 다항 회귀 모델 작성
import pandas as pd
import numpy as np # 배열 및 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리
from sklearn.metrics import r2_score, mean_squared_error # 모델 평가를 위한 결정계수(R-제곱)와 평균제곱오차 함수
from sklearn.linear_model import LinearRegression # 선형회귀 모델
from sklearn.preprocessing import PolynomialFeatures # 다항식 특성 변환을 위한 클래스

plt.rc('font', family='malgun gothic')

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data', header=None, sep=r'\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head(2))
print(df.corr()) 
x = df[['LSTAT']].values
y = df['MEDV'].values

model = LinearRegression()

quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)

# 단순회귀
model.fit(x, y)
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)
# print(y_lin_fit)
model_r2 = r2_score(y, model.predict(x))
print('model_r2 :', model_r2)

# 다항(2차)
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
q_r2 =  r2_score(y, model.predict(x_quad))
print('q_r2 :', q_r2) #q_r2 : 0.6407168971636611

# 다항(3차)
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
c_r2 =  r2_score(y, model.predict(x_cubic))
print('c_r2 :', c_r2) # c_r2 : 0.6578476405895719

# 시각화
plt.scatter(x, y, label='학습데이터', c='lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(d=1),$R^2=%.2f$'%model_r2, c='b', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle='-', label='quad fit(d=2),$R^2=%.2f$'%q_r2, c='r', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='cubic fit(d=3),$R^2=%.2f$'%c_r2, c='k', lw=3)

plt.xlabel('하위계층비율')
plt.ylabel('주택 가격')
plt.legend()
plt.show()