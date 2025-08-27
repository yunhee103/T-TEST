# 비선형회귀 분석

# 데이터 분석에 필요한 라이브러리 및 함수 임포트
import numpy as np # 배열 및 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리
from sklearn.metrics import r2_score, mean_squared_error # 모델 평가를 위한 결정계수(R-제곱)와 평균제곱오차 함수
from sklearn.linear_model import LinearRegression # 선형회귀 모델
from sklearn.preprocessing import PolynomialFeatures # 다항식 특성 변환을 위한 클래스

# x, y 데이터 정의 및 x의 차원 확장 (모델 입력 형식에 맞게)
# np.newaxis를 사용하여 (10,) 형태의 1차원 배열을 (10, 1) 형태의 2차원 배열로 변환
x = np.array([257, 270, 294, 320, 342, 368, 396, 446, 480, 580])[:, np.newaxis]
print(x.shape) # x 배열의 형태 출력: (10, 1)
y = np.array([236, 234, 253, 298, 314, 342, 360, 368, 390, 388])

# 원본 데이터의 산점도 시각화
# plt.scatter(x,y)
# plt.show()

# 일반회귀모델(선형)과 다항회귀모델(2차 다항식) 작성 후 비교
lr = LinearRegression() # 일반 선형회귀 모델 객체 생성
pr = LinearRegression() # 다항회귀 모델도 결국 선형회귀 모델의 일종이므로 LinearRegression 객체를 사용
polyf = PolynomialFeatures(degree=2) # 2차 다항식 변환을 위한 객체 생성. 'degree=2'는 x와 x^2 항을 생성하겠다는 의미.
x_quad = polyf.fit_transform(x) # x 데이터에 2차항(x^2)을 추가하여 새로운 특징 행렬(x, x^2) 생성

# 일반회귀모델(lr) 훈련
lr.fit(x,y) # 원본 x와 y를 사용하여 선형회귀 모델을 훈련. y = ax + b 형태의 최적의 직선을 찾음.

# 시각화를 위한 예측 범위 설정 (원본 데이터의 범위를 벗어나 곡선/직선 추세를 더 잘 보여주기 위해)
x_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(x_fit) # 일반 선형회귀 모델이 예측한 값
print(y_lin_fit) # 예측값 출력

# 다항회귀모델(pr) 훈련
pr.fit(x_quad, y) # 2차항이 추가된 x_quad와 y를 사용하여 다항회귀 모델을 훈련. y = a1*x + a2*x^2 + b 형태의 최적의 곡선을 찾음.
y_quad_fit = pr.predict(polyf.fit_transform(x_fit)) # 다항회귀 모델이 예측한 값. 예측하려는 x_fit도 2차항으로 변환해야 함.
print(y_quad_fit) # 예측값 출력

# 시각화
plt.scatter(x, y, label='training point') # 원본 데이터(학습 데이터)를 산점도로 표시
plt.plot(x_fit, y_lin_fit, label = 'linear fit', linestyle='--', c='r') # 일반 선형회귀 모델의 예측 선을 빨간색 점선으로 표시
plt.plot(x_fit, y_quad_fit, label = 'quadratic fit', linestyle='-.', c='b') # 2차 다항회귀 모델의 예측 선을 파란색 점선으로 표시
plt.legend() # 각 선의 레이블을 표시하는 범례
plt.show() # 그래프를 화면에 출력

# 모델 성능비교를 위한 예측값 재계산
# 훈련 데이터셋(x)에 대한 예측값으로 성능을 평가
y_lin_fit = lr.predict(x)
y_quad_fit = pr.predict(x_quad)

# 성능비교 점수 출력
print('MSE : 선형:%.3f, 다항:%.3f'%(mean_squared_error(y, y_lin_fit), mean_squared_error(y, y_quad_fit)))
# MSE(평균제곱오차): 예측값과 실제값의 차이를 제곱하여 평균 낸 값. 값이 작을수록 모델의 예측 정확도가 높다는 의미.
# '선형'과 '다항' 모델의 MSE를 비교하여 어떤 모델이 더 정확한지 확인.
print('설명력 : 선형:%.3f, 다항:%.3f'%(r2_score(y, y_lin_fit), r2_score(y, y_quad_fit)))
# R-제곱(결정계수): 모델이 데이터의 변동성을 얼마나 잘 설명하는지 나타내는 지표. 1에 가까울수록 설명력이 높음.
# 이 값을 통해 두 모델 중 어느 것이 데이터를 더 잘 표현하는지 비교할 수 있음.