# 비선형회귀 분석
# 선형관계분석의 경우 모델에 다항식 또는 교호작용이 있는 경우에는 해설이 덜 직관적이다.
# 결과의 신뢰성이 떨어진다.
# 선형가정이 어긋날 때(정규성 위배) 대처하는 방법으로 다항식항을 추가한 다항회귀 모델을 작성할 수 있다.

import numpy as np # 수치 계산을 위한 핵심 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리
from sklearn.metrics import r2_score # 모델의 성능 평가를 위한 결정계수(R-제곱) 함수

# 학습에 사용할 샘플 데이터 생성
x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

# 데이터 시각화
# plt.scatter(x,y)
# plt.show()
print(np.corrcoef(x,y)) # x와 y의 상관 계수 계산. 1에 가까울수록 양의 선형 관계, -1에 가까울수록 음의 선형 관계를 나타냄. 0에 가까우면 선형 관계가 약하다는 의미.

# 선형회귀 모델 작성

from sklearn.linear_model import LinearRegression # 선형 회귀 모델을 위한 클래스
x = x[:, np.newaxis] # scikit-learn 모델에 입력하기 위해 (5,) 형태의 1차원 배열을 (5, 1) 형태의 2차원 배열로 차원 확대
print(x) # 변환된 x의 형태 출력
# [[1], [2], [3], [4], [5]]

# 선형회귀 모델 학습 및 예측
model1 = LinearRegression().fit(x,y) # 데이터를 사용하여 모델을 학습. y = ax + b 형태의 최적의 직선을 찾음.
ypred = model1.predict(x) # 학습된 모델을 사용하여 x 값에 대한 y 예측값 계산
print('예측값 : ', ypred)
# [4.5 3.5 2.5 3.  4. ] -> 예시와 다른 예측값 출력.
# 2.8 3.5 4.2 4.9 5.6
print('결정계수 : ', r2_score(y,ypred)) # R-제곱 값 계산. 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미. 0에 가까우면 설명력이 낮다는 뜻.
# 0.16071428571428573 -> 예시와 다른 예측값 출력.
# 0.1607142857142857

# 선형회귀 모델 시각화 (산점도와 예측선)
# plt.scatter(x,y)
# plt.plot(x,ypred, c='r')
# plt.show()

# 다항회귀 모델 작성 - 추세선의 유연성을 위해 열 추가
# degree = 열수
from sklearn.preprocessing import PolynomialFeatures # 다항식 특성 변환을 위한 클래스
poly = PolynomialFeatures(degree=2, include_bias=False) # 2차 다항식을 위한 객체 생성. 'degree'는 추가할 최고차항의 차수.
x2 = poly.fit_transform(x) # 특징 행렬 만듦. 원본 x에 x의 제곱(x^2) 항을 추가하여 새로운 특성 행렬을 생성.
print(x2)
"""
[[ 1.  1.]
 [ 2.  4.]
 [ 3.  9.]
 [ 4. 16.]
 [ 5. 25.]]
다항 회귀 모델에서 특징 행렬을 만드는 이유는 선형 모델이 비선형 관계를 학습할 수 있도록 데이터를 변환하기 위함. 
이는 직선 형태의 모델(선형 회귀)을 곡선 형태의 모델(다항 회귀)로 확장하여 데이터의 복잡한 추세를 더 잘 표현하기 위해서
"""
# 변환된 특징 행렬 x2를 사용하여 다항회귀 모델 학습 및 예측
model2 = LinearRegression().fit(x2, y) # 변환된 데이터(x, x^2)를 사용하여 모델을 학습. y = a1*x + a2*x^2 + b 형태의 곡선을 찾음.
ypred2 = model2.predict(x2) # 학습된 다항회귀 모델을 사용하여 y 예측값 계산
print('예측값 : ', ypred2)
# [4.58571429 2.17142857 0.98571429 2.02857143 4.2        ]
print('결정계수 : ', r2_score(y,ypred2)) # 다항회귀 모델의 R-제곱 값 계산. 선형회귀보다 더 높은 값이 나올 가능성이 큼.
# 0.9161764705882352
# plt.scatter(x,y)
# plt.plot(x, ypred2, c='b') # 다항회귀 모델이 생성한 곡선을 파란색 선으로 시각화
# plt.show()