# LogistcRegression 클래스 : 다항분류 가능
# 다항 분류도 가능한 LogisticRegression 클래스
# 활성화 함수는 softmax (다항 분류시)

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
iris = datasets.load_iris()
# print(iris.DESCR)  # 데이터셋에 대한 설명을 출력 
print(iris.keys())  # iris 데이터셋이 담고 있는 키(key), 즉 어떤 정보가 있는지 확인
print(iris.target)  # 붓꽃의 종류(0, 1, 2)를 나타내는 정수 배열을 출력
x = iris['data'][:,[3]]  # 붓꽃 데이터 중 네 번째 열(인덱스 3)인 'petal width(꽃잎 너비)'만 선택하여 독립 변수 x로 사용
print(x)
y = (iris.target == 2).astype(np.int32)  # 종속 변수 y를 생성
# 이 코드는 붓꽃의 종류(iris.target)가 '2'인 경우(즉, 버지니카 품종) True, 그 외에는 False를 반환하는 불리언(boolean) 배열
# .astype(np.int32)를 사용해 True는 1로, False는 0으로 변환하여 이진 분류 문제에 맞는 형태
# 즉, 이 코드는 '버지니카(versicolor) 품종인가?'를 예측하는 이진 분류 모델을 만들기 위한 데이터 변환 과정
print(y[:3]) # 변환된 종속 변수 y의 첫 3개 값을 출력
print(type(y))

log_reg = LogisticRegression().fit(x,y)  # LogisticRegression 모델 객체를 생성하고 학습
# solver : 1bfgs(softmax사용)
# solver='lbfgs'는 LogisticRegression의 기본값이며, 
# 'lbfgs'는 경사 하강법과 유사한 최적화 알고리즘. 
# 이 알고리즘은 다항 분류 시 softmax 활성화 함수를 내부적으로 사용
print(log_reg)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1) #새로운 예측값을 얻기위해 독립변수 생성
# np.linspace(0, 3, 1000)는 0부터 3까지 1000개의 등간격으로 나뉜 숫자를 생성합니다.
# 이 숫자들은 새로운 꽃잎 너비 값으로 사용됩니다.
# .reshape(-1, 1)은 1000개의 값으로 이루어진 1차원 배열을 1000행 1열의 2차원 배열로 변환
# sklearn 모델은 보통 2차원 배열 형태의 입력 데이터를 요구하므로, 이 과정이 필요
# print(x_new) # 새로 생성된 꽃잎 너비 데이터(x_new)를 출력
y_proba = log_reg.predict_proba(x_new)
# print(y_proba)

import matplotlib.pylab as plt
plt.plot(x_new, y_proba[:, 1], 'r-', label = 'virginica')
# x축: 새로운 꽃잎 너비 값(x_new)
# y축: 모델이 예측한 확률 값(y_proba) 중 두 번째 열(인덱스 1)을 사용. 이는 '버지니카'일 확률을 의미.
# 'r-': 빨간색 실선으로 그래프.
# label = 'virginica': 이 곡선의 범례(legend) 이름을 'virginica'로 설정다.
plt.plot(x_new, y_proba[:,0], 'b--', label = 'not virginica')
# x축: 새로운 꽃잎 너비 값(x_new)
# y축: 모델이 예측한 확률 값(y_proba) 중 첫 번째 열(인덱스 0)을 사용. 이는 '버지니카'가 아닐 확률을 의미.
# 'b--': 파란색 점선으로 그래프 .
# label = 'not virginica': 이 곡선의 범례 이름을 'not virginica'로 설정
plt.xlabel('petal width')
plt.legend()
plt.show()
"""그래프를 보면, 꽃잎 너비가 작을 때는 '버지니카가 아닐' 확률이 매우 높다가, 꽃잎 너비가 커질수록 '버지니카'일 확률이 높아지는 것을 확인할 수 있음. 
두 곡선이 교차하는 지점은 확률이 50%가 되는 지점이며, 이 지점이 바로 모델이 두 클래스를 구분하는 결정 경계가 됨
"""
