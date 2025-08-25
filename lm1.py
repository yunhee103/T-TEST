# 최소 제곱해를 선형 행렬 방정식으로 구하기

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import pandas as pd

x = np.array([0,1,2,3])
y = np.array([-1,0.2,0.5,2.1]) 
# plt.scatter(x, y)
# plt.show()
# plt.close()

A = np.vstack([x, np.ones(len(x))]).T
print(A)

import numpy.linalg as lin
# y = wx + b라는 일차 방정식의 w, b?
w, b = lin.lstsq(A, y, rcond=None)[0]    #데이터값을 이차원으로 줘야함 독립변수는 2차원 numpy 배열이나 pandas DataFrame, 종속변수는 1차원 numpy 배열이나 pandas Series로 입력받도록 설계되어 있습니다.
#  최소제곱법 연산
#  잔차 제곱의 총합이 최소가 되는 값을 얻을 수 있다.
print('w(weight, 기울기, slope) : ' ,w)
print('b(bias, 절편, 편향, inbtercept) : ' ,b)
# y = 0.95999 * x - 0.9899   단순선형회귀수식 모델
plt.scatter(x, y)
plt.plot(x, w * x + b, label='실제값')
plt.legend()
plt.show()

# 수식으로 예측값 얻기
print(w*1 + b)  # -0.0299 (예측값) -  0.2(실제값)  <=====잔차, 오차, 손실, 에러  / x=1넣은 이유 경험해보지않은 결과를 보여줌 100%신뢰 x, 확률 -> 그러할 것이다 참고자료

