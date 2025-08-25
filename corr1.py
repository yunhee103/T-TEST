# 공분산, 상관계수
#  두 변수의 패턴을 확인하기 위해 공분산을 사용. 단위크기에 영향을 받음
#  상관계수 : 공분산을 표준화, -1~0  ~ 1.  +-1에 근사하면 관계가 강함
#  공분산은 두 변수의 방향성(함께 변하는지 반대로 변하는지) 확인 가능 , 구체적인 크기를 표현은 곤란
#  공분산의 의미 : 공분산은 두 변수(X와 Y)의 값이 각자의 평균으로부터 얼마나 함께 퍼져 있는지 측정
import numpy as np
print(np.cov(np.arange(1,6,1), np.arange(2,7))) # 늘어남
print(np.cov(np.arange(10,60,10), np.arange(20,70,10))) # 늘어남
print(np.cov(np.arange(100,600,100), np.arange(200,700,100))) # 늘어남

print(np.cov(np.arange(1,6), (3,3,3,3,3))) #고정
print(np.cov(np.arange(1,6), np.arange(6,1,-1)))  #줄어듬
print('----------------------------------------')
x = [8,3,6,6,9,4,3,9,3,4]
print('x의 평균 : ' , np.mean(x))
print('x의 분산: ' , np.var(x))  # 평균과의 거리와 관련이 있음
y = [6,2,4,6,9,5,1,8,4,5]
print('y의 평균 : ' , np.mean(y))
print('y의 분산: ' , np.var(y))

# 상관계수 중요!!!

import matplotlib.pyplot as plt
# plt.scatter(x,y)
# plt.show()
print('x,y 공분산 :' ,np.cov(x,y))
print('x,y 공분산 : ',np.cov(x,y)[0,1])
print('x,y 상관계수 : ',np.corrcoef(x,y))
print('x,y 상관계수 : ',np.corrcoef(x,y)[0,1])

# 참고 : 비선형인 경우는 일반적인 상관계수 방법을 사용하면 안됨  -> 산포도를 먼저 그린 후 패턴 확인
m = [-3, -2, -1, 0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]

plt.scatter(m, n)
plt.show()
print('m, n 상관계수 :' , np.corrcoef(m,n)[0,1])  #무의미한 작업 