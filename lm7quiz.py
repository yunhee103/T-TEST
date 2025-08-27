# 회귀분석 문제 2) 
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측          -단순
#   - 국어, 영어 점수를 입력하면 수학 점수 예측     -다중
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 정리
student = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
print(student)
print(student.columns)
print(student.info())
numeric_student = student.drop('이름', axis=1)

# 2. 상관관계 분석
print(numeric_student.corr())

# 3. 단순 선형 회귀: 국어 점수로 수학 점수 예측
# 올바른 예측: result1 모델 사용
print('\n--- 단순 선형 회귀 모델: 국어 → 수학 ---')
result1 = smf.ols(formula='수학 ~ 국어', data=numeric_student).fit()
print(result1.summary())          # Prob (F-statistic):     8.16e-05 < 0.05 이므로 유의하다


# 산점도 시각화
plt.scatter(numeric_student['국어'], numeric_student['수학'])
plt.xlabel('KOREAN')
plt.ylabel('MATH')
plt.title('단순 선형 회귀: 국어-수학')
# 예측선 추가
plt.plot(numeric_student['국어'], result1.predict(), color='red')
plt.show()
# y =  0.5705*X +  32.1069 
# 국어 (기울기): 0.5705 : 국어 점수가 1점 증가할 때마다 수학 점수가 평균적으로 0.5705점 증가한다
# 계수 값이 양수이므로, 국어 점수가 높을수록 수학 점수도 높아지는 양의 상관관계

print('국어 점수 70점의 예측 수학 점수 :', result1.predict(pd.DataFrame({'국어':[70]})))


# 4. 다중 선형 회귀: 국어, 영어 점수로 수학 점수 예측
print('\n--- 다중 선형 회귀 모델: 국어, 영어 → 수학 ---')
# '수학'을 종속 변수로, '국어'와 '영어'를 독립 변수로 설정
result2 = smf.ols(formula='수학 ~ 국어 + 영어', data=numeric_student).fit()
print(result2.summary())

print('국어 점수, 영어 점수에 따른 예측 수학 점수 :', result2.predict(pd.DataFrame({'국어':[70],'영어':[70]})))

"""
모델 전체: Prob (F-statistic) 값이 0.000105로 0.05보다 작기 때문에 전체 모델은 통계적으로 유의
개별 변수:
국어: P>|t| 값이 0.663 > 0.05  국어 점수가 수학 점수 예측에 통계적으로 유의미한 영향을 주지 못한다.
영어: P>|t| 값이 0.074 > 0.05  비록 0.05에 가깝지만, 일반적으로 통용되는 유의수준인 0.05를 넘기므로 통계적으로 유의하지 않다고 판단.

모델 자체는 유의하지만, 포함된 개별 변수(국어, 영어)는 유의수준 0.05를 기준으로 볼 때 통계적으로 유의미한 기여를 하고 있지 않습니다.
특히 국어 점수의 p-값(0.663)은 매우 높아, 이 모델에서 국어 점수는 수학 점수 예측에 거의 의미가 없다고 볼 수 있습니다.
"""