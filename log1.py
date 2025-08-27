
# 로지스틱 회귀 (Logistic Regression)
# 독립 변수(FEATURE, x): 연속형, 종속 변수(LABEL, CLASS, y): 범주형
# 이항 분류 (다항 분류도 가능)
# 출력도니 연속형(확률) 자료를 logit변환해 최종적으로 sigmoid function에 의해 0에서 부터 1사이의 실수값이 나오는데 0.5를 기준으로 0,1 분류
# 원리:
# 1. 독립 변수들(mpg, hp)을 사용하여 연속형 값을 예측.
# 2. 이 연속형 값을 시그모이드(sigmoid) 함수에 넣어 0에서 1 사이의 확률값으로 변환.
# 3. 이 확률값이 0.5보다 크면 1(자동), 0.5보다 작으면 0(수동)으로 분류.

# 시그모이드 함수 예시

import math
def sigmoidFunc(x):
    return 1 / (1+math.exp(-x))

print(sigmoidFunc(3))
print(sigmoidFunc(1))
print(sigmoidFunc(-123))


# mtcars 데이터셋을 이용한 로지스틱 회귀 모델 분석
import statsmodels.api as sm
# 'mtcars'는 R 언어에 포함된 자동차 데이터셋입니다.
# statsmodels.datasets.get_rdataset을 사용해 이 데이터를 파이썬으로 가져옵니다.
mtcardata =sm.datasets.get_rdataset('mtcars')
print(mtcardata.keys())
mtcars = mtcardata.data
print(mtcars.head(2))
# mpg(연비)와 hp(마력)가 am(자동/수동)에 미치는 영향 분석
mtcar = mtcars.loc[:, ['mpg' , 'hp' , 'am']]
print(mtcar.head(3))
print(mtcar['am'].unique()) # am 변수의 고유한 값들([1, 0])

# 연비와 마력수에 따른 변속기 분류 모델 생성
# 모델 생성 방법: logit() 함수 사용
# 'am ~ hp + mpg'는 'am'을 종속 변수로, 'hp'와 'mpg'를 독립 변수로 사용하겠다는 의미
import statsmodels.formula.api as smf
formula = 'am ~ hp + mpg'
model1 = smf.logit(formula=formula, data=mtcar).fit()
print(model1.summary()) # Logit Regression Results

import numpy as np
# print( '예측값 :', model1.predict())
pred = model1.predict(mtcar[:10])
# model1.predict()는 각 데이터 포인트에 대해 모델이 계산한 '확률'을 반환합니다.
# 이 확률은 0에서 1사이의 실수값입니다.
print('예측값 : ', pred.values)
# np.around()는 반올림 함수
# 예측된 확률(pred)을 0.5 기준으로 0 또는 1로 반올림하여 최종 예측값(분류 결과)을 얻음
print('예측값 : ', np.around(pred.values)) 
print('실제값 : ', mtcar['am'][:10].values)
print()
# 분류 모델의 정확도(accuracy) 확인
# model1.pred_table()은 혼동 행렬(Confusion Matrix)을 반환
conf_tab = model1.pred_table() #수치에 대한 집계표
print('confusion matrix : ', conf_tab)
print('분류 정확도 : ' , (16+10) / len(mtcar))  #모델이 맞춘 갯수 / 전체 갯수
print('분류 정확도 : ' , (conf_tab[0][0] + conf_tab[1][1])/ len(mtcar))

"""

            예측: 0(수동) | 예측: 1(자동)
실제: 0(수동) |    16     |     3
실제: 1(자동) |     3     |    10
실제 0을 0으로 예측한 경우(True Negative): 16개. 모델이 수동 변속기 차량을 정확하게 수동으로 분류.
실제 0을 1로 예측한 경우(False Positive): 3개. 모델이 수동 변속기 차량을 잘못해서 자동으로 분류.
실제 1을 0으로 예측한 경우(False Negative): 3개. 모델이 자동 변속기 차량을 잘못해서 수동으로 분류.
실제 1을 1로 예측한 경우(True Positive): 10개. 모델이 자동 변속기 차량을 정확하게 자동으로 분류.

분류 정확도(Accuracy) 계산
분류 정확도는 모델이 전체 데이터 중에서 얼마나 많은 샘플을 올바르게 예측했는지를 나타내는 지표입니다.

정확도 공식
정확도=(올바르게 예측한 샘플의 수) / (전체 샘플의 수)

제공된 혼동 행렬을 사용하여 계산하면:

올바르게 예측한 샘플 수: 16 (실제 0, 예측 0) + 10 (실제 1, 예측 1) = 26

전체 샘플 수: 16 + 3 + 3 + 10 = 32

따라서, 정확도는:
정확도=26/32=0.8125

이 값은 **81.25%**의 정확도로 모델이 변속기 종류를 예측했다는 것을 의미
이 모델은 100개 중 약 81개를 올바르게 분류한다고 해석할 수 있습니다.
"""