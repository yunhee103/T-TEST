import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf  
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

plt.rc('font', family='malgun gothic')
data = pd.read_csv('./Eatingout.txt', sep=r'\s+', engine='python')
"""
로지스틱 분류분석 문제1]
문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

종속변수 (Dependent Variable): 외식 여부 (1: 외식, 0: 외식 안 함)
독립변수 (Independent Variable): 소득 수준
"""
data2 = pd.DataFrame() 
data2 = data.drop(['요일'], axis=1)
print(data2.소득수준.unique())
# 학습데이터와 검정데이터로 분리
train, test = train_test_split(data2, test_size=0.3, random_state=10)
print(train.shape, test.shape) 
print(data2.columns)

formula = '외식유무 ~ 소득수준'
model = smf.glm(formula=formula, data=train, family=sm.families.Binomial()).fit()
print(model.summary())
print(model.params)  # 모델의 각 독립 변수(설명 변수)에 대한 계수(Coefficient)를 출력
print('예측값 : ' , np.rint(model.predict(test)[:5].values))
print('실제값 : ' , test['외식유무'][:5].values)

pred = model.predict(test)
print('분류 정확도 :' , accuracy_score(test['외식유무'], np.around(pred)))

x = int(input("\n소득 수준을 입력하세요 (정수): "))
new_df = pd.DataFrame({'소득수준':[x]})

prob = model.predict(new_df)[0]

print(f"\n입력한 소득: {x}")
print(f" [GLM]   외식확률={prob:.4f}, 분류결과={np.rint(prob)}")