# 날씨 예보 (강우 여부)

import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf  # R 스타일의 수식을 사용해 모델을 만드는 기능을 제공
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print(data.head(3), data.shape)     # (366, 12)
data2 = pd.DataFrame() # 빈 데이터프레임을 생성
data2 = data.drop(['Date', 'RainToday'], axis=1)   # 'Date'와 'RainToday' 열을 삭제
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
print(data2.head(3), data2.shape)   # (366, 10)
print(data2.RainTomorrow.unique())  # 'RainTomorrow' 열에 어떤 고유값들이 있는지 확인

# 학습데이터와 검정데이터로 분리
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(train.shape, test.shape)   # 분리된 학습 데이터와 테스트 데이터의 크기를 확인
print(data2.columns) # 모든 열 이름(변수)을 출력
col_select = "+".join(train.columns.difference(['RainTomorrow']))
# 'RainTomorrow'를 제외한 모든 열 이름을 '+' 기호로 연결하여 하나의 문자열로 만듬.
# 이는 모델의 독립 변수(설명 변수)로 사용
print(col_select)
my_formula = 'RainTomorrow ~' + col_select # 'RainTomorrow ~' 뒤에 독립 변수들을 연결하여 로지스틱 회귀 모델을 위한 수식(formula)

# model = smf.glm(formula=my_formula, data=train, family=sm.families.Binomial()).fit() #(ver1)  glm()은 다양한 분포에 적용할 수 있는 일반화 선형 모델.
model = smf.logit(formula=my_formula, data=train).fit() #(ver2)   logit() 함수를 사용해 로지스틱 회귀 모델을 생성하고 학습 ogit()은 이진 분류를 위한 로지스틱 회귀에 특화된 함수

print(model.summary())
# print(model.params)  # 모델의 각 독립 변수(설명 변수)에 대한 계수(Coefficient)를 출력
print('예측값 : ' , np.rint(model.predict(test)[:5].values))
print('실제값 : ' , test['RainTomorrow'][:5].values)

# 분류 정확도 확인
conf_tab = model.pred_table()   # 모델의 예측값을 바탕으로 혼동 행렬(Confusion Matrix)을 생성
# 혼동 행렬은 모델의 예측 성능을 나타내는 표로, 오분류된 항목들을 한눈에 보여줌
print('conf_tab :\n' , conf_tab) # glm - pred_table 지원 안함  -> ver2 logit으로 써야함
print('분류 정확도 :', (conf_tab[0][0]+ conf_tab[1][1]/ len(train)))

# 혼동 행렬을 이용해 직접 정확도를 계산하는 방식 (정확도 = (진짜 양성 + 진짜 음성) / 전체 데이터 수)

from sklearn.metrics import accuracy_score  # 모델의 정확도를 쉽게 계산해주는 함수를 가져옴
pred = model.predict(test)
print('분류 정확도 :' , accuracy_score(test['RainTomorrow'], np.around(pred)))
