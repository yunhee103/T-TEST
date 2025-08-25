# ML (기계학습 - 지도학습)
# 회기분석 : 입력 데이터에 대한 잔차제곱합이 최소가 되는 추세선(회귀선)을 만들고 ,
# 이를 통해 독립 변수가 종속변수에 얼마나 영향을 주는지 인과관계를 분석
# 독립변수 : 연속형, 종속변수 : 연속형 두 변수는 상관관계가 있어야하며 인과관계를 보여야 한다.
# 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
np.random.seed(12)

# 모델 생성후 맛보기
# 방법 1: make_regression을 사용 : model x

x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x)
print(y)
print(coef)  # 89.47430739278907 기울기
# 회귀식 : y = wx + b  y = 89.4743  * x + 100
y_pred = 89.47430739278907 * -1.7007563 + 100
print(y_pred)  #    실제값 : -52.17214291 예측값 : -52.17399198642261

# 미지의 x(5)에 대한 예측값 y 얻기
print('y_pred_new : ', 89.47430739278907 * 5 + 100)  # y_pred_new :  547.3715369639453

print()
xx = x
yy = y
# 방법 2 : LinerRegression을 사용 : model o
from sklearn.linear_model import LinearRegression

model = LinearRegression()
fit_model = model.fit(xx, yy)  #학습 데이터로 모형 추정. 절편, 기울기 얻이므
print(fit_model.coef_)  # 기울기
print(fit_model.intercept_) # 절편 
# [89.47430739]  100.0
print('예측값y의[0] : ' , 89.47430739 * xx[[0]] + 100.0)  # -52.1721429
print('예측값y의[0] : ' , model.predict(xx[[0]])) # -52.1721429
# 미지의 xx(5)에 대한 예측값 y 얻기   
print('미지의 x에 대한 예측값 y : ' , model.predict([[5]]))  
print('미지의 x에 대한 예측값 y : ' , model.predict([[5],[3]]))
print()
# 방법 3 : ols 사용 : model o
import statsmodels.formula.api as smf
import pandas as pd
x1 = xx.ravel()  # 차원 축소   => x1 = xx.flatten()
print(x1.shape)
y1 = yy

data = np.array([x1, y1])
# print(data.T)
df = pd.DataFrame(data.T)
df.columns = ['x1','y1']
print(df.head(2))
model2 = smf.ols(formula='y1 ~ x1', data=df).fit()
print(model2.summary())             # Intercept:100.0000 , x1의 기울기: 89.4743
print(x1[:2]) # [-1.70073563 -0.67794537] 
new_df = pd.DataFrame({'x1' : [-1.70073563, -0.67794537]})  #기존 자료로 예측값(성능) 확인
new_pred = model2.predict(new_df)
print('new_pred :', new_pred.values) 

# 전혀 새로운 독립변수로 종속변수 예측
new_df2 = pd.DataFrame({'x1' : [123, -2.677]}) 
new_pred2 = model2.predict(new_df2)
print('new_pred2 :', new_pred2.values) 
# 방법 4 : linregress model o 
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# iq에 따른 시험 점수 값 예측
score_iq = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())

x = score_iq.iq
y = score_iq.score


# 상관계숙 확인
print(np.corrcoef(x,y)[0,1])  # 0.88222
print(score_iq.corr())
plt.scatter(x,y)
plt.show()
plt.close()
#  r-value : 결정계수 
model = stats.linregress(x,y)
print(model)
print('기울기 : ', model.slope)
print('절 편 : ', model.intercept)
print('R² - 결정계수(설명력): ' , model.rvalue)
print('p-value : ' , model.pvalue)
print('표준오차 : ' , model.stderr)

# y^ = wx +b => 0.6514309527270075 * x + -2.8564471221974657
"""
기울기 :  0.6514309527270075
절 편 :  -2.8564471221974657
R² - 결정계수 :  0.8822203446134699 : 독립변수가 종속변수를 88% 정도 설명하고 있다.
p-value :  2.8476895206683644e-50 < 0.05  이므로 현재 모델은 유의하다.(독립변수와 종혹변수)
표준오차 :  0.028577934409305443
"""

plt.scatter(x,y)
plt.plot(x, model.slope * x + model.intercept)
plt.show()

# 점수예측
print('점수 예측 :', model.slope*80 + model.intercept)
print('점수 예측 :', model.slope*120 + model.intercept)

# predict X가 없음
print('점수 예측 :', \
      np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))

print()
newdf = pd.DataFrame({'iq':[55,66,77,88,150]})
print('점수 예측 :', \
      np.polyval([model.slope, model.intercept], np.array(newdf)))
