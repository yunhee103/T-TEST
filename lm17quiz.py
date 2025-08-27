import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
plt.rc('font', family='Malgun Gothic')

# 1. 데이터 로드 및 열 이름 지정
# header=None: 데이터에 헤더가 없어 첫 번째 행을 데이터로 인식하도록 지정합니다.
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data', header=None, sep=r',')
# 데이터셋의 열 이름이 없으므로, 열 이름을 직접 지정
df.columns = ["motor", "screw", "pgain", "vgain", "class"] 
print(df.head(2)) # 데이터프레임의 상위 2개 행을 출력하여 데이터가 제대로 로드되었는지 확인합니다.
print(df.info()) # 데이터프레임의 기본 정보(열, 데이터 타입, 결측치 유무)를 출력합니다.

# 2. 타깃/피처 분리
# 문제 요구사항에 따라, 숫자형 피처인 'pgain'과 'vgain'을 독립 변수(x)로,
# 예측하고자 하는 'class'를 종속 변수(y)로 분리합니다.
# .astype(float): 'pgain'과 'vgain' 열의 데이터 타입을 실수형으로 변환합니다.
x = df[["pgain", "vgain"]].astype(float) 
y = df["class"].values

# 3. 학습/테스트 분할 (8:2)
# train_test_split: 데이터를 학습용(80%)과 테스트용(20%)으로 나눕니다.
# test_size=0.2: 전체 데이터의 20%를 테스트 데이터로 설정합니다.
# random_state=: 무작위 분할 시 난수 시드를 고정하여, 코드를 다시 실행해도 동일한 결과를 얻을 수 있도록 합니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# 4. 스케일링 (StandardScaler)
# StandardScaler: 평균이 0, 표준편차가 1이 되도록 데이터를 표준화(정규화)합니다.
#                 피처들의 스케일 차이로 인해 특정 피처가 모델에 더 큰 영향을 미치는 것을 방지합니다.
scaler = StandardScaler()
# fit_transform(): 학습 데이터의 평균과 표준편차를 계산(fit)하고,
#                   이를 바탕으로 데이터를 변환(transform)합니다.
x_train_scaled = scaler.fit_transform(x_train)
# transform(): 테스트 데이터는 학습 데이터의 통계량(평균, 표준편차)을 사용하여 변환만 합니다.
#              이를 통해 데이터 편향을 방지하고 실제 환경에 가까운 평가를 수행합니다.
x_test_scaled = scaler.transform(x_test)

# 5. 다항 특성 변환 (degree=2)
# PolynomialFeatures: 피처의 다항 조합을 생성하여 비선형 관계를 모델링할 수 있게 합니다.
# degree=2: 2차 다항식을 생성합니다 (예: $x_1$, $x_2$, $x_1^2$, $x_2^2$, $x_1x_2$).
# include_bias=False: 상수항($x_0$)을 생성하지 않도록 설정합니다.
poly = PolynomialFeatures(degree=2, include_bias=False)
# 학습 데이터에 다항 특성을 적용하여 변환합니다.
x_train_poly = poly.fit_transform(x_train_scaled)
# 테스트 데이터에도 동일한 변환을 적용합니다.
x_test_poly = poly.transform(x_test_scaled)

"""
x_test에는 fit()을 사용하지 않을까요?
데이터 누수를 방지하기 위해서.
데이터 누수는 테스트 데이터의 정보가 모델 학습에 영향을 미치는 현상으로, 실제 환경에서의 성능을 과대평가하게 만듬
마치 시험 문제를 풀기 전에 답을 미리 보는 것과 같습니다. x_train은 공부한 내용이고, x_test는 실제 시험 문제입니다.
공부 (학습 데이터 fit): 시험 문제를 풀기 위해 공부 내용만으로 규칙을 익힙니다.
시험 (테스트 데이터 transform): 시험 문제를 풀 때는 공부한 규칙만 사용해야 합니다. 시험 문제 자체에서 새로운 규칙을 배우면 안 됩니다.
PolynomialFeatures의 경우, fit 과정은 '어떤 조합을 만들어낼지'에 대한 규칙을 학습합니다. 
만약 x_test에 fit_transform을 사용하면, 모델이 학습 단계에서는 보지 못했던 테스트 데이터의 특성(예: 특정 피처 조합)을 미리 알게 되어 모델의 일반화 능력을 제대로 평가할 수 없게 됩니다.
따라서, 모델이 학습 데이터에만 의존하여 규칙을 배우고, 그 규칙을 테스트 데이터에 적용해야만 모델의 실제 성능을 공정하게 평가할 수 있습니다.
이것이 x_train에 fit_transform을, x_test에 transform을 사용하는 이유입니다.
"""

print(f"변환 전 피처 개수: {x_train_scaled.shape[1]} -> 변환 후 피처 개수: {x_train_poly.shape[1]}")
"""
<'피처(Feature)'는 머신러닝에서 데이터를 구성하는 특성(속성)을 의미. 통계학에서는 독립 변수(Independent Variable).
pgain, vgain 2차항: pgain², vgain², 교차항: pgain * vgain
"""
# 6. 모델 학습 (LinearRegression)
# LinearRegression: 선형 회귀 모델을 생성합니다.
# fit(): 변환된 학습 데이터(x_train_poly)와 타깃 데이터(y_train)를 사용하여 모델을 학습시킵니다.
model = LinearRegression()
model.fit(x_train_poly, y_train)

# 7. 성능 평가
# predict(): 학습된 모델을 사용하여 테스트 데이터(x_test_poly)의 'class' 값을 예측합니다.
y_pred = model.predict(x_test_poly)
# mean_squared_error: 실제 값(y_test)과 예측 값(y_pred) 간의 평균제곱오차(MSE)를 계산합니다.
#                     오류 값의 제곱을 사용해 큰 오류에 더 큰 패널티를 줍니다.
MSE = mean_squared_error(y_test, y_pred)

MAE = mean_absolute_error(y_test, y_pred)
# r2_score: 결정계수(R-squared)를 계산합니다.
#           모델이 분산을 얼마나 잘 설명하는지 나타내며, 1에 가까울수록 좋은 성능을 의미합니다.
r2 = r2_score(y_test, y_pred)
print("-" * 30)
print(f"평균제곱오차 (MSE): {MSE:.4f}")
print(f"결정계수 (R²): {r2:.4f}")
print(f"평균 절대 오차 (MAE): {MAE:.4f}")
# 8. 시각화
# plt.scatter: 산점도(scatter plot)를 생성하는 함수입니다.
# 첫 번째 plt.scatter: 테스트 데이터의 실제 값('pgain' vs. 'class')을 파란색 점으로 표시합니다.
plt.scatter(x_test['pgain'], y_test, color='b', label='테스트 데이터 (실제값)')
# 두 번째 plt.scatter: 테스트 데이터의 예측 값('pgain' vs. 예측값)을 빨간색 점으로 표시합니다.
#                      'pgain' 피처는 두 피처 중 하나를 선택하여 2차원 시각화를 가능하게 합니다.
plt.scatter(x_test['pgain'], y_pred, color='r', label='예측값 (LinearRegression)')
#plt.scatter(x_test['pgain'], y_test, color='y', label='테스트 데이터 (실제값)') -> 이 코드는 첫 번째 plt.scatter와 동일한 데이터를 표시하므로 중복되어 제거하는 것이 좋습니다.

plt.title('pgain에 대한 실제값 vs 예측값') # 그래프의 제목을 설정합니다.
plt.xlabel('pgain') # x축 레이블을 'pgain'으로 설정합니다.
plt.ylabel('class') # y축 레이블을 'class'로 설정합니다.
plt.legend() # 범례(label)를 표시합니다.
plt.grid(True) # 그래프에 격자(grid)를 표시합니다.
plt.show() # 생성된 그래프를 화면에 출력합니다.
plt.scatter(x_test['vgain'], y_test, color='b', label='테스트 데이터 (실제값)')
plt.scatter(x_test['vgain'], y_pred, color='r', label='예측값 (LinearRegression)')
plt.title('vgain에 대한 실제값 vs 예측값')
plt.xlabel('vgain') 
plt.ylabel('class')
plt.legend()
plt.grid(True) 
plt.show() 

plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # 대각선
plt.xlabel("실제 값 (y_test)")
plt.ylabel("예측 값 (y_pred)")
plt.title("다항 특성 Ridge 모델: 실제 값 vs. 예측 값")
plt.grid(True)
plt.show()

