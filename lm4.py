import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용

나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.

- 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.

- 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.

참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

"""

# 1. 데이터 불러오기 및 이상치 제거
data = pd.read_csv('Book1.csv' , encoding='euc-kr')
# '운동' 시간이 10시간을 초과하는 이상치 행 제거
# 해당 행의 '운동' 값은 35.0이므로, 해당 행이 제거됩니다.
data = data[data['운동'] <= 10]
# 2. 결측치(NaN)를 '지상파' 열의 평균값으로 대체
mean_gs = data['지상파'].mean()
data['지상파'] = data['지상파'].fillna(mean_gs)
print(data.info())

# 독립 변수(x)와 종속 변수(y) 설정
x = data['지상파']
y = data['운동']

# 상관계수 확인
corr_zy = np.corrcoef(x, y)[0, 1]
print(f"지상파-운동 상관계수: {corr_zy:.4f}")

# 회귀분석 모델 생성
model_zy = stats.linregress(x, y)
print('\n[지상파 vs 운동] 회귀분석 결과')
print(f"기울기 (slope): {model_zy.slope:.4f}")
print(f"절편 (intercept): {model_zy.intercept:.4f}")
print(f"결정계수 (R²): {model_zy.rvalue**2:.4f}")
print(f"p-value: {model_zy.pvalue}")

# 지상파 5시간 시청 시 예상 운동 시간
predicted_exercise = model_zy.slope * 5 + model_zy.intercept
print(f"\n지상파 시청 시간 5시간일 때 예상 운동 시간: {predicted_exercise:.4f}시간")

# 독립 변수(x)와 종속 변수(y) 설정
x = data['지상파']
z = data['종편']

# 상관계수 확인
corr_zj = np.corrcoef(x, z)[0, 1]
print(f"지상파-종편 상관계수: {corr_zj:.4f}")

# 회귀분석 모델 생성
model_zj = stats.linregress(x, z)
print('\n[지상파 vs 종편] 회귀분석 결과')
print(f"기울기 (slope): {model_zj.slope:.4f}")
print(f"절편 (intercept): {model_zj.intercept:.4f}")
print(f"결정계수 (R²): {model_zj.rvalue**2:.4f}")
print(f"p-value: {model_zj.pvalue}")

# 예측에 사용할 새로운 데이터
new_data = np.array([55, 66, 77, 88, 150])

# np.polyval()을 사용하여 예측
# model_zj.slope와 model_zj.intercept를 사용해야 합니다.
predicted_cable = np.polyval([model_zj.slope, model_zj.intercept], new_data)
print(f"\n지상파 시청 시간 입력값에 대한 예상 종편 시청 시간:\n{predicted_cable}")




print('-----------------Statsmodels OLS --------------')



import statsmodels.api as sm

# 독립 변수(x)에 상수항(const) 추가
x_zy = sm.add_constant(data['지상파'])
y_zy = data['운동']

# OLS 모델 적합
model_zy_sm = sm.OLS(y_zy, x_zy).fit()

# 결과 요약(summary) 출력
print("[지상파 vs 운동] Statsmodels OLS 결과")
print(model_zy_sm.summary())

# 독립 변수(x)에 상수항(const) 추가
x_zj = sm.add_constant(data['지상파'])
y_zj = data['종편']

# OLS 모델 적합
model_zj_sm = sm.OLS(y_zj, x_zj).fit()

# 결과 요약(summary) 출력
print("\n[지상파 vs 종편] Statsmodels OLS 결과")
print(model_zj_sm.summary())




print('-----------------LinearRegression--------------')


from sklearn.linear_model import LinearRegression

# 독립 변수를 2차원 배열로 변환 (sklearn 요구 형식)
x_sk = data[['지상파']]
y_sk = data['운동']

# 1. 지상파 vs 운동
model_xy_sk = LinearRegression().fit(x_sk, y_sk)
print("[지상파 vs 운동] Scikit-learn LinearRegression 결과")
print(f"기울기 (slope): {model_xy_sk.coef_[0]:.4f}")
print(f"절편 (intercept): {model_xy_sk.intercept_:.4f}")
print(f"결정계수 (R²): {model_xy_sk.score(x_sk, y_sk):.4f}")
# 지상파 vs 운동 산점도 그리기
plt.scatter(x, y)

# 회귀선 그리기
# y = 기울기 * x + 절편
y_pred_zy = model_zy.slope * x + model_zy.intercept
plt.plot(x, y_pred_zy, color='red')

plt.title('지상파 vs 운동')
plt.xlabel('지상파 시청 시간')
plt.ylabel('운동 시간')
plt.show()

# 2. 지상파 vs 종편
x_sk = data[['지상파']]
z_sk = data['종편']

model_xz_sk = LinearRegression().fit(x_sk, z_sk)
print("\n[지상파 vs 종편] Scikit-learn LinearRegression 결과")
print(f"기울기 (slope): {model_xz_sk.coef_[0]:.4f}")
print(f"절편 (intercept): {model_xz_sk.intercept_:.4f}")
print(f"결정계수 (R²): {model_xz_sk.score(x_sk, z_sk):.4f}")

# 지상파 vs 종편 산점도 그리기
plt.scatter(x, z)

# 회귀선 그리기
y_pred_zj = model_zj.slope * x + model_zj.intercept
plt.plot(x, y_pred_zj, color='red')

plt.title('지상파 vs 종편')
plt.xlabel('지상파 시청 시간')
plt.ylabel('종편 시청 시간')
plt.show()