# 이원카이제곱검정 - 교차분할표를 사용
# 변인이 두개 - 독립성 또는 동질성
# - 독립성(관련성)검정- 동일 집단의 두 변인 학력수준과 대학진학 여부 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.

# 실습 : 교육 수준과 흡연율 간의 관련성 분석 smoke csv
# 귀무 : 교육수준과 흡연율 간의 관련이 없다 (독립이다, 연관성이 없다)
# 대립 : 교육수준과 흡연율 간의 관련이 있다 (독립이 아니다, 연관성이 있다)

import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv")
print(data.head(2))
print(data['education'].unique()) #[1 2 3]
print(data['smoking'].unique()) #[1 2 3]

#  학력별 흡연 인원수를 위한 교차표
ctab = pd.crosstab(index=data['education'], columns=data['smoking'])

# ctab = pd.crosstab(index=data['education'], columns=data['smoking'], normalize=True) 
print(ctab)
ctab.index = ["대학원졸", "대졸", "고졸"]
ctab.columns = ["과흡연" , "보통", "노담"]
print(ctab)
chi2, p , dof, _ = stats.chi2_contingency(ctab)
msg = "test statics:{},, p-value:{}, df:{}"
format(msg.format(chi2, p , dof))
print(msg.format(chi2,p,dof))
# test statics:18.910915739853955,, p-value:0.0008182572832162924, df:4

# 결론 : p-value(0.000818)  < 0.05 이므로 귀무가설 기각.
# 따라서 교육수준과 흡연율 간의 관련이 있다.

print('음료 종류와 성별간의 선호도 차이 검정')
# 남성과 여성의 음료 선호는 서로 관련이 있을까? 없을까?

# 귀무 : 성별과 음료 선호는 서로 관련이 없다. (성별에 따라 선호가 같다)
# 대립 : 성별과 음료 선호는 서로 관련이 있다. (성별에 따라 선호가 다름)

data = pd.DataFrame({
    '게토레이' : [30, 20],
    '포카리' : [20, 30],
    '비타500' : [10, 30]
}, index = ['남성', '여성'])

print(data)
chi2, p , dof, expected = stats.chi2_contingency(data)
print("카이제곱 표본 통계량 :", chi2)
print("유의확률 (p값):", p)
print("자유도 :", dof)
print("기대도수 :", expected)

"""
카이제곱 표본 통계량 : 11.375
유의확률 (p값): 0.003388052521834713
자유도 : 2
기대도수 : [[21.42857143 21.42857143 17.14285714]
 [28.57142857 28.57142857 22.85714286]]

"""

#  결론 : p값이 0.05보다 작으므로 귀무기각 즉, 성별에 따라 선호가 다름

#  시각화 : heatmap
#  히트 맵은 색상을 활용해 값의 분포를 보여주는 그래프
#  히스토그램이 하나의 변수에 대한 강도(높이)를 활용할 수 있다면, 컬러맵은 색상을 활용해 두개의 기준에 따른 강도를 보여줌

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
sns.heatmap(data, annot=True, fmt='d', cmap='Blues')
plt.title('성별에 따른 음료 선호(빈도)')
plt.xlabel('음료')
plt.ylabel('성별')
plt.show()


