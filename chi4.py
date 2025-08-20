# 이원카이제곱
# 동질성 :
# 검정 두 집단의 분포가 동일한가? 다른 분포인가? 를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로 동일한가를검정하게된다.
# 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된것인지 검정하는 방법이다.
# 동질성검정실습1) 교육방법에 따른 교육생들의 만족도 분석-동질성검정 survey_method.csv
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
print(data.head(3))
print(data['method'].unique())
print(set(data['survey']))

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족', '만족', '보통', '불만족', '매우불만족']
ctab.index = ['방법1', '방법2', '방법3']
print(ctab)

chi2, p, ddof, _ = stats.chi2_contingency(ctab)
msg = "Test statistic : {}, p-value : {}, df:{}"
print(msg.format(chi2, p, ddof))

# 해석 :  0.5864574 > 0.05 이므로 귀무가설 채택.

print('---------------------------------------')


# 동질성 검정실습2) 연령대별 sns 이용률의 동질성 검정
# # 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용현황을 조사한 자료를 바탕으로 연령대별로 홍보 전략을 세우고자 한다.
# 귀무 가설 : 연령대별로 sns서비스별 이용현황은 동일하다.
# 대립 가설 : 연령대별로 sns서비스별 이용현황은 동일하지않다.

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')

print(data2['age'].unique()) # [1 2 3]
print(data2['service'].unique()) # ['F' 'T' 'K' 'C' 'E']
# print(set(data2['service']))

ctab2 = pd.crosstab(index=data2['age'], columns=data2['service'])#, margins=True)
print(ctab2)
chi2, p, ddof, _ = stats.chi2_contingency(ctab2)
msg = "Test statistic : {}, p-value : {}, df:{}"
print(msg.format(chi2, p, ddof))
# Test statistic : 102.75202494484225, p-value : 1.1679064204212775e-18, df:8
# 해석 :   p-value : 1.1679064 > 0.05 이므로 귀무가설 채택.

# 사실 위 데이터는 샘플 데이터이다.
# 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본을 추출해 처리해보자.

sample_data = data2.sample(n=50, replace=True, random_state=1)
print(len(sample_data))
ctab3 = pd.crosstab(index=data2['age'], columns
                    =data2['service'])#, margins=True)
print(ctab3)
chi2, p, ddof, _ = stats.chi2_contingency(ctab3)
msg = "Test statistic : {}, p-value : {}, df:{}"
print(msg.format(chi2, p, ddof))