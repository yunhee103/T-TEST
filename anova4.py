# 이원분산분석(Two-way ANOVA)은 두 개 이상의 독립 변수가 종속 변수의 평균 미치는 영향을 동시에 분석하는 통계적 방법. 
# 주로 두 독립 변수의 주 효과와 이들 간의 상호작용 효과를 검정하는 데 사용됩니다.
# 가설이 주효과2개, 교호작용 1개가 나옴

# 교호작용 (interaction term): 한 쪽 요인이 취하는 수준에 따라 다른 쪽 요인이 영향을 받는 요인의 조합효과를 말하는 것으로 상승과 상쇄효과가 있다.
# ex)  초밥-간장, 감자튀김-간장, 초밥과 케찹... 상승효과 / 상쇄효과


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 실습 1 ) 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향을 주는가?
# 귀무 : 태아 수와 태아의 머리둘레 평균은 차이가 없다.
# 대립 : 태아 수와 태아의 머리둘레 평균은 차이가 있다.
# 귀무 : 태아 수와 관측자 수의 머리둘레 평균은 차이가 없다.
# 대립 : 태아 수와 관측자 수의 머리둘레 평균은 차이가 있다.
# 교호작용 가설
# 귀무 : 교호작용이 없다. (태아수와 관측자수는 관련이 없다.)
# 대립 : 교호작용이 있다. (태아수와 관측자수는 관련이 있다.)

url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(url)
print(data.head(3),data.shape)
print(data['태아수'].unique())
print(data['관측자수'].unique())
# 이원분산분석(Two-way ANOVA)을 수행하기 위한 OLS(Ordinary Least Squares) 모델을 정의합니다.
# '머리둘레'가 종속 변수(y), '태아수'와 '관측자수'가 독립 변수(x)입니다.
# C()는 범주형 변수(Categorical Variable)임을 명시합니다.
# 교호작용(interaction)을 포함하지 않는 모델: 두 변수의 주 효과만 검정
# C(태아수) + C(관측자수)
reg = ols("머리둘레 ~ C(태아수) + C(관측자수)", data=data).fit()  #교호작용 확인 x
reg = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수) ", data=data).fit() 
reg = ols("머리둘레 ~ C(태아수) * C(관측자수) ", data=data).fit()  #교호작용 확인 o  위와 같은 값
# 교호작용을 명시적으로 추가한 모델: 두 변수의 주 효과와 상호작용 효과 모두 검정
# C(태아수):C(관측자수)가 상호작용을 의미

# anova_lm 함수를 사용하여 분산분석표(ANOVA table)를 생성합니다.
# type=2는 II형(Type 2) 분산분석을 수행하도록 지정합니다.
result = anova_lm(reg, type=2)
print(result)


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 실습 1) 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향을 주는가?
# 귀무 가설: 태아 수에 따라 머리둘레 평균에 차이가 없다.
# 대립 가설: 태아 수에 따라 머리둘레 평균에 차이가 있다.
# 귀무 가설: 관측자 수에 따라 머리둘레 평균에 차이가 없다.
# 대립 가설: 관측자 수에 따라 머리둘레 평균에 차이가 있다.
# 교호작용 가설
# 귀무 가설: 교호작용이 없다. (태아 수와 관측자 수는 머리둘레에 미치는 영향이 독립적이다.)
# 대립 가설: 교호작용이 있다. (태아 수와 관측자수는 서로 영향을 미치며 머리둘레에 영향을 준다.)

url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(url) # URL에서 데이터를 읽어와 DataFrame으로 변환
print(data.head(3), data.shape) # 데이터의 상위 3개 행과 전체 데이터의 행, 열 개수 출력
print(data['태아수'].unique()) # '태아수' 컬럼의 고유한 값들을 출력
print(data['관측자수'].unique()) # '관측자수' 컬럼의 고유한 값들을 출력

# 이원분산분석(Two-way ANOVA)을 수행하기 위한 OLS(Ordinary Least Squares) 모델을 정의합니다.
# '머리둘레'가 종속 변수(y), '태아수'와 '관측자수'가 독립 변수(x)입니다.
# C()는 범주형 변수(Categorical Variable)임을 명시합니다.

# 교호작용(interaction)을 포함하지 않는 모델: 두 변수의 주 효과만 검정
# C(태아수) + C(관측자수)
reg = ols("머리둘레 ~ C(태아수) + C(관측자수)", data=data).fit()

# 교호작용을 명시적으로 추가한 모델: 두 변수의 주 효과와 상호작용 효과 모두 검정
# C(태아수):C(관측자수)가 상호작용을 의미
reg = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)", data=data).fit() 

# 축약된 표현으로 교호작용을 포함한 모델을 정의합니다. 위 코드와 동일한 결과를 반환합니다.
# * 기호는 주 효과와 모든 가능한 상호작용 항을 자동으로 포함합니다.
reg = ols("머리둘레 ~ C(태아수) * C(관측자수)", data=data).fit()

# anova_lm 함수를 사용하여 분산분석표(ANOVA table)를 생성합니다.
# type=2는 II형(Type 2) 분산분석을 수행하도록 지정합니다.
result = anova_lm(reg, type=2)
print(result) # 분산분석 결과 출력


# ANOVA Type 1, 2, 3 설명
# 분산분석(ANOVA)에서 type은 **제곱합(Sum of Squares, SS)**을 계산하는 방법을 의미하며, 주로 불균형 데이터(unbalanced data)가 있을 때 결과에 영향을 줍니다. 
# 균형 데이터(balanced data)에서는 세 가지 유형 모두 동일한 결과를 반환합니다.
# Type 1 (Sequential Sum of Squares)
# 설명: 모델에 변수를 순차적으로 추가하면서 각 변수가 추가될 때마다 설명되는 변동(variance)을 계산합니다.
# 특징: 변수의 순서에 따라 결과가 달라집니다. 따라서 분석가가 변수들을 이론적 중요도나 논리적 순서에 따라 순서를 정할 때 사용됩니다.
# 용도: 계층적 회귀 분석(hierarchical regression)과 유사한 방식으로 사용됩니다.

# Type 2 (Partial Sum of Squares)
# 설명: 다른 모든 주 효과(main effects)를 고려한 후 특정 주 효과에 대한 제곱합을 계산합니다. 상호작용 효과는 고려하지 않습니다.
# 특징: 주 효과의 순서에 영향을 받지 않습니다. 상호작용 효과가 유의미하지 않다고 가정할 때 사용하기 적합합니다.
# 용도: 불균형 데이터에서 주 효과를 검정할 때 주로 사용됩니다.

# Type 3 (Partial Sum of Squares)
# 설명: 다른 모든 주 효과와 상호작용 효과를 고려한 후 특정 효과(주 효과 또는 상호작용 효과)에 대한 제곱합을 계산합니다.
# 특징: 변수의 순서에 영향을 받지 않으며, 모든 다른 항들을 고려합니다. 가장 보수적인 방법으로, 상호작용 효과가 유의미할 때 주 효과를 해석하는 데 사용됩니다.
# 용도: 불균형 데이터에서 주 효과와 상호작용 효과를 모두 검정할 때 가장 널리 사용되는 방법입니다.
# 결론 : 대부분의 통계 패키지에서는 불균형 데이터를 처리할 때 Type 2 또는 Type 3을 기본값으로 사용합니다. 특히 상호작용 효과가 중요한 연구에서는 Type 3이 더 적합할 수 있습니다. 


#                  df      sum_sq     mean_sq            F        PR(>F)
# C(태아수)           2.0  324.008889  162.004444  2113.101449  1.051039e-27            < 0.05  귀무 기각     
# C(관측자수)          3.0    1.198611    0.399537     5.211353  6.497055e-03           > 0.05 귀무 채택
# C(태아수):C(관측자수)   6.0    0.562222    0.093704     1.222222  3.295509e-01        > 0.05 귀무 채택

# 태아수는 머리 둘레에 강력한 영향을 미침. 관측자 수는 유의한 영향을 미침.
# 하지만 태아수와 관측자 수의 상호작용은 유의하지 않다.


# 실습 2) poiston종류와 treat(응급처치)가 독퍼짐 시간의 평균에 영향을 주는가?
# 주효화 가설
# 귀무 : poison 종류와 독퍼짐 시간의 평균에 차이가 없다. 
# 대립 : poison 종류와 독퍼짐 시간의 평균에 차이가 있다. 
# 귀무 : treat(응급처치) 방법과 독퍼짐 시간의 평균에 차이가 없다. 
# 대립 : treat(응급처치) 방법과 독퍼짐 시간의 평균에 차이가 있다. 

# 교호작용 가설
# 귀무 : 교효작용이 없다. (poison 종류와 treat(응급처치) 방법은 관련이 없다.)
# 대립 : 교효작용이 있다. (poison 종류와 treat(응급처치) 방법은 관련이 있다.)
data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv", index_col=0)
print(data2.head(3), data2.shape)

print(data2.groupby('poison').agg(len))
print(data2.groupby('treat').agg(len))
print(data2.groupby(['poison', 'treat']).agg(len))
# 모든 집단 별 표본 수가 동일하므로 균형설계가 잘 되었다.

result2 = ols('time ~ C(poison) * C(treat)', data=data2).fit()
print(anova_lm(result2))

#                       df    sum_sq   mean_sq          F        PR(>F)
# C(poison)            2.0  1.033012  0.516506  23.221737  3.331440e-07         < 0.05 귀무 기각
# C(treat)             3.0  0.921206  0.307069  13.805582  3.777331e-06         < 0.05 귀무 기각
# C(poison):C(treat)   6.0  0.250138  0.041690   1.874333  1.122506e-01         > 0.05 상호작용효과는 없다.

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult1 = pairwise_tukeyhsd(endog=data2.time, groups=data2.poison)
print(turkyResult1)

turkyResult2 = pairwise_tukeyhsd(endog=data2.time, groups=data2.treat)
print(turkyResult2)

turkyResult1.plot_simultaneous(xlabel='mean', ylabel='poison')
turkyResult2.plot_simultaneous(xlabel='mean', ylabel='treat')
plt.show()
plt.close()




# reject 열은 귀무가설을 기각하는지 여부를 보여줍니다.

# reject=True: p-adj (조정된 p-value)가 유의 수준(FWER=0.05)보다 작으므로, 두 그룹의 평균에 통계적으로 유의미한 차이가 있다는 뜻. 
# 이는 두 그룹이 서로 비슷하지 않다는 의미.
# reject=False: p-adj가 유의 수준(FWER=0.05)보다 크므로, 귀무가설을 기각할 수 없습니다. 이는 두 그룹의 평균이 통계적으로 유의미한 차이가 없다는 뜻. 
# 즉, 두 그룹이 서로 비슷하다고 볼 수 있음.