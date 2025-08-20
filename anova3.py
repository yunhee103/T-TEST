#  온도(세 개의 집단)에 따른 매출액의 평균차이 검정
#  공통 칼람이 년월일인 두개의 파일을 조합을 해서 작업

#  귀무 : 온도에 따른 음식점 매출액 평균 차이는 없다.
#  대립 : 온도에 따른 음식점 매출액 평균 차이는 있다.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon


#  매출 자료 읽기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(3)) # 328 entries, 0 to 327
print(sales_data.info())

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(3)) #  702 entries,  9 columns
print(wt_data.info())

# sales 데이터의 날짜를 기준으로 두개의 자료를 병합 작업 진행
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-',''))
print(wt_data.head(3))
frame = sales_data.merge(wt_data, how='left',left_on='YMD',right_on='tm')
print(frame.head(3), '', len(frame))
print(frame.columns) #(['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes']
data = frame.iloc[:,[0,1,7,8]]
print(data.head(3))

print(data.maxTa.describe())
# 일별 최고온도(연속형) 변수를 이용해 명목형(구간화) 변수 추가
data['ta_gubun'] = pd.cut(data.maxTa, bins=[-5,8,24,37], labels=[0,1,2])
print(data.head(3))
print(data.ta_gubun.unique())
print(data.isnull().sum())

# 최고 온도를 세 그룹으로 나눈 뒤, 등분산/정규성 검정
x1 = np.array(data[data.ta_gubun == 0].AMT)
x2 = np.array(data[data.ta_gubun == 1].AMT)
x3 = np.array(data[data.ta_gubun == 2].AMT)
print(x1[:5], len(x1))
print(stats.levene(x1,x2,x3).pvalue) # 0.0390 < 0.05 등분산만족x
print(stats.shapiro(x1).pvalue)
print(stats.shapiro(x2).pvalue)
print(stats.shapiro(x3).pvalue)

# 0.2481924204382751
# 0.03882572120522948
# 0.3182989573650957
# 정규성은 어느정도 만족 (2개)


spp = data.loc[:,['AMT', 'ta_gubun']]
print(spp.groupby('ta_gubun').mean())
print(pd.pivot_table(spp, index=['ta_gubun'], aggfunc='mean'))

# anova진행
sp = np.array(spp)
group1 = sp[sp[:, 1]==0, 0]
group2 = sp[sp[:, 1]==1, 0]
group3 = sp[sp[:, 1]==2, 0]

print(stats.f_oneway(group1,group2,group3).pvalue)   # 2.360737101089604e-34  < 0.05 귀무기각
# 참고 : 등분산성을 만족 X  -> Welch's anova test 
# pip install pingouin

from pingouin import welch_anova
print(welch_anova(dv='AMT', between='ta_gubun', data=data))
#   7.907874e-35 < 0.05 귀무기각


# 참고 : 정규성을 만족 X  -> Kruskal wallis test
print('kruskal : ' , stats.kruskal(group1,group2,group3))
# pvalue=np.float64(1.5278142583114522e-29)) < 0.05 귀무기각    
# 온오데 따라 매출액의 차이가 유의미하다.

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult = pairwise_tukeyhsd(endog=spp['AMT'], groups=spp['ta_gubun'])
print(turkyResult)
import matplotlib.pyplot as plt
turkyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()



# 문제 ) 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
# 귀무 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.
# 대립 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 있다.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

data = pd.read_csv('ANOVA1.csv')
print(data, type(data), data.shape)
print(data.info())

data['quantity'] = data.groupby('kind')['quantity'].transform(lambda x: x.fillna(x.mean()))
# print(data, type(data), data.shape)
# print(data.info())
# print(data.groupby('kind').mean())   # kind별 평균 확인

gr1 = data[data['kind'] ==1]['quantity']
gr2 = data[data['kind'] ==2]['quantity']
gr3 = data[data['kind'] ==3]['quantity']
gr4 = data[data['kind'] ==4]['quantity']

f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3, gr4)
print('f_statistic :', f_statistic)
print('p_value : ', p_value)

# p_value :  0.8089979993442262  > 0.05  귀무가설 채택.


# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult = pairwise_tukeyhsd(endog=data.kind, groups=data.quantity)
print(turkyResult)

turkyResult.plot_simultaneous(xlabel='kind', ylabel='quantity')
plt.show()
plt.close()



# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

import numpy as np
import pandas as pd
import scipy.stats as stats
import MySQLdb
import csv
from scipy import stats
import numpy as np
import MySQLdb

# 귀무가설: 각 부서별로 연봉의 차이가 없다.
# 대립가설: 각 부서별로 연봉의 차이가 있다.

conn = MySQLdb.connect(
    host='localhost',
    user='root',
    passwd='1234',
    db='mydb',
    port=3306,
    charset='utf8'
)

sql = """
    SELECT b.busername, j.jikwonpay
    FROM jikwon AS j
    JOIN buser AS b ON j.busernum = b.buserno
    WHERE b.busername IN ('총무부', '영업부', '전산부', '관리부');
    """
df = pd.read_sql(sql, conn)
df.columns = ['부서명', '연봉']
print(df)
buser_group = df.groupby('부서명')['연봉'].mean()
print(buser_group)

gwanli = df[df['부서명'] == '관리부']['연봉']
yeongup = df[df['부서명'] == '영업부']['연봉']
jeonsan = df[df['부서명'] == '전산부']['연봉']
chongmu = df[df['부서명'] == '총무부']['연봉']

f_statistic, p_value = stats.f_oneway(gwanli, yeongup, jeonsan, chongmu)
print('f_statistic:', f_statistic)
print('p_value: ', p_value)
