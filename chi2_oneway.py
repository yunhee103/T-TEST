# 일원 카이제곱 검정 : 변인이 1개
# 적합도 (선호도) 검정
# 실험을 통해 얻은 관찰값들이 어떤 이론적 분포를 따르고 있는지 확인하는 검정
# # 꽃색깔의 표현 분리 비율이 3:1이 맞는가?

# 적합도 검정실습
# 주사위를 60회 던져서 나온 관측도수 기대도수가 아래와 같이 나온 경우에 이 주사위는 적합한지


# 가설

# 귀무가설 : 기대치와 관찰치는 차이가 없다.(주사위는 게임에 적합하다.)
# 대립가설 : 기대치와 관찰치는 차이가 있다.(주사위는 게임에 적합하지 않다.)


import pandas as pd
import scipy.stats as stats
data = [4,6,17,16,8,9]  #관측값
exp  = [10,10,10,10,10,10] #기대값

print(stats.chisquare(data))
# Power_divergenceResult(statistic=np.float64(14.200000000000001), pvalue=np.float64(0.014387678176921308))
# 카이 제곱 : 14.20 , p-value = 0.014
# # 결론 : p-value (0.01438) < 유의 수준 (0.05)   -> 귀무기각 
# 주사위는 게임에 적합하지 않다
# 관측값은 우연히 발생 한 것이 아니라 어떠한 원인에 의해 얻어진 값이다
print(stats.chisquare(data,exp))
result = stats.chisquare(data,exp)
print('chi2:' , result[0])
print('p-value' , result[1])
print('-----------------------------------------')
# 선호도 분석 실습 5개의 스포츠 음료에 대한 선호도 차이가 있는지 검정

# 귀무가설(H0​): 5개 스포츠 음료의 선호도에 차이가 없다. 즉, 5개 음료가 동일한 비율로 선택될 것으로 가정합니다.
# 대립가설(H1): 5개 스포츠 음료의 선호도에 차이가 있다. 즉, 5개 음료 중 특정 음료가 더 선호될 수 있습니다.
sdata = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv')
print(sdata)
print(stats.chisquare(sdata['관측도수']))  #Power_divergenceResult(statistic=np.float64(20.488188976377952), pvalue=np.float64(0.00039991784008227264))
#  결론 : p-value (0.00039) < 유의 수준 (0.05)   -> 귀무기각 -> 선호도에 차이가 있다.


#  시각화 : 어떤 음료가 기대보다 많이 선택했는지 확인
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='malgun gothic') # matplotlib에서 한글 폰트(맑은 고딕) 사용 설정
plt.rcParams['axes.unicode_minus'] = False


# 기대도수
total = sdata['관측도수'].sum()
expected = [total /len(sdata)] * len(sdata)
print('expected : ', expected)

x = np.arange(len(sdata))
width = 0.35 # 막대너비
plt.figure(figsize=(9,5))
plt.bar(x-width/2, sdata['관측도수'], width=width, label='관측도수')
plt.bar(x-width/2, expected, width=width, label='기대도수', alpha=0.6)
plt.xticks(x, sdata['음료종류'])
plt.xlabel('음료종류')
plt.ylabel('도수')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# #그래프와 카이제곱 검정 결과를 바탕으로 어떤 음료가 더 인기있는지 구체적으로 분석
# 총합과 기대도수 이미구함
# 차이계산
sdata['기대도수'] = expected
sdata['차이(관측-기대)'] = sdata['관측도수']  - sdata['기대도수']
sdata['차이비율(%)'] = round(sdata['차이(관측-기대)'] / expected * 100, 2)
print(sdata.head(3))
sdata.sort_values(by='차이(관측-기대)', ascending=False, inplace=True)
sdata.reset_index(drop=True, inplace=True)
print(sdata)