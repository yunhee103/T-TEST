# https://github.com/pykwon/python 에 있는 Advertising.csv 파일을 읽어 tv,radio,newspaper 간의 상관관계를 파악하시오. 

# 그리고 이들의 관계를 heatmap 그래프로 표현하시오.

import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import seaborn as sns

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv')
# print(data.head(3))
data = data[['tv', 'radio', 'newspaper']]
corr = data.corr()
# print(corr)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()