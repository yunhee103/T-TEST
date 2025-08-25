# 외국인 대상 국내 주요 관광지 방문 상관관계 분석
import json 
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family='Malgun Gothic')
import pandas as pd
import numpy as np

# scatter Graph 작성
def setScatterGraph(tour_table, all_table, tourPoint):
    # print(tourPoint)
    # 계산할 관광지 명에 해당하는 자료만 뽑아 별도 저장하고, 외국인 관광 자료와 병합
    tour = tour_table[tour_table['resNm'] == tourPoint]
    # print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    # print(merge_table)
    fig = plt.figure()
    fig.suptitle(tourPoint + '상관관계분석')

    plt.subplot(1,3,1)
    plt.xlabel('중국인 입국수')
    plt.ylabel('중국인 입장 객수')
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    print('r1:' , r1)
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, s=6, c='red')

    plt.subplot(1,3,2)
    plt.xlabel('일본인 입국수')
    plt.ylabel('일본인 입장 객수')
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    print('r2:' , r2)
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, s=6, c='blue')
    
    plt.subplot(1,3,3)
    plt.xlabel('미국인 입국수')
    plt.ylabel('미국인 입장 객수')
    lambda3 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lambda3(merge_table)
    print('r3:' , r3)
    plt.title('r={:.5f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], alpha=0.7, s=6, c='green')
    plt.tight_layout()
    plt.show()

    return [tourPoint, r1, r2, r3]

def chulbal():
    # 서울시 관광지 정보 읽어 DataFrame으로 저장
    fname = '서울특별시_관광지입장정보_2011_2016.json'
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm', 'resNm', 'ForNum'))
    tour_table = tour_table.set_index('yyyymm')
    # print(tour_table)
    resNm = tour_table.resNm.unique()   # 관광지 이름
    # print('resNm : ', resNm[:5])
    # 중국인 관광 정보를 읽어 dataFrame으로 저장
    cdf = '중국인방문객.json'
    jdata = json.loads(open(cdf, 'r', encoding='utf-8').read())
    china_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    china_table = china_table.rename(columns={'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    # print(china_table[:2])

    jdf = '일본인방문객.json'
    jdata = json.loads(open(jdf, 'r', encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    # print(japan_table[:2])

    udf = '미국인방문객.json'
    jdata = json.loads(open(udf, 'r', encoding='utf-8').read())
    usa_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    # print(usa_table[:2])


    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    r_list = [] # 각 관광지별 상관계수 기억
    for tourPoint in resNm[:5]:
        # print(tourPoint)
        # 각 관광지별 상관계수와 그래프 그리기
        r_list.append(setScatterGraph(tour_table, all_table, tourPoint))

        # r_list로 DataFrame 작성
    r_df = pd.DataFrame(r_list, columns=('고궁명', '중국', '일본', '미국'))
    r_df = r_df.set_index('고궁명')
    print(r_df)

    r_df.plot(kind='bar', rot = 50)
    plt.show()
    plt.close()


if __name__ == '__main__':
    chulbal()