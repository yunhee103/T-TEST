from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
print(x[:3])
print(y[:3])

# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
# plt.show()

model = LogisticRegression().fit(x,y)
y_hat=model.predict(x)
print('y_hat : ', y_hat[:3])

f_value = model.decision_function(x) # 결정함수(판별함수, 불확실성 추정)
print('f_value :', f_value[:10])

df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=["f","y_hat", "y"])
print(df)

print(confusion_matrix(y, y_hat))


# 혼동 행렬의 각 값
#       	예측: 양성(Positive)	예측: 음성(Negative)
# 실제: 양성(Positive)	44 (TP)	4 (FN)
# 실제: 음성(Negative)	8 (FP)	44 (TN)

acc = (44+44)/100                   # (TP+TN)/(TP+TN+FP+FN)
recall = 44/ (44+4)                 # TP/(TP+FN)
precision = 44 / (44+8)             # TP/(TP+FP)
specificity = 44 / (44+8)           # TN/(TN+FP)
fallout = 4 / (44+8)                # FP/(TN+FP) 또는 1−Specificity
print('acc(정확도) : ' , acc)
print('recall(재현율) : ', recall)
print('precision(정밀도) : ', precision)
print('specificity(특이도) : ', specificity)
print('fallout(위양성율) : ', fallout)
print('fallout(위양성율) : ', 1 - specificity)
# 정리하면 TPR은 1에  근사하면 좋고 , FPR은 0에 근사하면 좋다.

print()
from sklearn import metrics
ac_sco = metrics.accuracy_score(y, y_hat)
print('ac_sco :' , ac_sco)
cl_rep = metrics.classification_report(y, y_hat)
print(cl_rep)
print()
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print('fpr :' ,fpr)
print('tpr :' ,tpr)
print('thresholds(분류임계결정값) :' ,thresholds)

# ROC 커브 시각화
# 기준선(Threshold)이 달라짐에 따라 분류 AI모델의 성능이 어떤지를 한눈에 볼 수 있다. 기준선에 따라 성능평가의 지표가 달라진다. 
# ROC는 위양성률(1-특이도)을 x축으로, 그에 대한 실제 양성률(민감도)을 y축으로 놓고 그 좌푯값들을 이어 그래프로 표현한 것이다. 
# 일반적으로 0.7~0.8 수준이 보통의 성능을 의미한다. 0.8~0.9는 좋음, 0.9~1.0은 매우 좋은 성능을 보이는 모델이라 평가할 수 있다.

plt.plot(fpr, tpr, 'o-', label = 'LogisticRegression')
plt.plot([0,1], [0,1], 'k--', label = 'random classfier line(AUC0.5)')
plt.plot([fallout], [recall], 'ro', ms=10) #위양성률과 재현율값 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# AUC (Area Under the Curve)- ROC커브의 면적
# 1에 가까울수록 좋은 분류 모델로 평가됨
print('AUC :' , metrics.auc(fpr, tpr))
