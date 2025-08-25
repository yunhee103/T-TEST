# 선형회귀 모델식 계산 - 최소제곱법(ols)으로 w=wx+b 형태의 추세식 파라미터 w와 b를 추정

import numpy as np

class MySimpleLinearRegrression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self , x:np.ndarray, y:np.ndarray):  #학습
        # ols로 w & b를 추정
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x-x_mean)*(y-y_mean))
        denominator = np.sum((x-x_mean)**2)
        self.w = numerator / denominator
        self.b = y_mean - (self.w * x_mean)


    def predict(self, x:np.ndarray):
        return self.w * x + self.b        


def main():
    np.random.seed()
    # 임의의 성인 남성 10명의 키, 몸무게 자료를 사용
    # 가우시안 분포 
    x_heights = np.random.normal(175, 5, 10)
    y_weights = np.random.normal(70, 10, 10)
    # 최소 제곱법을 수행하는 클래스 객체 생성 후 학습
    model = MySimpleLinearRegrression()
    model.fit(x_heights,y_weights)
    
    # 추정된 w 와 b출력
    print('w :' , model.w)
    print('b :' , model.b)

    # 예측값 확인
    y_pred = model.predict(x_heights)
    print(y_pred)

    print('실제 몸무게와 예측 몸무게 비교')
    for i in range(len(x_heights)):
        print(f"키:{x_heights[i]:.2f}cm, 실제 몸무게 : {y_weights[i]:.2f}kg, 예측 몸무게 :{y_pred[i]:.2f}kg")

    print("미지의 남성 키 199의 몸무게는?", model.predict(199))
    

if __name__ == "__main__":
    main()

