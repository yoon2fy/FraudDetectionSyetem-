# 결과를 정리한 페이지입니다.
## 1. 결과 정리
앙상블 모델을 수행한 결과 98%의 성능을 보였음. 이는 기존의 단일 머신러닝 모델보다 더 높은 정확도였음. feature selection의 경우 Top10을 선별하여 진행하였음.

## 2. 향후 계획
앞으로는 Fine Tunning을 하여 99%까지 성능을 올리려고 함.
그리고 목표치(ROC 99%)를 달성하게 된다면, XAI를 통해 모델의 해석까지 진행을 하려고 함.

## 3. 방법론 및 결과 정리
1. 데이터 전처리
- 계획대로 모두 수행하였음. 이상치의 경우 데이터의 특성 상 삭제하지 아니하였음.

1. 탐색적 자료 분석(EDA, Exploratory Data Analysis)

Partial Least Squares(PLS) Regression

<img width="600" alt="image" src="https://github.com/user-attachments/assets/148ca92b-3cd7-4028-8048-019edcc392ab" />
<img width="600" alt="image" src="https://github.com/user-attachments/assets/819d3f79-76c4-4d5d-a11d-6b9b5800f07c" />
2 Components, 3 Components

<img width="600" alt="image" src="https://github.com/user-attachments/assets/0da4afd7-aa57-4210-8b33-e1939a908d13" />
VIP Scores from PLS Regression

<img width="600" alt="image" src="https://github.com/user-attachments/assets/5212d995-78dc-45d1-b57a-985c67a3db73" />
Feature Selection (Top 10)

--> # 사용할 feature 목록

```
selected_features = ['V17', 'V14', 'V12', 'V10', 'V7',
                     'V16', 'V3', 'V11', 'V4', 'V9'
```

3. Modeling_ Ensemble modeling - `VotingClassifier`

`ROC AUC Score =  0.9806730568868514`

4. Comparison Analysis

단일로는 lgb를 했는데, `ROC-AUC: 0.9591`가 나왔음
이번에 개발한 앙상블 모델이 성능이 훨씬 좋았음.

5. Fine Tunning

현재 목표치인 95% 보다 성능이 매우 좋은 98%가 나왔음. 다음 버전에는 99%를 목표로 튜닝을 하려고 함.

7. XAI
Global XAI
<img width="600" alt="image" src="https://github.com/user-attachments/assets/0c98b711-3783-491d-ae5a-d07fbc5314c9" />

Local XAI

<img width="600" alt="image" src="https://github.com/user-attachments/assets/18c8dd57-64ee-40ac-8187-a7adc6ae1026" />
<img width="600" alt="image" src="https://github.com/user-attachments/assets/db39308c-d030-41d0-8b2d-79fc97890c28" />

<img width="600" alt="image" src="https://github.com/user-attachments/assets/88ed3496-673f-4077-a49a-b76b8b5b9f94" />
- False Positive 사례에서는 V17, V14 값이 사기 거래에서 자주 나타나는 패턴과 유사하게 작용하여 모델이 사기 거래로 판단하였다. 그러나 실제 라벨은 정상 거래로, 이는 특정 feature 조합이 정상 거래에서도 극단적으로 나타날 수 있음을 시사한다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/513da0f4-86d4-4c85-87bf-e0bfdf232072" />
- False Negative 사례에서는 V17, V12 등의 주요 feature가 사기 거래를 강하게 시사할 만큼 극단적인 값을 가지지 않아 모델이 정상 거래로 오판하였다. 이는 사기 패턴이 점점 정상 거래와 유사해지는 경우 모델이 탐지에 실패할 수 있음을 보여준다.

SHAP Importance Comparison
<img width="600" alt="image" src="https://github.com/user-attachments/assets/7662fd9f-c50b-4093-8316-f2a08fb66671" />



