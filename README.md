# 💳 FraudDetectionSyetem-
과거의 결재 정보를 이용하여 이상거래 여부를 인공지능으로 판단하고자 한다. 향후 모든 거래에 대하여 이상거래 여부를 모니터링 하여 나은 핀테크 서비스를 준비하기 위하여 구현해 보았다.

<br>

## 프로젝트 기간
2026.01.11.Sun ~ 2023.01.14.Wed (4 days)

## 프로젝트 정의
- 신용을 바탕으로 하는 금융거래 중 하나는 신용카드결제입니다.
- (주)김한카드는 최근 콜센터 내용을 분석한 결과 신용카드 결재가 본인이 아니라는 주장이 증가하는 추세를 파악하였습니다.
- 주장의 내용에 대한 민원처리 결과를 검토해 보니, 실제로 카드 소유주가 구매한 물건이 아닌 거래가 있었습니다.
- 이에 과거의 결재 정보를 이용하여 이상거래 여부를 인공지능으로 판단하고자 합니다.
- 향후 모든 거래에 대하여 이상거래 여부를 판단하여 경고할 수 있도록 서비스를 준비하기 위한 과정입니다.

## 데이터 세트
- 데이터 세트명 : Credit Card Fraud Detection
- 데이터 세트 출처 : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 데이터 세트 상세 설명
- 데이터는 284,807건이며, 총 31개의 컬럼으로 구성됨
- 이상거래여부(`Class`), 거래시간(`Time`)과 거래금액(`Amount`)을 제외하고 나머지 값은 차원축소(PCA)된 값입니다.
  - `Time` : Unix 시스템으로 작성된 거래일시
  - `V1` ~ `V28` : 원본 데이터를 차원축소한 값
  - `Amount` : 거래금액
  - `Class` : 이상거래여부 (0-정상, 1-이상)

## 모델의 성능 지표 (Metric) : ROC_AUC (Receiver Operating Characteristic, Area Under the Curve)
- 이진분류 문제의 예측력을 측정하기 위하여 사용
- 관심있는 데이터라고 판단하는 임계값을 조정하여 결과를 바꿀 수 있는 측정방법을 보완

<br>

## 분석 과정
1. 데이터 전처리
- Missing Data check
- Data Scaling
  - Log Transform
  - Min-Max Scaling

2. 탐색적 자료 분석(EDA, Exploratory Data Analysis)
- Partial Least Squares(PLS) Regression
  - 2 Components, 3 Components
  - VIP Scores from PLS Regression
- Feature Selection (Top 10)
  - Pearson Correlation
  - Spearman Rank Correlation
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
  - Mutual Information Score(MIS)
  - PLS - VIP Score

3. Modeling_ Ensemble modeling - `VotingClassifier`
- `LogistRegression`, `RandomForestClassifier`, `LGBMClassifier`, `XGBClassifier`
- `예측 -> 검증 -> 변수중요도 확인`
- Comparison Analysis
  - 기존의 단일 모델에 비해 앙상블 모델의 성능을 체크함.

4. Fine Tunning

5. XAI
- Global XAI
- Local XAI
- SHAP Importance Comparison

## 최종 코드 및 결과

