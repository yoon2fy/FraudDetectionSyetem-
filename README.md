# 0. 프로젝트 명
💳 FraudDetectionSyetem (가제)

## 프로젝트 기간
2026.01.11.Sun ~ 2023.01.14.Wed (4 days)

# 1. 문제 정의
## 1.1. 프로젝트 시나리오 개요
신용카드 거래는 신용을 기반으로 이루어지는 대표적인 금융 거래로, 거래의 신뢰성과 안전성은 카드사와 고객 모두에게 매우 중요한 요소이다. 특히 카드 소유주 본인이 아닌 제 3자에 의해 이루어지는 부정거래(Fraud Transaction)는 고객의 금전적 손실뿐 아니라 카드사에 대한 신뢰 저하로 이어질 수 있다.

카드사 입장에서는 정상 거래가 압도적으로 많기 때문에 단순 정확도(Accuracy)에 기반한 판단만으로는 부정 거래를 효과적으로 탐지하기가 어렵다. 실제로 소수의 부정 거래를 놓칠 경우 고객 피해로 직결되므로, 부정 거래를 얼마나 정확히 탐지할 수 있는지가 핵심 과제이다.

이에 본 프로젝트에서는 과거 신용카드 거래 정보를 활용하여, 각 거래가 정상 거래인지 부정 거래인지를 인공지능 모델로 판단하는 시스템을 구축하고자 한다. 이를 통해 향후 발생하는 모든 카드 결제 거래에 대해 실시간 이상 거래 탐지 및 사전 경고가 가능한 서비스 구축의 기초 단계를 마련하는 것을 목표로 한다. 특히 부정 거래 비율이 매우 낮은 특성을 고려하여, 단순 정확도가 아닌 Precision-Recall 곡선 기반 지표(AUPRC)를 중심으로 모델 성능을 평가하고, 실제 금융 서비스 환경에 적합한 이상 거래 탐지 모델을 구현하는 것을 핵심 문제로 정의하였다.

---

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

