# 모델링
X_scaled
y
X_scaled.describe()

# 사용할 feature 목록
selected_features = ['V17', 'V14', 'V12', 'V10', 'V7',
                     'V16', 'V3', 'V11', 'V4', 'V9']

# feature selection 적용
X_selected = X_scaled[selected_features]

# 데이터 분할 (280,000 : 4,807)
x_train, x_valid, y_train, y_valid = train_test_split(
    X_selected,
    y,
    test_size=4807,
    random_state=42,
    stratify=y
)

x_train.shape, x_valid.shape, y_train.shape, y_valid.shape

X_selected.shape # (284807, 10) 이어야 정상


# 기본 파라미터 (baseline)
params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'num_leaves': 30,
    'objective': 'binary',
    'random_state': 42,
    'n_jobs': -1
    # 'boost_from_average': False  # LightGBM v2.1.0 이상 + 극단적 불균형일 때 고려
}

model = lgb.LGBMClassifier(**params) # 단일 머신러닝 모델 개

model.fit(
    x_train, y_train,
    eval_set=[(x_valid, y_valid)],
    eval_metric='auc',
    # verbose=100
)

# 확률 예측
y_pred = model.predict_proba(x_valid)[:, 1]

# valid 데이터프레임에 결과 추가
valid_data = x_valid.copy()
valid_data['Class'] = y_valid.values
valid_data['pred'] = y_pred

valid_data.head()

auc = roc_auc_score(y_valid, y_pred)
print(f"ROC-AUC: {auc:.4f}") # ROC-AUC: 0.9591

# 임계값 0.5 기준 성능:
print(classification_report(y_valid, (y_pred > 0.5).astype(int)))

# 변수 중요도
val_imp = pd.DataFrame(model.feature_importances_, index=model.feature_name_, columns=['imp'])
val_imp
