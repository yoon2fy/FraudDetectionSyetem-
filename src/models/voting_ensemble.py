## 앙상블 모델 개발
### Model 1 학습
params = {
    'solver': 'liblinear',
    'random_state': 42,
}
model_1 = LogisticRegression(**params)
model_1.fit(x_train, y_train)

### Model 2 학습
params = {
    'criterion': 'entropy',
    'max_depth': 30,
    'random_state': 42,
}
model_2 = RandomForestClassifier(**params)
model_2.fit(x_train, y_train)

# Model 3 학습
params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'num_leaves': 30,
    'objective': 'binary',
    'random_state': 42,
    # 'boost_from_average': False,  # 불균형 데이터 학습의 경우 필수 적용 v2.1.0 이상
}

model_3 = lgb.LGBMClassifier(**params)
model_3.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='auc')

# Model 4 학습
params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'random_state': 42,
    'eval_metric': 'auc'
}

model_4 = XGBClassifier(**params)
model_4.fit(
    x_train, y_train,
    eval_set=[(x_valid, y_valid)],
    verbose=False
)

# 앙상블 모델 생성
# hard vote (Majority Voting) : 모델로부터 가장 많은 표를 얻은 클래스 예측
# soft vote (Probability Voting) : 모델에서 합산 ​​확률이 가장 큰 클래스 예측
final_model = VotingClassifier(estimators=[('lr', model_1), ('rf', model_2), ('lgbm', model_3),  ('xgb', model_4)], voting='soft')
final_model.fit(x_train, y_train)

# 예측
y_pred = final_model.predict_proba(x_valid)[:, 1]
valid_data['pred'] = y_pred
valid_data

# 검증
score = roc_auc_score(y_valid, y_pred)
print('ROC AUC Score = ', score) # ROC AUC Score =  0.9806730568868514

# 변수 중요도
val_imp = pd.DataFrame(model_2.feature_importances_, index=model_2.feature_names_in_, columns=['imp'])
val_imp

# 변수 중요도 시각화
val_imp['imp'].plot(kind='bar')

# 저장 객체 정의
save_object = [final_model, params, valid_data]

# 저장
with open(file='my_model.pickle', mode='wb') as f:
    pickle.dump(save_object, f)

# 저장된 객체 불러오기
with open(file='my_model.pickle', mode='rb') as f:
    load_object = pickle.load(f)

# 저장된 객체 분리
final_model = load_object[0]
params = load_object[1]
valid_data = load_object[2]

# 예측
valid_data['pred'] = final_model.predict_proba(x_valid)[:, 1]
valid_data

# 검증
score = roc_auc_score(valid_data['Class'], valid_data['pred'])
print('ROC AUC Score = ', score)
