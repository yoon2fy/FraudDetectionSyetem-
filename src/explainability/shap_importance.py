# 앙상블 예측 확률
y_pred_prob = final_model.predict_proba(x_valid)[:, 1]

# 임계값 (기본 0.5)
threshold = 0.5
y_pred_label = (y_pred_prob > threshold).astype(int)

# False Positive / False Negative 인덱스
fp_idx = np.where((y_valid == 0) & (y_pred_label == 1))[0]
fn_idx = np.where((y_valid == 1) & (y_pred_label == 0))[0]
len(fp_idx), len(fn_idx)


explainer = shap.TreeExplainer(model_4)
shap_values = explainer.shap_values(x_valid)

fp_sample = fp_idx[0]
